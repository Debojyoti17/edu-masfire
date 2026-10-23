# Judge model for the redesigned pipeline: combines the label probabilities of the three LLM
# graders and the trained TF-IDF classifier into one final label with a confidence.
# It is a logistic regression trained only on the 2000-answer train sample, never on test data.
#
# Two regimes:
#   seen   - the question has teacher-graded answers in the train split (test_ua):
#            uses each LLM grader's "same" run
#   unseen - a new question (test_uq, test_ud): uses each LLM's "similar" and "zeroshot" runs;
#            its training simulates unseen questions by never letting a question appear on both
#            sides of a train/validation split.
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold

from scripts.run_tfidf_baseline import build_features, load_split
from scripts.run_pipeline import load_jsonl

LABELS = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant"]
MODELS = ["qwen2.5:7b", "mistral:7b", "gemma2:9b"]
REGIME_MODES = {"seen": ["same"], "unseen": ["similar", "zeroshot"]}
TRAIN_SET = "train_sample_2000"
TFIDF_C = 3.0
C_VALUES = [0.03, 0.1, 0.3, 1.0, 3.0]
TARGET_ACCURACY = 0.80
RAG_DIR = os.path.join("results", "rag")


def grader_probs(set_name, mode, model):
    """{task_id: probability list in LABELS order} for one grader run; missing answers get
    an even 0.25 each so the judge still has something to work with."""
    slug = model.replace(":", "-")
    out = {}
    for r in load_jsonl(os.path.join(RAG_DIR, f"{set_name}_{mode}_{slug}.jsonl")):
        p = r["probabilities"]
        out[r["task_id"]] = [p[label] for label in LABELS] if p else [0.25] * 4
    return out


def tfidf_train_oof(regime):
    """TF-IDF classifier probabilities for every train answer, each predicted by a model that
    never saw it (5 folds). For 'unseen', folds are split by question so the classifier never
    saw the question either. Cached in results/rag/tfidf_oof_<regime>.json."""
    path = os.path.join(RAG_DIR, f"tfidf_oof_{regime}.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    rows = load_split("train")
    y = np.array([r["label_name"] for r in rows])
    if regime == "seen":
        folds = StratifiedKFold(5, shuffle=True, random_state=42).split(rows, y)
    else:
        folds = GroupKFold(5).split(rows, y, groups=[r["question"] for r in rows])
    out = {}
    for train_idx, test_idx in folds:
        vs, vr = TfidfVectorizer(ngram_range=(1, 2), min_df=2), TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        X_tr = build_features(vs, vr, [rows[i] for i in train_idx], fit=True)
        model = LogisticRegression(C=TFIDF_C, max_iter=3000).fit(X_tr, y[train_idx])
        probs = model.predict_proba(build_features(vs, vr, [rows[i] for i in test_idx]))
        order = list(model.classes_)
        for i, p in zip(test_idx, probs):
            out[rows[i]["id"]] = [float(p[order.index(label)]) for label in LABELS]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


def tfidf_test_probs(set_name):
    """TF-IDF probabilities on a test set, from the classifier trained on the full train split."""
    recs = json.load(open(os.path.join("results", f"baseline_tfidf_{set_name}.json"), encoding="utf-8"))
    return {r["task_id"]: [r["probabilities"][label] for label in LABELS] for r in recs}


def features(tasks, set_name, regime):
    """One row of numbers per task: every grader run's four probabilities, then the TF-IDF
    probabilities. Returns (matrix, list of the feature names)."""
    columns, names = [], []
    for model in MODELS:
        for mode in REGIME_MODES[regime]:
            runs = grader_probs(set_name, mode, model)
            columns.append([runs[t["task_id"]] for t in tasks])
            names += [f"{model}|{mode}|{label}" for label in LABELS]
    tfidf = tfidf_train_oof(regime) if set_name == TRAIN_SET else tfidf_test_probs(set_name)
    columns.append([tfidf[t["task_id"]] for t in tasks])
    names += [f"tfidf|{label}" for label in LABELS]
    return np.hstack([np.array(c) for c in columns]), names


def train_data(regime):
    """Judge training data for one regime: features, gold labels and the 5 cross-validation
    splits (split by question for 'unseen', to simulate questions the judge has never seen)."""
    tasks = json.load(open(os.path.join("data", f"eval_{TRAIN_SET}.json"), encoding="utf-8"))
    X, _ = features(tasks, TRAIN_SET, regime)
    y = np.array([t["gold_label"] for t in tasks])
    if regime == "seen":
        splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    else:
        splits = list(GroupKFold(5).split(X, y, groups=[t["question"] for t in tasks]))
    return X, y, splits


def train_judge(regime):
    """Train the judge for one regime on the train sample, choosing its regularisation by
    5-fold cross-validation. Returns (model, cross-validated accuracy, chosen C)."""
    X, y, splits = train_data(regime)
    best = None
    for c in C_VALUES:
        correct = 0
        for tr, va in splits:
            model = LogisticRegression(C=c, max_iter=5000).fit(X[tr], y[tr])
            correct += (model.predict(X[va]) == y[va]).sum()
        acc = correct / len(y)
        if best is None or acc > best[1]:
            best = (c, acc)
    model = LogisticRegression(C=best[0], max_iter=5000).fit(X, y)
    return model, best[1], best[0]


def judge_predict(model, X):
    """Final labels and the probability of each label, in LABELS order."""
    probs = model.predict_proba(X)
    order = list(model.classes_)
    probs = probs[:, [order.index(label) for label in LABELS]]
    return [LABELS[i] for i in probs.argmax(axis=1)], probs


def choose_threshold(regime, c):
    """Escalation threshold: the lowest judge confidence at which cross-validated accuracy on the
    train sample reaches TARGET_ACCURACY. Grades less confident than this go to a teacher."""
    X, y, splits = train_data(regime)
    confidence = np.zeros(len(y))
    right = np.zeros(len(y), dtype=bool)
    for tr, va in splits:
        model = LogisticRegression(C=c, max_iter=5000).fit(X[tr], y[tr])
        p = model.predict_proba(X[va])
        confidence[va] = p.max(axis=1)
        right[va] = model.classes_[p.argmax(axis=1)] == y[va]
    for t in sorted(set(confidence)):
        keep = confidence >= t
        if right[keep].mean() >= TARGET_ACCURACY:
            return float(t)
    return 1.0
