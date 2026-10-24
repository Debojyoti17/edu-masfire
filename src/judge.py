# Judge model for the redesigned pipeline: combines the label probabilities of the three LLM
# graders and the trained TF-IDF classifier into one final label with a confidence.
# It is a logistic regression trained only on the 2000-answer train sample, never on test data.
# By default it is trained fault-aware (with copies where one grader lies or crashes), which
# makes it far less vulnerable to a single faulty grader at almost no cost on clean answers.
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
C_VALUES = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
TARGET_ACCURACY = 0.80
RAG_DIR = os.path.join("results", "rag")
# Extra graders whose probabilities the judge also uses (fine-tuned models from
# scripts/train_finetuned_grader.py). Added only if they improve cross-validated accuracy.
# deberta-v3-base-nli raised cross-validated accuracy from 0.729 to 0.772 (seen questions) and from
# 0.598 to 0.651 (new questions) - see scripts/benchmark_budget.py, results/budget/benchmark.json.
EXTRA_GRADERS = ["deberta-v3-base-nli"]


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


def finetuned_saved_probs(name, set_name, regime):
    """Saved fine-tuned grader probabilities: out-of-fold for the train sample, final model otherwise."""
    file = f"train_oof_{regime}.json" if set_name == TRAIN_SET else f"{set_name}.json"
    with open(os.path.join("results", "finetuned", name, file), encoding="utf-8") as f:
        return json.load(f)


def features(tasks, set_name, regime, extras=None):
    """One row of numbers per task: every grader run's four probabilities, then the TF-IDF
    probabilities, then any extra (fine-tuned) graders. Returns (matrix, list of the feature names)."""
    extras = EXTRA_GRADERS if extras is None else extras
    columns, names = [], []
    for model in MODELS:
        for mode in REGIME_MODES[regime]:
            runs = grader_probs(set_name, mode, model)
            columns.append([runs[t["task_id"]] for t in tasks])
            names += [f"{model}|{mode}|{label}" for label in LABELS]
    tfidf = tfidf_train_oof(regime) if set_name == TRAIN_SET else tfidf_test_probs(set_name)
    columns.append([tfidf[t["task_id"]] for t in tasks])
    names += [f"tfidf|{label}" for label in LABELS]
    for name in extras:
        saved = finetuned_saved_probs(name, set_name, regime)
        columns.append([saved[t["task_id"]] for t in tasks])
        names += [f"{name}|{label}" for label in LABELS]
    return np.hstack([np.array(c) for c in columns]), names


def train_data(regime, extras=None):
    """Judge training data for one regime: features, gold labels and the 5 cross-validation
    splits (split by question for 'unseen', to simulate questions the judge has never seen)."""
    tasks = json.load(open(os.path.join("data", f"eval_{TRAIN_SET}.json"), encoding="utf-8"))
    X, _ = features(tasks, TRAIN_SET, regime, extras)
    y = np.array([t["gold_label"] for t in tasks])
    if regime == "seen":
        splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    else:
        splits = list(GroupKFold(5).split(X, y, groups=[t["question"] for t in tasks]))
    return X, y, splits


def grader_columns(regime):
    """Which feature columns belong to each LLM grader (all of its modes), as (start, end)."""
    per = 4 * len(REGIME_MODES[regime])
    return {m: (k * per, (k + 1) * per) for k, m in enumerate(MODELS)}


def with_faulty_graders(X, y, regime, seed=42):
    """Fault-aware training data: the original rows, plus for each LLM grader one copy where it
    gives a random wrong label (any of the other three) at a random confidence 0.5-0.99, and one
    copy where it crashes (all 0.25). This is a generic fault model, not any specific test fault,
    so the judge learns not to trust a single grader blindly."""
    rng = np.random.default_rng(seed)
    rows, labels = [X], [y]
    for model in MODELS:
        a, b = grader_columns(regime)[model]
        lying = X.copy()
        for row in lying:
            for s in range(a, b, 4):
                own = int(np.argmax(row[s:s + 4]))
                wrong = rng.choice([i for i in range(4) if i != own])
                conf = rng.uniform(0.5, 0.99)
                row[s:s + 4] = (1 - conf) / 3
                row[s + wrong] = conf
        crashed = X.copy()
        crashed[:, a:b] = 0.25
        rows += [lying, crashed]
        labels += [y, y]
    return np.vstack(rows), np.concatenate(labels)


def fit(X, y, regime, c, robust):
    """Fit one judge; robust=True trains on the fault-aware data."""
    if robust:
        X, y = with_faulty_graders(X, y, regime)
    return LogisticRegression(C=c, max_iter=5000).fit(X, y)


def cv_predictions(regime, c, robust, extras=None):
    """Cross-validated judge probabilities on the clean train sample (fault-aware copies are only
    ever added to the training folds, never to the fold being predicted)."""
    X, y, splits = train_data(regime, extras)
    probs = np.zeros((len(y), len(LABELS)))
    for tr, va in splits:
        model = fit(X[tr], y[tr], regime, c, robust)
        order = list(model.classes_)
        probs[va] = model.predict_proba(X[va])[:, [order.index(label) for label in LABELS]]
    return probs, y


def train_judge(regime, robust=True, extras=None):
    """Train the judge for one regime on the train sample, choosing its regularisation by
    5-fold cross-validated accuracy on clean answers. Returns (model, cv accuracy, chosen C)."""
    X, y, _ = train_data(regime, extras)
    best = None
    for c in C_VALUES:
        probs, _ = cv_predictions(regime, c, robust, extras)
        acc = (np.array(LABELS)[probs.argmax(axis=1)] == y).mean()
        if best is None or acc > best[1]:
            best = (c, acc)
    return fit(X, y, regime, best[0], robust), best[1], best[0]


def judge_predict(model, X):
    """Final labels and the probability of each label, in LABELS order."""
    probs = model.predict_proba(X)
    order = list(model.classes_)
    probs = probs[:, [order.index(label) for label in LABELS]]
    return [LABELS[i] for i in probs.argmax(axis=1)], probs


def choose_threshold(regime, c, robust=True, extras=None):
    """Escalation threshold: the lowest judge confidence at which cross-validated accuracy on the
    train sample reaches TARGET_ACCURACY. Grades less confident than this go to a teacher."""
    probs, y = cv_predictions(regime, c, robust, extras)
    confidence = probs.max(axis=1)
    right = np.array(LABELS)[probs.argmax(axis=1)] == y
    for t in sorted(set(confidence)):
        keep = confidence >= t
        if right[keep].mean() >= TARGET_ACCURACY:
            return float(t)
    return 1.0
