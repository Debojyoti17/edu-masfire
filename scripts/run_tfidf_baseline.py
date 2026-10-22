# A non-LLM baseline grader: a classic text classifier (TF-IDF word features + logistic
# regression) trained on the SciEntsBank train split, then tested on every test split and on
# our 120-task set. Gives reviewers a standard trained model to compare the LLM pipeline with.
#
# Needs data/raw/ from get_splits.py.
# Usage: python -m scripts.run_tfidf_baseline
import json
import os

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

LABELS = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant"]
SPLITS = ["test_ua", "test_uq", "test_ud"]
C_VALUES = [1.0, 3.0, 10.0, 30.0, 100.0]


def load_split(split):
    """Load one SciEntsBank split from data/raw, dropping the rare non_domain label."""
    with open(os.path.join("data", "raw", f"scientsbank_{split}.json"), encoding="utf-8") as f:
        rows = json.load(f)
    return [r for r in rows if r["label_name"] in LABELS]


def overlap_features(rows):
    """Three simple numbers per answer: share of reference-answer words the student used,
    share of student words found in the reference, and the student answer's length."""
    feats = []
    for r in rows:
        ref = set(r["reference_answer"].lower().split())
        stu = r["student_answer"].lower().split()
        stu_set = set(stu)
        feats.append([
            len(ref & stu_set) / max(1, len(ref)),
            len(ref & stu_set) / max(1, len(stu_set)),
            min(len(stu), 60) / 60,
        ])
    return csr_matrix(np.array(feats))


def build_features(vec_student, vec_reference, rows, fit=False):
    """Turn answers into numbers: word features of the student answer, word features of the
    reference answer, and the overlap features. fit=True learns the vocabulary (train only)."""
    students = [r["student_answer"] for r in rows]
    references = [r["question"] + " " + r["reference_answer"] for r in rows]
    if fit:
        a, b = vec_student.fit_transform(students), vec_reference.fit_transform(references)
    else:
        a, b = vec_student.transform(students), vec_reference.transform(references)
    return hstack([a, b, overlap_features(rows)]).tocsr()


def metrics(gold, pred):
    """Accuracy, macro-F1 (every label counts equally) and accuracy per label."""
    per_label = {}
    for label in LABELS:
        idx = [i for i, g in enumerate(gold) if g == label]
        per_label[label] = sum(pred[i] == label for i in idx) / len(idx) if idx else None
    return {
        "n": len(gold),
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=LABELS, average="macro"),
        "per_label_accuracy": per_label,
    }


def main():
    """Pick the regularisation strength by 5-fold cross-validation on train, train the final
    model, then score every test split and our 120-task set."""
    train = load_split("train")
    y_train = [r["label_name"] for r in train]

    best = None
    for c in C_VALUES:
        vs, vr = TfidfVectorizer(ngram_range=(1, 2), min_df=2), TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        X = build_features(vs, vr, train, fit=True)
        model = LogisticRegression(C=c, max_iter=3000)
        pred = cross_val_predict(model, X, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42))
        f1 = f1_score(y_train, pred, labels=LABELS, average="macro")
        print(f"C={c}: cross-validated macro-F1 on train = {f1:.3f}")
        if best is None or f1 > best[1]:
            best = (c, f1)
    print(f"Chosen C={best[0]}")

    vs, vr = TfidfVectorizer(ngram_range=(1, 2), min_df=2), TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_train = build_features(vs, vr, train, fit=True)
    model = LogisticRegression(C=best[0], max_iter=3000).fit(X_train, y_train)

    report = {"model": "tfidf_logreg", "C": best[0], "train_cv_macro_f1": best[1], "splits": {}}
    for split in SPLITS:
        rows = load_split(split)
        pred = list(model.predict(build_features(vs, vr, rows)))
        report["splits"][split] = metrics([r["label_name"] for r in rows], pred)

    tasks = json.load(open("data/tasks.json", encoding="utf-8"))
    rows = [{"question": t["question"], "reference_answer": t["reference_answer"],
             "student_answer": t["student_answer"]} for t in tasks]
    X = build_features(vs, vr, rows)
    probs = model.predict_proba(X)
    order = list(model.classes_)
    records = []
    for t, p in zip(tasks, probs):
        pr = {label: float(p[order.index(label)]) for label in LABELS}
        records.append({"task_id": t["task_id"], "gold_label": t["gold_label"],
                        "predicted_label": max(pr, key=pr.get), "probabilities": pr, "mass": 1.0,
                        "model": "tfidf_logreg"})
    report["splits"]["our_120_tasks"] = metrics([r["gold_label"] for r in records],
                                                [r["predicted_label"] for r in records])

    with open(os.path.join("results", "baseline_tfidf.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join("results", "baseline_tfidf_120.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print()
    for split, m in report["splits"].items():
        print(f"{split:14} n={m['n']:5}  accuracy={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}  "
              + "  ".join(f"{k[:7]}={v:.2f}" for k, v in m["per_label_accuracy"].items()))
    print("\nSaved results/baseline_tfidf.json and results/baseline_tfidf_120.json")


if __name__ == "__main__":
    main()
