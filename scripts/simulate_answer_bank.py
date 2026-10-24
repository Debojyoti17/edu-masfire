# Teacher-built answer bank (idea 1): how much does it help if, for a brand-new question, the
# teacher grades the first few student answers by hand and the system then uses those as worked
# examples for the rest of the class?
#
# Simulation on a test set of new questions (test_uq, or test_ud_600): for every question the
# answers are shuffled once (seed 42). The first 8 are the "teacher pool"; only the answers after
# them are used for scoring, so every setting is scored on exactly the same answers.
#   k = 0      : no bank - the question is new, so the unseen-question judge is used (similar
#                examples + zero-shot), read from the saved runs in results/rag
#   k = 2, 4, 8: the teacher's first k graded answers per question go into the bank; the question
#                now counts as "seen", so the three LLM graders get same-question examples from the
#                bank, the TF-IDF classifier is retrained on train + bank, and the seen-question
#                judge combines them
# Reports judge accuracy, how many answers are graded automatically at the usual escalation
# threshold, how many wrong grades would reach students, and the teacher's total workload.
# Also tries averaging the two judges (bank judge + new-question judge) for each k.
# per_answer in the saved file holds each judge's four probabilities (LABELS order) per answer.
#
# Usage: python -m scripts.simulate_answer_bank test_uq      (or test_ud_600)
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.graders import grade_with_prompt, rag_prompt
from src.judge import LABELS, MODELS, choose_threshold, features, judge_predict, train_judge
from src.retrieval import retrieve, use_answer_bank
from scripts.run_pipeline_v2 import CACHE_DIR, key_of, load_cache, save_to_cache
from scripts.run_tfidf_baseline import build_features, load_split

POOL = 8
BANK_SIZES = [2, 4, 8]


def split_pool(tasks):
    """Per question, shuffle the answers once (seed 42). Returns ({question: first POOL answers},
    list of the remaining answers, which are the ones scored)."""
    by_question = defaultdict(list)
    for t in tasks:
        by_question[t["question"]].append(t)
    rng = random.Random(42)
    pool, scored = {}, []
    for question in sorted(by_question):
        answers = sorted(by_question[question], key=lambda t: t["task_id"])
        rng.shuffle(answers)
        pool[question] = answers[:POOL]
        scored += answers[POOL:]
    return pool, scored


def bank_rows(pool, k):
    """The teacher's first k graded answers per question, in the train-row format."""
    return [{"id": t["task_id"], "question": t["question"], "reference_answer": t["reference_answer"],
             "student_answer": t["student_answer"], "label_name": t["gold_label"]}
            for answers in pool.values() for t in answers[:k]]


def tfidf_with_bank(bank):
    """TF-IDF classifier retrained on the train split plus the bank; returns a function giving
    probabilities (LABELS order) for a list of tasks."""
    rows = load_split("train") + bank
    vs, vr = TfidfVectorizer(ngram_range=(1, 2), min_df=2), TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    model = LogisticRegression(C=3.0, max_iter=3000).fit(build_features(vs, vr, rows, fit=True),
                                                         [r["label_name"] for r in rows])
    order = list(model.classes_)

    def probs(tasks):
        """Probabilities for each task, in LABELS order."""
        p = model.predict_proba(build_features(vs, vr, tasks))
        return [[float(row[order.index(label)]) for label in LABELS] for row in p]
    return probs


def summarise(name, labels, probs, threshold, scored, teacher_graded_bank):
    """One result row: accuracy, auto-graded share and its accuracy, wrong grades reaching
    students, and the teacher's workload (bank answers + escalated answers)."""
    gold = np.array([t["gold_label"] for t in scored])
    right = np.array(labels) == gold
    auto = np.array(probs).max(axis=1) >= threshold
    return {"setting": name, "scored": len(scored), "accuracy": round(float(right.mean()), 3),
            "auto_share": round(float(auto.mean()), 3),
            "auto_accuracy": round(float(right[auto].mean()), 3) if auto.any() else None,
            "wrong_reaching_students": round(float((~right & auto).mean()), 3),
            "teacher_graded": teacher_graded_bank + int((~auto).sum())}


def main():
    """Run k = 0, 2, 4, 8 on one test set and save results/answer_bank/<set>.json."""
    set_name = sys.argv[1] if len(sys.argv) > 1 else "test_uq"
    tasks = json.load(open(os.path.join("data", f"eval_{set_name}.json"), encoding="utf-8"))
    pool, scored = split_pool(tasks)
    print(f"{set_name}: {len(pool)} questions, {len(scored)} answers scored (beyond the first {POOL} per question)", flush=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    out_dir = os.path.join("results", "answer_bank")
    os.makedirs(out_dir, exist_ok=True)
    results, per_answer = [], {}

    # k = 0: the question is new; use the saved unseen-question grader runs.
    judge, _, c = train_judge("unseen")
    threshold = choose_threshold("unseen", c)
    all_X, _ = features(tasks, set_name, "unseen")
    row_of = {t["task_id"]: i for i, t in enumerate(tasks)}
    X = all_X[[row_of[t["task_id"]] for t in scored]]  # same order as the scored answers
    labels, probs = judge_predict(judge, X)
    results.append(summarise("k=0 (no bank, new-question judge)", labels, probs, threshold, scored, 0))
    per_answer["0"] = {t["task_id"]: [float(x) for x in p] for t, p in zip(scored, probs)}
    probs_new_question = probs
    print(results[-1], flush=True)

    # k = 2, 4, 8: the teacher's first k answers become same-question examples.
    judge, _, c = train_judge("seen")
    threshold = choose_threshold("seen", c)
    cache = load_cache("graders")
    for k in BANK_SIZES:
        bank = bank_rows(pool, k)
        use_answer_bank(bank)
        columns = []
        for model in MODELS:
            start, col = time.time(), []
            for n, t in enumerate(scored, start=1):
                prompt = rag_prompt(t, retrieve(t, "same"), "same")
                key = key_of(model, prompt)
                if key not in cache:
                    save_to_cache("graders", key, list(grade_with_prompt(prompt, model)), cache)
                p = cache[key][1]
                col.append([p[l] for l in LABELS] if p else [0.25] * 4)
                if n % 100 == 0 or n == len(scored):
                    print(f"  k={k} {model}: {n}/{len(scored)} ({time.time() - start:.0f}s)", flush=True)
            columns.append(np.array(col))
            single = np.array(LABELS)[np.array(col).argmax(axis=1)]
            results.append({"setting": f"k={k} {model} alone",
                            "accuracy": round(float((single == np.array([t["gold_label"] for t in scored])).mean()), 3)})
        columns.append(np.array(tfidf_with_bank(bank)(scored)))
        labels, probs = judge_predict(judge, np.hstack(columns))
        results.append(summarise(f"k={k} (bank, seen-question judge)", labels, probs, threshold, scored,
                                 len(bank)))
        per_answer[str(k)] = {t["task_id"]: [float(x) for x in p] for t, p in zip(scored, probs)}
        print(results[-1], flush=True)
        # Both judges together: average the new-question judge and the bank judge.
        both = (np.array(probs) + np.array(probs_new_question)) / 2
        results.append(summarise(f"k={k} + new-question judge (average)", list(np.array(LABELS)[both.argmax(axis=1)]),
                                 both, threshold, scored, len(bank)))
        per_answer[f"{k}+0"] = {t["task_id"]: [float(x) for x in p] for t, p in zip(scored, both)}
        print(results[-1], flush=True)
    use_answer_bank([])

    with open(os.path.join(out_dir, f"{set_name}.json"), "w", encoding="utf-8") as f:
        json.dump({"set": set_name, "pool_per_question": POOL, "results": results, "per_answer": per_answer}, f, indent=1)
    print(f"Saved results/answer_bank/{set_name}.json", flush=True)


if __name__ == "__main__":
    main()
