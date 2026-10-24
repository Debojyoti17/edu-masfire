# Summarise the answer-bank simulation (results/answer_bank/<set>.json from
# scripts/simulate_answer_bank.py) in a way that compares settings fairly: for each setting, how many
# answers the teacher must grade by hand (the bank answers plus the ones held back) if at most 2%,
# 5% or 10% of the scored answers may receive a wrong grade. The judge's most confident grades are
# sent to students first. Also gives accuracy and how well confidence separates right from wrong
# grades (AUROC: 0.5 = no better than chance, 1.0 = perfect).
#
# The cut-offs here are chosen by looking at the test answers, so these numbers describe the best
# possible trade-off for each setting, for comparing settings - not what a school would get with a
# cut-off fixed in advance (that is the "teacher_graded" number in the simulation output).
#
# Writes results/answer_bank_summary.md.
# Usage: python -m scripts.analyze_answer_bank
import json
import os
from collections import Counter

import numpy as np

from scripts.compare_confidence import auroc
from src.judge import LABELS

SETS = ["test_uq", "test_ud_600"]
TARGETS = [0.02, 0.05, 0.10]


def workload(confidence, right, bank_size, target):
    """Fewest answers the teacher must grade so that wrong grades reaching students stay at or
    below `target` (share of scored answers), sending the most confident grades first."""
    order = np.argsort(-np.array(confidence), kind="stable")
    n, wrong, best = len(order), 0, 0
    for j, i in enumerate(order, start=1):
        wrong += not right[i]
        if wrong / n <= target:
            best = j
    return bank_size + n - best


def setting_name(key):
    """Readable name of a per_answer key ('0', '2', '2+0', ...)."""
    if key == "0":
        return "no bank (new-question judge)"
    if key.endswith("+0"):
        return f"{key[:-2]} per question, both judges averaged"
    return f"{key} per question (bank judge)"


def summarise_set(set_name):
    """Markdown table for one test set, or None if its simulation has not been run."""
    path = os.path.join("results", "answer_bank", f"{set_name}.json")
    if not os.path.exists(path):
        return None
    run = json.load(open(path, encoding="utf-8"))
    tasks = json.load(open(os.path.join("data", f"eval_{set_name}.json"), encoding="utf-8"))
    gold = {t["task_id"]: t["gold_label"] for t in tasks}
    questions = len(Counter(t["question"] for t in tasks))
    lines = [f"## {set_name}: {questions} new questions, "
             f"{len(next(iter(run['per_answer'].values())))} answers scored", "",
             "| teacher grades first | accuracy | confidence AUROC | "
             + " | ".join(f"teacher workload at <= {t:.0%} wrong" for t in TARGETS) + " |",
             "|---|---|---|" + "---|" * len(TARGETS)]
    for key, answers in run["per_answer"].items():
        ids = list(answers)
        probs = np.array([answers[i] for i in ids])
        right = [LABELS[int(p.argmax())] == gold[i] for i, p in zip(ids, probs)]
        confidence = probs.max(axis=1)
        k = int(key.split("+")[0])
        cells = [str(workload(confidence, right, k * questions, t)) for t in TARGETS]
        lines.append(f"| {setting_name(key)} | {np.mean(right):.3f} | {auroc(list(zip(confidence, right))):.3f} | "
                     + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    """Write the summary for every set that has been simulated."""
    parts = ["# Teacher-built answer bank: simulation summary", "",
             "Teacher workload = bank answers graded by hand + answers held back for the teacher, "
             "with the best cut-off for each target (see the note in scripts/analyze_answer_bank.py).", ""]
    for set_name in SETS:
        table = summarise_set(set_name)
        if table:
            parts.append(table)
    out = os.path.join("results", "answer_bank_summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
