# Compare the single Grader (v2 + chosen prefill) for every model against the trained TF-IDF
# baseline on each evaluation set: our 120 tasks, full test_ua (unseen answers), test_uq (unseen
# questions) and a 600-answer sample of test_ud (unseen domains). Prints a markdown table and
# writes it to results/eval_sets_summary.md.
#
# Usage: python -m scripts.compare_eval_sets
import json
import math
import os

LABELS = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant"]
MODELS = ["qwen2.5:7b", "llama3.1:8b", "mistral:7b", "gemma2:9b"]
SETS = [("120", "our 120 tasks"), ("test_ua", "unseen answers"), ("test_uq", "unseen questions"),
        ("test_ud_600", "unseen domains")]
OUT = os.path.join("results", "eval_sets_summary.md")


def grader_file(model, set_key):
    """Path of the saved single-Grader results for this model and evaluation set."""
    parts = ["grader_v2_prefill"]
    if model != "qwen2.5:7b":
        parts.append(model.replace(":", "-"))
    if set_key != "120":
        parts.append(set_key)
    return os.path.join("results", "_".join(parts) + ".json")


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k correct out of n."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def macro_f1(recs):
    """Average of the per-label F1 scores, so every label counts equally."""
    scores = []
    for label in LABELS:
        tp = sum(r["predicted_label"] == label and r["gold_label"] == label for r in recs)
        fp = sum(r["predicted_label"] == label and r["gold_label"] != label for r in recs)
        fn = sum(r["predicted_label"] != label and r["gold_label"] == label for r in recs)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * p * r / (p + r) if p + r else 0.0)
    return sum(scores) / len(scores)


def cell(path):
    """'accuracy [interval], F1' for one results file, or '-' if it does not exist yet."""
    if not os.path.exists(path):
        return "-"
    recs = json.load(open(path, encoding="utf-8"))
    k = sum(r["predicted_label"] == r["gold_label"] for r in recs)
    lo, hi = wilson(k, len(recs))
    return f"{k / len(recs):.3f} [{lo:.2f}-{hi:.2f}], F1 {macro_f1(recs):.3f} (n={len(recs)})"


def main():
    """Build the table: one row per grader, one column per evaluation set."""
    lines = ["# Single Grader vs trained baseline across evaluation sets", "",
             "Each cell: accuracy [95% Wilson interval], macro-F1 (n answers).", "",
             "| grader | " + " | ".join(f"{k} ({desc})" for k, desc in SETS) + " |",
             "|---|" + "---|" * len(SETS)]
    for m in MODELS:
        lines.append(f"| {m} (zero-shot LLM) | " + " | ".join(cell(grader_file(m, k)) for k, _ in SETS) + " |")
    lines.append("| TF-IDF + logistic regression (trained on train split) | "
                 + " | ".join(cell(os.path.join("results", f"baseline_tfidf_{k}.json")) for k, _ in SETS) + " |")
    text = "\n".join(lines)
    print(text)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
