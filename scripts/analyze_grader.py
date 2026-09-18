# Analyze a grader results file (like results/grader_only.json):
# how often the model is confidently wrong, how confidence compares
# between correct and wrong predictions, and a couple of examples.
#
# Usage: python -m scripts.analyze_grader results/grader_only.json
import json
import sys

CONFIDENT_THRESHOLD = 0.9  # top probability above this counts as "confident"
NUM_EXAMPLES = 2  # how many confidently-wrong examples to print


def top_prob(result):
    """The probability of whichever label the model predicted."""
    return max(result["probabilities"].values())


def is_correct(result):
    return result["predicted_label"] == result["gold_label"]


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.analyze_grader <results_file.json>")
        sys.exit(1)

    results_path = sys.argv[1]
    results = json.load(open(results_path, encoding="utf-8"))
    tasks = {t["task_id"]: t for t in json.load(open("data/tasks.json", encoding="utf-8"))}

    correct_top_probs = []
    wrong_top_probs = []
    confidently_wrong = []

    for r in results:
        tp = top_prob(r)
        if is_correct(r):
            correct_top_probs.append(tp)
        else:
            wrong_top_probs.append(tp)
            if tp > CONFIDENT_THRESHOLD:
                confidently_wrong.append(r)

    print(f"1. Confidently wrong (wrong + top prob > {CONFIDENT_THRESHOLD}): "
          f"{len(confidently_wrong)} out of {len(results)}")

    print("\n2. Average top probability:")
    if correct_top_probs:
        avg_correct = sum(correct_top_probs) / len(correct_top_probs)
        print(f"   correct predictions: {avg_correct:.3f} (n={len(correct_top_probs)})")
    else:
        print("   correct predictions: no examples")
    if wrong_top_probs:
        avg_wrong = sum(wrong_top_probs) / len(wrong_top_probs)
        print(f"   wrong predictions:   {avg_wrong:.3f} (n={len(wrong_top_probs)})")
    else:
        print("   wrong predictions: no examples")

    print(f"\n3. Examples of confidently wrong predictions (showing up to {NUM_EXAMPLES}):")
    for r in confidently_wrong[:NUM_EXAMPLES]:
        t = tasks[r["task_id"]]
        print("---")
        print("task_id:", r["task_id"])
        print("question:", t["question"])
        print("student_answer:", t["student_answer"])
        print("gold_label:", r["gold_label"])
        print(f"predicted_label: {r['predicted_label']} (p={top_prob(r):.3f})")

    print("\nAll confidently-wrong task_ids:", [r["task_id"] for r in confidently_wrong])


if __name__ == "__main__":
    main()
