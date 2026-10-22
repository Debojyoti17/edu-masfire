# Run the Grader step on its own (no Planner/Tutor/Feedback) on every task in data/tasks.json.
# For each task, ask qwen2.5:7b to pick a label as a letter A-D, save the
# probability of every label plus the raw mass, then report accuracy
# overall and broken down by gold label.
import json
from collections import defaultdict

from src.llm import choice_probabilities

# The four labels, in the same order as the letters we ask the model to pick.
LETTERS = ["A", "B", "C", "D"]
LABELS = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant"]
LETTER_TO_LABEL = dict(zip(LETTERS, LABELS))


def build_prompt(task):
    """Turn one task into the grading question we send to the model."""
    return f"""Question: {task['question']}
Reference answer: {task['reference_answer']}
Student answer: {task['student_answer']}

Is the student answer correct, contradictory, partially_correct_incomplete, or irrelevant?
Answer with one letter only.
A) correct
B) contradictory
C) partially_correct_incomplete
D) irrelevant
"""


def grade_task(task):
    """Ask the model to grade one task, return a result dict for it."""
    prompt = build_prompt(task)
    result = choice_probabilities(prompt, LETTERS)

    if result["probs"] is None:
        # The model didn't produce any of A/B/C/D among its top candidates.
        predicted_label = None
        probabilities = {label: 0.0 for label in LABELS}
    else:
        # Pick the letter with the highest probability as the prediction.
        best_letter = max(result["probs"], key=result["probs"].get)
        predicted_label = LETTER_TO_LABEL[best_letter]
        probabilities = {LETTER_TO_LABEL[letter]: prob for letter, prob in result["probs"].items()}

    return {
        "task_id": task["task_id"],
        "gold_label": task["gold_label"],
        "predicted_label": predicted_label,
        "probabilities": probabilities,
        "mass": result["mass"],
    }


def main():
    tasks = json.load(open("data/tasks.json", encoding="utf-8"))

    results = []
    for i, task in enumerate(tasks, start=1):
        print(f"[{i}/{len(tasks)}] grading {task['task_id']}...")
        results.append(grade_task(task))

    with open("results/grader_only.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Overall accuracy.
    correct = sum(1 for r in results if r["predicted_label"] == r["gold_label"])
    print(f"\nOverall accuracy: {correct}/{len(results)} = {correct / len(results):.3f}")

    # Accuracy split by gold label.
    totals_by_label = defaultdict(int)
    correct_by_label = defaultdict(int)
    for r in results:
        totals_by_label[r["gold_label"]] += 1
        if r["predicted_label"] == r["gold_label"]:
            correct_by_label[r["gold_label"]] += 1

    print("\nAccuracy by gold label:")
    for label in LABELS:
        total = totals_by_label[label]
        right = correct_by_label[label]
        print(f"  {label}: {right}/{total} = {right / total:.3f}")

    print("\nSaved full results to results/grader_only.json")


if __name__ == "__main__":
    main()
