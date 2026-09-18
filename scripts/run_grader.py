# Grade all 48 tasks using a chosen prompt version, save results to
# results/grader_{version}.json.
#
# Versions:
#   v1 - label names only (the original prompt)
#   v2 - v1 plus the rubric definitions from src/rubric.py
#   v3 - v2 plus the 8 worked examples from data/examples.json
#
# Usage: python -m scripts.run_grader v1
import json
import sys

from src.llm import choice_probabilities
from src.rubric import RUBRIC

LETTERS = ["A", "B", "C", "D"]
LABELS = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant"]
LETTER_TO_LABEL = dict(zip(LETTERS, LABELS))
VERSIONS = ["v1", "v2", "v3"]


def format_choices():
    """The bare A) label lines used in v1."""
    return "\n".join(f"{letter}) {label}" for letter, label in zip(LETTERS, LABELS))


def format_rubric():
    """The A) label: definition lines used in v2 and v3."""
    return "\n".join(f"{letter}) {label}: {RUBRIC[label]}" for letter, label in zip(LETTERS, LABELS))


def format_examples():
    """The 8 worked examples from data/examples.json, formatted like the
    grading question itself so the model sees question/reference/student/answer."""
    examples = json.load(open("data/examples.json", encoding="utf-8"))
    letter_for_label = {label: letter for letter, label in zip(LETTERS, LABELS)}
    blocks = []
    for ex in examples:
        letter = letter_for_label[ex["gold_label"]]
        blocks.append(
            f"Question: {ex['question']}\n"
            f"Reference answer: {ex['reference_answer']}\n"
            f"Student answer: {ex['student_answer']}\n"
            f"Answer: {letter}"
        )
    return "\n\n".join(blocks)


def build_prompt(task, version):
    """Build the grading prompt for the given version (v1/v2/v3)."""
    choices = format_choices() if version == "v1" else format_rubric()

    question_block = f"""Question: {task['question']}
Reference answer: {task['reference_answer']}
Student answer: {task['student_answer']}

Is the student answer correct, contradictory, partially_correct_incomplete, or irrelevant?
Answer with one letter only.
{choices}
"""

    if version == "v3":
        return (
            "Here are worked examples of how to grade, one per label:\n\n"
            + format_examples() + "\n\n---\n\n" + question_block
        )
    return question_block


def grade_task(task, version):
    """Ask the model to grade one task with the given prompt version."""
    prompt = build_prompt(task, version)
    result = choice_probabilities(prompt, LETTERS)

    if result["probs"] is None:
        # The model didn't produce any of A/B/C/D among its top candidates.
        predicted_label = None
        probabilities = {label: 0.0 for label in LABELS}
    else:
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
    if len(sys.argv) != 2 or sys.argv[1] not in VERSIONS:
        print(f"Usage: python -m scripts.run_grader <{'|'.join(VERSIONS)}>")
        sys.exit(1)

    version = sys.argv[1]
    tasks = json.load(open("data/tasks.json", encoding="utf-8"))

    results = []
    for i, task in enumerate(tasks, start=1):
        print(f"[{i}/{len(tasks)}] grading {task['task_id']} ({version})...")
        results.append(grade_task(task, version))

    out_path = f"results/grader_{version}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    correct = sum(1 for r in results if r["predicted_label"] == r["gold_label"])
    print(f"\n{version} accuracy: {correct}/{len(results)} = {correct / len(results):.3f}")
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
