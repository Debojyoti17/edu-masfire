# Grade every task in data/tasks.json using a chosen prompt version, save results to
# results/grader_{version}.json.
#
# Versions:
#   v1 - label names only (the original prompt)
#   v2 - v1 plus the rubric definitions from src/rubric.py
#   v3 - v2 plus the 8 worked examples from data/examples.json
#   v4 - two-step: first ask the model to reason in 2-3 sentences (v2
#        prompt, no examples), then make a second call with that
#        reasoning included and read the confidence from that call
#   v5 - same two-step approach as v4, but with three labels instead of
#        four: "contradictory" and "irrelevant" are merged into
#        "incorrect". Gold labels are remapped the same way for scoring.
#   v6 - same two-step approach as v4, but step 1 is only allowed to
#        describe how the student answer compares to the reference
#        answer - it must not name or hint at a label. Step 2 then
#        makes the actual label decision from that description.
#
# By default everything runs on src.llm.DEFAULT_MODEL. Pass a second
# argument to use a different Ollama model instead, e.g.:
#   python -m scripts.run_grader v2 llama3.1:8b
#
# Add --prefill to plant "Label: " as the start of the model's decision
# reply, so the next token has to be a letter. This fixes a measurement
# problem where some models (llama3.1:8b especially) want to open their
# reply with a full sentence instead of a bare letter, leaving most of
# the probability mass on non-option tokens:
#   python -m scripts.run_grader v6 llama3.1:8b --prefill
import json
import os
import sys

from src.llm import DEFAULT_MODEL, choice_probabilities, generate
from src.rubric import RUBRIC

LETTERS = ["A", "B", "C", "D"]
LABELS = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant"]
LETTER_TO_LABEL = dict(zip(LETTERS, LABELS))
VERSIONS = ["v1", "v2", "v3", "v4", "v5", "v6"]
REASONING_MAX_TOKENS = 200
PREFILL_TEXT = "Label: "

# v5's 3-label scheme: "contradictory" and "irrelevant" merged into "incorrect".
V5_LETTERS = ["A", "B", "C"]
V5_LABELS = ["correct", "partially_correct_incomplete", "incorrect"]
V5_LETTER_TO_LABEL = dict(zip(V5_LETTERS, V5_LABELS))
V5_RUBRIC = {
    "correct": RUBRIC["correct"],
    "partially_correct_incomplete": RUBRIC["partially_correct_incomplete"],
    "incorrect": (
        "The student's answer is wrong for this question - either because "
        "it states something that conflicts with the reference answer (a "
        "wrong fact, a reversed relationship, or a wrong cause), or because "
        "it never actually addresses what the question is asking (restating "
        "the question, describing something unrelated, or answering a "
        "different question)."
    ),
}


def merge_label_v5(label):
    """Map a 4-way gold label to v5's 3-way scheme: contradictory and
    irrelevant both become "incorrect"."""
    if label in ("contradictory", "irrelevant"):
        return "incorrect"
    return label


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


def build_reasoning_prompt(task):
    """v4 step 1: ask the model to reason about which label fits, using the
    v2 rubric prompt but asking for a short explanation instead of a letter."""
    return f"""Question: {task['question']}
Reference answer: {task['reference_answer']}
Student answer: {task['student_answer']}

Compare the student answer to the reference answer using this rubric:
{format_rubric()}

In 2-3 sentences, reason about which label fits best, then end by stating the label.
"""


def build_decision_prompt(task, reasoning):
    """v4 step 2: give the model its own reasoning back and ask for only the
    letter, so the confidence we read comes from a decision made after
    reasoning rather than from the first token of a cold guess."""
    return f"""Question: {task['question']}
Reference answer: {task['reference_answer']}
Student answer: {task['student_answer']}

Rubric:
{format_rubric()}

Reasoning:
{reasoning}

Based on this reasoning, answer with one letter only: A, B, C, or D.
"""


def format_v5_rubric():
    """The A) label: definition lines used in v5, over the 3-label scheme."""
    return "\n".join(f"{letter}) {label}: {V5_RUBRIC[label]}" for letter, label in zip(V5_LETTERS, V5_LABELS))


def build_v5_reasoning_prompt(task):
    """v5 step 1: same idea as v4's reasoning step, but over the 3-label rubric."""
    return f"""Question: {task['question']}
Reference answer: {task['reference_answer']}
Student answer: {task['student_answer']}

Compare the student answer to the reference answer using this rubric:
{format_v5_rubric()}

In 2-3 sentences, reason about which label fits best, then end by stating the label.
"""


def build_v5_decision_prompt(task, reasoning):
    """v5 step 2: same idea as v4's decision step, but over the 3-label rubric."""
    return f"""Question: {task['question']}
Reference answer: {task['reference_answer']}
Student answer: {task['student_answer']}

Rubric:
{format_v5_rubric()}

Reasoning:
{reasoning}

Based on this reasoning, answer with one letter only: A, B, or C.
"""


def build_comparison_prompt(task):
    """v6 step 1: describe how the student answer compares to the reference
    answer - what it covers, what it misses, and whether anything in it
    contradicts the reference answer. Explicitly forbidden from naming or
    hinting at a label, so step 2 still has a real decision left to make."""
    return f"""Question: {task['question']}
Reference answer: {task['reference_answer']}
Student answer: {task['student_answer']}

In 2-3 sentences, describe how the student answer compares to the reference
answer: what it covers, what it is missing, and whether anything in it
contradicts the reference answer.
Do not name or hint at a grading label. Only describe the comparison.
"""


def build_v6_decision_prompt(task, comparison):
    """v6 step 2: given only the comparison description (no label hinted),
    make the actual label decision using the rubric."""
    return f"""Question: {task['question']}
Reference answer: {task['reference_answer']}
Student answer: {task['student_answer']}

Rubric:
{format_rubric()}

Comparison:
{comparison}

Based on this comparison, answer with one letter only: A, B, C, or D.
"""


def package_result(task, result, letter_to_label, labels, model, prefill=None, reasoning=None, gold_label=None):
    """Turn a choice_probabilities() result into the dict we save per task."""
    if result["probs"] is None:
        # The model didn't produce any of the option letters among its top candidates.
        predicted_label = None
        probabilities = {label: 0.0 for label in labels}
    else:
        best_letter = max(result["probs"], key=result["probs"].get)
        predicted_label = letter_to_label[best_letter]
        probabilities = {letter_to_label[letter]: prob for letter, prob in result["probs"].items()}

    packaged = {
        "task_id": task["task_id"],
        "gold_label": gold_label if gold_label is not None else task["gold_label"],
        "predicted_label": predicted_label,
        "probabilities": probabilities,
        "mass": result["mass"],
        "model": model,
        "prefill": prefill,
    }
    if reasoning is not None:
        packaged["reasoning"] = reasoning
    return packaged


def grade_task(task, version, model, prefill=None):
    """Ask the model to grade one task with the given prompt version.
    prefill, if given, is planted as the start of the model's decision
    reply (see choice_probabilities) - it never affects the reasoning/
    comparison step, only the final letter-choosing call."""
    if version == "v4":
        reasoning = generate(build_reasoning_prompt(task), model=model, max_tokens=REASONING_MAX_TOKENS)
        result = choice_probabilities(build_decision_prompt(task, reasoning), LETTERS, model=model, prefill=prefill)
        return package_result(task, result, LETTER_TO_LABEL, LABELS, model, prefill=prefill, reasoning=reasoning)

    if version == "v5":
        reasoning = generate(build_v5_reasoning_prompt(task), model=model, max_tokens=REASONING_MAX_TOKENS)
        result = choice_probabilities(build_v5_decision_prompt(task, reasoning), V5_LETTERS, model=model, prefill=prefill)
        gold_label = merge_label_v5(task["gold_label"])
        return package_result(task, result, V5_LETTER_TO_LABEL, V5_LABELS, model, prefill=prefill,
                               reasoning=reasoning, gold_label=gold_label)

    if version == "v6":
        comparison = generate(build_comparison_prompt(task), model=model, max_tokens=REASONING_MAX_TOKENS)
        result = choice_probabilities(build_v6_decision_prompt(task, comparison), LETTERS, model=model, prefill=prefill)
        return package_result(task, result, LETTER_TO_LABEL, LABELS, model, prefill=prefill, reasoning=comparison)

    prompt = build_prompt(task, version)
    result = choice_probabilities(prompt, LETTERS, model=model, prefill=prefill)
    return package_result(task, result, LETTER_TO_LABEL, LABELS, model, prefill=prefill)


def model_slug(model):
    """Turn a model name like "llama3.1:8b" into a filename-safe piece like
    "llama3.1-8b" (":" is not allowed in Windows filenames)."""
    return model.replace(":", "-")


def main():
    """Read the command-line options, grade every task, save and print the accuracy.
    --tasks FILE grades a different task file (e.g. data/eval_test_uq.json)."""
    args = sys.argv[1:]
    use_prefill = "--prefill" in args
    if use_prefill:
        args = [a for a in args if a != "--prefill"]
    tasks_path = "data/tasks.json"
    if "--tasks" in args:
        at = args.index("--tasks")
        tasks_path = args[at + 1]
        args = args[:at] + args[at + 2:]

    if len(args) not in (1, 2) or args[0] not in VERSIONS:
        print(f"Usage: python -m scripts.run_grader <{'|'.join(VERSIONS)}> [model] [--prefill] [--tasks FILE]")
        sys.exit(1)

    version = args[0]
    model = args[1] if len(args) == 2 else DEFAULT_MODEL
    prefill = PREFILL_TEXT if use_prefill else None
    tasks = json.load(open(tasks_path, encoding="utf-8"))

    results = []
    for i, task in enumerate(tasks, start=1):
        if i % 50 == 0 or i == len(tasks):
            print(f"[{i}/{len(tasks)}] graded ({version}, {model}{', prefill' if use_prefill else ''})", flush=True)
        results.append(grade_task(task, version, model, prefill=prefill))

    # Keep the original filename for the default model with no prefill and the default task
    # file, so existing results stay where earlier scripts expect them. Non-default model,
    # prefill and task file each add a piece to the filename.
    name_parts = [f"grader_{version}"]
    if use_prefill:
        name_parts.append("prefill")
    if model != DEFAULT_MODEL:
        name_parts.append(model_slug(model))
    if tasks_path != "data/tasks.json":
        name_parts.append(os.path.splitext(os.path.basename(tasks_path))[0].replace("eval_", ""))
    out_path = f"results/{'_'.join(name_parts)}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    correct = sum(1 for r in results if r["predicted_label"] == r["gold_label"])
    print(f"\n{version} ({model}{', prefill' if use_prefill else ''}) "
          f"accuracy: {correct}/{len(results)} = {correct / len(results):.3f}")
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
