# Grade all tasks with the v2 prompt (rubric, no examples) and get a
# confidence number for each prediction, using one of four methods. Saves
# to results/confidence_{method}.json in a shared schema so every method
# can be compared the same way (see scripts/compare_confidence.py).
#
# Methods:
#   first_token     - read the probability of the predicted letter from
#                      prefilled logprobs (this is just v2+prefill, so we
#                      reuse results/grader_v2_prefill.json instead of
#                      re-running the model).
#   verbalized       - ask the model to state a label and a 0-100
#                      confidence in words, then parse both out.
#   self_consistency - run v2 grading 5 times at temperature 0.7, use the
#                      majority label and the fraction of runs agreeing
#                      with it as the confidence.
#   verifier         - a second call that shows the label first_token
#                      chose and asks "Is this label correct? Yes or No",
#                      reading P(Yes) from prefilled logprobs.
#
# Usage: python -m scripts.run_confidence_method verbalized [model]
import json
import re
import sys
from collections import Counter

from src.llm import DEFAULT_MODEL, choice_probabilities, generate
from scripts.run_grader import LETTER_TO_LABEL, build_prompt, format_rubric, model_slug

METHODS = ["first_token", "verbalized", "self_consistency", "verifier"]
SELF_CONSISTENCY_RUNS = 5
SELF_CONSISTENCY_TEMPERATURE = 0.7


def parse_letter(text):
    """Find the first standalone option letter (A/B/C/D) in free text."""
    match = re.search(r"\b([ABCD])\b", text)
    return match.group(1) if match else None


def parse_verbalized(text):
    """Pull the label letter and a 0-100 confidence number out of a
    verbalized reply like "B\nConfidence: 80". Returns (letter, confidence
    as a 0-1 float), either of which is None if it couldn't be found."""
    letter = parse_letter(text)

    number_match = re.search(r"confidence[:\s]*?(\d{1,3})", text, re.IGNORECASE)
    if number_match is None:
        number_match = re.search(r"(\d{1,3})", text)

    confidence = None
    if number_match is not None:
        value = max(0, min(100, int(number_match.group(1))))
        confidence = value / 100.0

    return letter, confidence


def make_record(task, predicted_label, confidence, gold_label=None):
    """The shared result shape every method saves, so they can all be
    compared with the same metrics code."""
    return {
        "task_id": task["task_id"],
        "gold_label": gold_label if gold_label is not None else task["gold_label"],
        "predicted_label": predicted_label,
        "confidence": confidence,
    }


def run_first_token(tasks, model):
    """Reuse the existing v2+prefill run instead of re-querying the model -
    it already is exactly this method (confidence = top renormalized
    probability from prefilled logprobs)."""
    source_path = "results/grader_v2_prefill.json" if model == DEFAULT_MODEL \
        else f"results/grader_v2_prefill_{model_slug(model)}.json"
    source = {r["task_id"]: r for r in json.load(open(source_path, encoding="utf-8"))}

    records = []
    for task in tasks:
        r = source[task["task_id"]]
        confidence = max(r["probabilities"].values()) if r["probabilities"] else None
        records.append(make_record(task, r["predicted_label"], confidence, gold_label=r["gold_label"]))
    return records


def build_verbalized_prompt(task):
    """v2 prompt, but also asking for a stated 0-100 confidence."""
    return f"""Question: {task['question']}
Reference answer: {task['reference_answer']}
Student answer: {task['student_answer']}

Rubric:
{format_rubric()}

First answer with one letter only: A, B, C, or D.
Then on a new line, write "Confidence: " followed by a number from 0 to 100
for how confident you are that this label is correct.
"""


def run_verbalized(tasks, model):
    records = []
    for i, task in enumerate(tasks, start=1):
        print(f"[{i}/{len(tasks)}] verbalized: grading {task['task_id']}...")
        reply = generate(build_verbalized_prompt(task), model=model, max_tokens=30)
        letter, confidence = parse_verbalized(reply)
        predicted_label = LETTER_TO_LABEL.get(letter)
        records.append(make_record(task, predicted_label, confidence))
    return records


def run_self_consistency(tasks, model):
    """Sample the v2 prompt 5 times at temperature 0.7 (a different seed
    each run, for reproducible-but-varied sampling) and take the majority
    label. Confidence is how many of the 5 runs agreed with that label."""
    prompt_cache = {}
    records = []
    for i, task in enumerate(tasks, start=1):
        print(f"[{i}/{len(tasks)}] self_consistency: grading {task['task_id']}...")
        if task["task_id"] not in prompt_cache:
            prompt_cache[task["task_id"]] = build_prompt(task, "v2")
        prompt = prompt_cache[task["task_id"]]

        letters = []
        for run in range(SELF_CONSISTENCY_RUNS):
            reply = generate(prompt, model=model, max_tokens=10,
                              temperature=SELF_CONSISTENCY_TEMPERATURE, seed=42 + run)
            letter = parse_letter(reply)
            if letter is not None:
                letters.append(letter)

        if letters:
            majority_letter, majority_count = Counter(letters).most_common(1)[0]
            predicted_label = LETTER_TO_LABEL[majority_letter]
            confidence = majority_count / SELF_CONSISTENCY_RUNS
        else:
            predicted_label = None
            confidence = None

        records.append(make_record(task, predicted_label, confidence))
    return records


def build_verifier_prompt(task, predicted_label):
    """Show the label first_token chose and ask whether it's correct."""
    return f"""Question: {task['question']}
Reference answer: {task['reference_answer']}
Student answer: {task['student_answer']}

A grader labeled this student answer as: {predicted_label}

Is this label correct? Answer Yes or No.
"""


def run_verifier(tasks, model):
    """Needs first_token's predicted labels to verify, so it runs that
    method first (cheaply, from the cached file) if not already available."""
    first_token_records = {r["task_id"]: r for r in run_first_token(tasks, model)}

    records = []
    for i, task in enumerate(tasks, start=1):
        print(f"[{i}/{len(tasks)}] verifier: grading {task['task_id']}...")
        predicted_label = first_token_records[task["task_id"]]["predicted_label"]
        if predicted_label is None:
            records.append(make_record(task, None, None))
            continue

        prompt = build_verifier_prompt(task, predicted_label)
        result = choice_probabilities(prompt, ["Yes", "No"], model=model, prefill="Answer: ")
        confidence = result["probs"]["Yes"] if result["probs"] is not None else None
        records.append(make_record(task, predicted_label, confidence))
    return records


RUNNERS = {
    "first_token": run_first_token,
    "verbalized": run_verbalized,
    "self_consistency": run_self_consistency,
    "verifier": run_verifier,
}


def main():
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in METHODS:
        print(f"Usage: python -m scripts.run_confidence_method <{'|'.join(METHODS)}> [model]")
        sys.exit(1)

    method = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) == 3 else DEFAULT_MODEL
    tasks = json.load(open("data/tasks.json", encoding="utf-8"))

    records = RUNNERS[method](tasks, model)

    out_path = f"results/confidence_{method}.json" if model == DEFAULT_MODEL \
        else f"results/confidence_{method}_{model_slug(model)}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    correct = sum(1 for r in records if r["predicted_label"] == r["gold_label"])
    missing = sum(1 for r in records if r["confidence"] is None)
    print(f"\n{method} ({model}) accuracy: {correct}/{len(records)} = {correct / len(records):.3f}")
    if missing:
        print(f"Warning: {missing} record(s) have no usable confidence value")
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
