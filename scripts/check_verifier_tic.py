# Check whether the verifier method's first-token confidence actually
# reflects what the model concludes, or just a "No." discourse-opener tic
# disconnected from the real verdict (a discourse-opener tic is when a
# sentence starts with a word like "No" as a rhetorical/contrastive habit,
# not because that's the actual answer - e.g. "No, actually this is
# correct because...").
#
# Two checks, both on the same first 20 tasks:
#   1. Generate the verifier's FULL answer (not just the first token) and
#      compare the stated first word against what the full explanation
#      actually concludes.
#   2. Compare two prefills for reading the verdict from logprobs:
#      "Answer: " (forces the very next token to be Yes/No) vs
#      "Answer: Yes/No because" (as literally specified) - report captured
#      mass and AUROC for each.
#
# Usage: python -m scripts.check_verifier_tic
import json

from src.llm import DEFAULT_MODEL, choice_probabilities, generate
from scripts.run_confidence_method import build_verifier_prompt, run_first_token
from scripts.compare_confidence import auroc

N_TASKS = 20


def check_first_token_vs_full_answer(tasks, first_token_by_id, model):
    """For each task, generate the verifier's full free-text answer (no
    prefill, no token cap) and record its literal first word, alongside
    the full text so a human can read what it actually concludes."""
    records = []
    for i, task in enumerate(tasks, start=1):
        label = first_token_by_id[task["task_id"]]["predicted_label"]
        prompt = build_verifier_prompt(task, label)
        full_answer = generate(prompt, model=model, max_tokens=100)
        first_word = full_answer.strip().split()[0].strip(".,:").lower() if full_answer.strip() else ""
        print(f"[{i}/{len(tasks)}] {task['task_id']} (label verified: {label})")
        print(f"  first word: {first_word!r}")
        print(f"  full answer: {full_answer!r}")
        records.append({
            "task_id": task["task_id"],
            "label_verified": label,
            "gold_label": task["gold_label"],
            "first_word": first_word,
            "full_answer": full_answer,
        })
    return records


def check_prefill_variant(tasks, first_token_by_id, model, prefill):
    """Read P(Yes) via choice_probabilities with the given prefill, for
    each task, and return (records, avg_mass, auc)."""
    records = []
    for task in tasks:
        r = first_token_by_id[task["task_id"]]
        label = r["predicted_label"]
        prompt = build_verifier_prompt(task, label)
        result = choice_probabilities(prompt, ["Yes", "No"], model=model, prefill=prefill)
        confidence = result["probs"]["Yes"] if result["probs"] is not None else None
        is_correct = label == r["gold_label"]
        records.append({
            "task_id": task["task_id"],
            "mass": result["mass"],
            "confidence": confidence,
            "is_correct": is_correct,
        })

    avg_mass = sum(rec["mass"] for rec in records) / len(records)
    scored = [(rec["confidence"], rec["is_correct"]) for rec in records if rec["confidence"] is not None]
    auc = auroc(scored) if scored else None
    return records, avg_mass, auc


def main():
    model = DEFAULT_MODEL
    all_tasks = json.load(open("data/tasks.json", encoding="utf-8"))
    tasks = all_tasks[:N_TASKS]
    first_token_by_id = {r["task_id"]: r for r in run_first_token(all_tasks, model)}

    print("=" * 70)
    print("PART 1: first token vs. what the full answer actually concludes")
    print("=" * 70)
    full_answer_records = check_first_token_vs_full_answer(tasks, first_token_by_id, model)
    with open("results/verifier_tic_full_answers.json", "w", encoding="utf-8") as f:
        json.dump(full_answer_records, f, indent=2)

    print()
    print("=" * 70)
    print("PART 2: prefill variants - captured mass and AUROC")
    print("=" * 70)
    for prefill in ["Answer: ", "Answer: Yes/No because"]:
        records, avg_mass, auc = check_prefill_variant(tasks, first_token_by_id, model, prefill)
        auc_str = f"{auc:.3f}" if auc is not None else "n/a"
        print(f"prefill={prefill!r}: avg mass={avg_mass:.3f}, AUROC={auc_str}")
        safe_name = prefill.strip().replace(":", "").replace("/", "-").replace(" ", "_").lower()
        with open(f"results/verifier_tic_prefill_{safe_name}.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)


if __name__ == "__main__":
    main()
