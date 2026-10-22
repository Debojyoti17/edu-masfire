# Runs the full Planner -> Tutor -> Grader -> Feedback pipeline on every task, either fault-free
# or with one injected fault. Each finished task is saved immediately as one line in
# results/pipeline/<model>/<fault>.jsonl, so a stopped run picks up where it left off.
#
# Usage:
#   python -m scripts.run_pipeline all                        baseline + every fault, qwen2.5:7b
#   python -m scripts.run_pipeline all llama3.1:8b            same on llama3.1:8b
#   python -m scripts.run_pipeline tutor_overclaim            one fault (baseline must exist)
#   python -m scripts.run_pipeline all qwen2.5:7b --limit 3   quick test on the first 3 tasks
import json
import os
import sys
import time

from src.agents import LETTER_TO_LABEL, LETTERS, feedback, grader, parse_verdict, planner, tutor
from src.faults import FAULTS, OVERCLAIM_TEXT, corrupt_task, donor_index, flip_label
from src.llm import DEFAULT_MODEL, choice_probabilities
from scripts.run_grader import PREFILL_TEXT, build_prompt, model_slug

INPUT_FAULTS = {"reference_swap", "student_truncate"}
PLANNER_FAULTS = {"planner_omit", "planner_wrong_task"}


def independent_check(task, model):
    """A second, separate grader: the v2 prompt on its own, without any pipeline notes.
    Used later to test a 'two graders must agree' safety check."""
    result = choice_probabilities(build_prompt(task, "v2"), LETTERS, model=model, prefill=PREFILL_TEXT)
    if result["probs"] is None:
        return None, None
    probs = {LETTER_TO_LABEL[letter]: p for letter, p in result["probs"].items()}
    label = max(probs, key=probs.get)
    return label, probs[label]


def load_jsonl(path):
    """Read a results file written one JSON record per line. Missing file = no records yet."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_one(task, fault, model, base, donor_task, donor_base):
    """Run the pipeline on one task with one fault. 'base' is this task's fault-free record;
    any stage the fault cannot affect is copied from it instead of re-running the model."""
    start = time.time()
    seen = corrupt_task(task, fault, donor_task)
    fresh = fault == "none" or fault in INPUT_FAULTS

    if fault == "planner_omit":
        plan = ""
    elif fault == "planner_wrong_task":
        plan = donor_base["plan"]
    elif fresh:
        plan = planner(seen, model)
    else:
        plan = base["plan"]

    if fault == "tutor_wrong_task":
        notes = donor_base["notes"]
    elif fault == "tutor_overclaim":
        notes = OVERCLAIM_TEXT
    elif fresh or fault in PLANNER_FAULTS:
        notes = tutor(seen, plan, model)
    else:
        notes = base["notes"]

    if fault == "grader_label_flip":
        label, probs, mass = base["grader_label"], base["grader_probs"], base["grader_mass"]
    else:
        label, probs, mass = grader(seen, plan, notes, model)

    sent = flip_label(label) if fault == "grader_label_flip" else label
    fb = feedback(seen, sent, notes, model)

    if fresh:
        check_label, check_conf = independent_check(seen, model)
    else:
        check_label, check_conf = base["check_label"], base["check_conf"]

    return {
        "task_id": task["task_id"],
        "gold_label": task["gold_label"],
        "fault": fault,
        "model": model,
        "plan": plan,
        "notes": notes,
        "grader_label": label,
        "grader_probs": probs,
        "grader_conf": max(probs.values()) if probs else None,
        "grader_mass": mass,
        "sent_label": sent,
        "feedback": fb,
        "delivered": parse_verdict(fb),
        "check_label": check_label,
        "check_conf": check_conf,
        "seconds": round(time.time() - start, 2),
    }


def run_fault(tasks, fault, model, folder, step, steps):
    """Run one fault over every task, skipping tasks already saved, and print progress with
    a rough time-left estimate."""
    path = os.path.join(folder, f"{fault}.jsonl")
    done = {r["task_id"] for r in load_jsonl(path)}
    base = {}
    if fault != "none":
        base = {r["task_id"]: r for r in load_jsonl(os.path.join(folder, "none.jsonl"))}
        missing = [t["task_id"] for t in tasks if t["task_id"] not in base]
        if missing:
            sys.exit(f"Baseline is missing {len(missing)} tasks - run the 'none' fault first.")

    todo = [i for i, t in enumerate(tasks) if t["task_id"] not in done]
    print(f"\n[{step}/{steps}] {fault}: {FAULTS[fault]} - {len(todo)} of {len(tasks)} tasks to run", flush=True)
    times = []
    with open(path, "a", encoding="utf-8") as f:
        for k, i in enumerate(todo, start=1):
            task = tasks[i]
            d = donor_index(tasks, i)
            rec = run_one(task, fault, model, base.get(task["task_id"]), tasks[d], base.get(tasks[d]["task_id"]))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            times.append(rec["seconds"])
            if k % 10 == 0 or k == len(todo):
                left = (len(todo) - k) * sum(times) / len(times) / 60
                print(f"  {fault}: {k}/{len(todo)} done, about {left:.0f} min left for this fault", flush=True)


def main():
    """Read the command-line options, then run the requested fault(s) in order (baseline first)."""
    args = sys.argv[1:]
    limit = None
    if "--limit" in args:
        at = args.index("--limit")
        limit = int(args[at + 1])
        args = args[:at] + args[at + 2:]
    if not args or (args[0] != "all" and args[0] not in FAULTS):
        sys.exit(f"Usage: python -m scripts.run_pipeline <all|{'|'.join(FAULTS)}> [model] [--limit N]")

    which = list(FAULTS) if args[0] == "all" else [args[0]]
    model = args[1] if len(args) > 1 else DEFAULT_MODEL
    tasks = json.load(open("data/tasks.json", encoding="utf-8"))
    if limit:
        tasks = tasks[:limit]

    folder = os.path.join("results", "pipeline", model_slug(model) + (f"_limit{limit}" if limit else ""))
    os.makedirs(folder, exist_ok=True)
    print(f"Model {model}, {len(tasks)} tasks, saving to {folder}", flush=True)

    for step, fault in enumerate(which, start=1):
        run_fault(tasks, fault, model, folder, step, len(which))
    print("\nAll requested runs finished.", flush=True)


if __name__ == "__main__":
    main()
