# Run one LLM grader of the redesigned pipeline over a task file, in one of three modes:
#   same      - with 8 teacher-graded answers to the same question (retrieval Planner)
#   similar   - with 8 similar graded answers from other questions (for unseen questions)
#   zeroshot  - no examples (the original v2 prompt)
# Saves one line per task to results/rag/<set>_<mode>_<model>.jsonl and resumes if stopped.
#
# Usage: python -m scripts.run_rag_grader qwen2.5:7b data/eval_test_ua.json same
import json
import os
import sys
import time

from src.graders import grade_with_prompt, rag_prompt
from src.retrieval import retrieve
from scripts.run_grader import build_prompt, model_slug
from scripts.run_pipeline import load_jsonl

MODES = ["same", "similar", "zeroshot"]


def output_path(model, tasks_path, mode):
    """Where this model + task set + mode is saved."""
    set_name = os.path.splitext(os.path.basename(tasks_path))[0].replace("eval_", "")
    return os.path.join("results", "rag", f"{set_name}_{mode}_{model_slug(model)}.jsonl")


def grade_one(task, model, mode):
    """Grade one task and return the record we save."""
    start = time.time()
    examples = [] if mode == "zeroshot" else retrieve(task, mode)
    prompt = build_prompt(task, "v2") if mode == "zeroshot" else rag_prompt(task, examples, mode)
    label, probs, mass = grade_with_prompt(prompt, model)
    return {"task_id": task["task_id"], "gold_label": task["gold_label"], "model": model, "mode": mode,
            "predicted_label": label, "probabilities": probs, "mass": mass,
            "example_ids": [e["id"] for e in examples], "seconds": round(time.time() - start, 3)}


def main():
    """Read the options, grade every task not yet saved, and print the accuracy."""
    if len(sys.argv) != 4 or sys.argv[3] not in MODES:
        sys.exit("Usage: python -m scripts.run_rag_grader <model> <task file> <same|similar|zeroshot>")
    model, tasks_path, mode = sys.argv[1:]
    tasks = json.load(open(tasks_path, encoding="utf-8"))
    path = output_path(model, tasks_path, mode)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = {r["task_id"] for r in load_jsonl(path)}
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"{model} {mode} on {tasks_path}: {len(todo)} of {len(tasks)} to run -> {path}", flush=True)
    start = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for k, task in enumerate(todo, start=1):
            f.write(json.dumps(grade_one(task, model, mode)) + "\n")
            f.flush()
            if k % 100 == 0 or k == len(todo):
                left = (len(todo) - k) * (time.time() - start) / k / 60
                print(f"  {k}/{len(todo)} done, about {left:.0f} min left", flush=True)
    recs = load_jsonl(path)
    acc = sum(r["predicted_label"] == r["gold_label"] for r in recs) / len(recs)
    print(f"accuracy {acc:.3f} on {len(recs)} tasks", flush=True)


if __name__ == "__main__":
    main()
