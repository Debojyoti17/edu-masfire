# Live demo of the redesigned pipeline: one student answer goes through every stage, and the
# screen shows what each stage did - the checks on the input, the Planner's worked examples, every
# grader's vote, the judge's grade and confidence, the safety layer, and the feedback the student
# would receive. With --fault, the same answer is then run with that fault injected, once with the
# safety layer OFF and once ON, to show what the fault does and whether the safety layer stops it.
# Outputs are deterministic (temperature 0) and mostly cached, so each run takes seconds.
#
# Usage:
#   python -m scripts.demo_v2 --suggest                       good demo answers for each fault
#   python -m scripts.demo_v2 --task 5                        one answer, no fault
#   python -m scripts.demo_v2 --task 5 --fault grader_lying   then the fault, safety layer off and on
#   python -m scripts.demo_v2 --task 5 --fault grader_lying --save    also save a transcript
# Faults: plan_lost, plan_misrouted, grader_crash, grader_lying, label_corrupted, reference_swap,
#         student_truncate, tutor_overclaim
import json
import os
import sys
import textwrap

from src.agents import LABEL_WORDS
from src.graders import grade_with_prompt, rag_prompt
from src.health import benched, canary_set, expected_canary_accuracy
from src.judge import LABELS, MODELS
from src.retrieval import retrieve
from scripts.grade_csv import load_judges
from scripts.run_grader import build_prompt
from scripts.run_pipeline import load_jsonl
from scripts.run_pipeline_v2 import (CACHE_DIR, FAULTS_V2, finish_task, key_of, lie, load_cache, prepare,
                                     save_to_cache, seed_grader_cache)

TASKS_PATH = os.path.join("data", "tasks.json")
WIDTH = 92
transcript = []


def say(text=""):
    """Print a line and keep a copy for the saved transcript."""
    print(text, flush=True)
    transcript.append(text)


def heading(text):
    """A step heading."""
    say()
    say(f"--- {text} " + "-" * max(0, WIDTH - len(text) - 5))


def wrapped(label, text, indent=4):
    """Print a labelled block of text wrapped to the screen width."""
    say(f"  {label}")
    for para in (text or "(empty)").splitlines():
        for line in textwrap.wrap(para, WIDTH - indent) or [""]:
            say(" " * indent + line)


def fill_cache(prompts, cache):
    """Make sure every grader prompt has an answer in the cache (asks the model if it is new)."""
    for model, _, prompt in prompts:
        if key_of(model, prompt) not in cache:
            save_to_cache("graders", key_of(model, prompt), list(grade_with_prompt(prompt, model)), cache)


def canary_benched(fault, cache):
    """Run the canary check for this fault: the (model, mode) graders that would be benched."""
    canaries = canary_set()
    out = set()
    for model in MODELS:
        for mode in ["same", "similar", "zeroshot"]:
            prompts = [(model, mode, build_prompt(c, "v2") if mode == "zeroshot" else rag_prompt(c, retrieve(c, mode), mode))
                       for c in canaries]
            fill_cache(prompts, cache)
            labels = [cache[key_of(model, p)][0] for _, _, p in prompts]
            if model == "qwen2.5:7b" and fault == "grader_lying":
                labels = [lie(l)[0] if l else l for l in labels]
            if benched(labels, [c["gold_label"] for c in canaries], expected_canary_accuracy(model, mode)):
                out.add((model, mode))
    return frozenset(out)


def run(tasks, i, fault, ft, judges, threshold, cache, text_cache, benched_graders):
    """One pass of the pipeline over task i. Returns (task the agents saw, regime, record)."""
    seen_task, regime, prompts, notes, flagged = prepare(tasks[i], tasks, i, fault, ft, full_checks=True)
    fill_cache(prompts, cache)
    rec = finish_task(seen_task, tasks[i], fault, ft, regime, prompts, cache, judges, threshold, text_cache, notes,
                      benched_graders=benched_graders if ft == "on" else frozenset(), input_flagged=flagged,
                      explain=True)
    return seen_task, regime, rec


def show_votes(rec):
    """Every grader's vote as label + confidence with a small text bar."""
    for name, probs in rec["inputs"].items():
        if probs is None:
            say(f"    {name:42} (no vote: no reply, or left out)")
            continue
        best = max(range(4), key=lambda k: probs[k])
        say(f"    {name:42} {LABEL_WORDS[LABELS[best]]:18} {probs[best]:4.2f}  " + "#" * round(probs[best] * 20))


def show_outcome(rec):
    """The judge's grade, the safety decision and what reaches the student."""
    say(f"  Judge's grade: {LABEL_WORDS[rec['final_label']]}   confidence {rec['confidence']:.2f} "
        f"(teacher review below {rec['threshold']:.2f})")
    for d in rec["detections"]:
        say(f"  Safety layer: {d}")
    if rec["ft"] == "on" and not rec["detections"]:
        say("  Safety layer: no problems found")
    right = rec["delivered"] == rec["gold_label"]
    if rec["escalated"]:
        say(f"  -> HELD BACK for the teacher (nothing wrong reaches the student)")
    else:
        say(f"  -> Student is told: {LABEL_WORDS.get(rec['delivered'], rec['delivered'])}   "
            f"(teacher's grade: {LABEL_WORDS[rec['gold_label']]})  ->  {'RIGHT' if right else 'WRONG'}")


def suggest():
    """From the saved final runs, answers where the fault turns a right grade into a wrong one with
    the safety layer off, and say what the safety layer does with them."""
    folder = os.path.join("results", "pipeline_v2", "120_robust_full_deberta-v3-base-nli")
    tasks = json.load(open(TASKS_PATH, encoding="utf-8"))
    index = {t["task_id"]: i for i, t in enumerate(tasks)}
    base = {r["task_id"]: r for r in load_jsonl(os.path.join(folder, "none_off.jsonl"))}
    say("Answers where the fault turns a RIGHT grade into a WRONG one (safety layer off):")
    for fault in FAULTS_V2:
        if fault == "none":
            continue
        off = load_jsonl(os.path.join(folder, f"{fault}_off.jsonl"))
        on = {r["task_id"]: r for r in load_jsonl(os.path.join(folder, f"{fault}_on.jsonl"))}
        picks = []
        for r in off:
            b = base.get(r["task_id"])
            if b and b["delivered"] == b["gold_label"] and r["delivered"] != r["gold_label"]:
                o = on[r["task_id"]]
                fate = "held" if o["escalated"] else ("fixed" if o["delivered"] == o["gold_label"] else "still wrong")
                picks.append(f"--task {index[r['task_id']]} ({fate})")
        say(f"  {fault:18} " + (", ".join(picks[:4]) if picks else "(none in the 120 tasks)"))
    say("\n(fixed = safety layer repairs it, held = sent to the teacher, still wrong = not caught)")


def main():
    """Read the options and run the demo."""
    args = sys.argv[1:]

    def option(name, default=None):
        """Value after a command-line flag, or the default."""
        return args[args.index(name) + 1] if name in args else default

    if "--suggest" in args:
        suggest()
        return
    if "--task" not in args:
        sys.exit("Usage: python -m scripts.demo_v2 --task N [--fault NAME] [--save]   or   --suggest")
    i, fault = int(option("--task")), option("--fault")
    if fault and fault not in FAULTS_V2:
        sys.exit(f"Unknown fault. Choose one of: {', '.join(f for f in FAULTS_V2 if f != 'none')}")
    tasks = json.load(open(TASKS_PATH, encoding="utf-8"))
    task = tasks[i]

    os.makedirs(CACHE_DIR, exist_ok=True)
    judges, threshold = load_judges()
    cache = {}
    for model in MODELS:
        seed_grader_cache(model, "test_ua", tasks, cache)
    cache.update(load_cache("graders"))
    text_cache = load_cache("text")

    say("=" * WIDTH)
    say(f"REDESIGNED PIPELINE - task {i} ({task['task_id']})")
    say("=" * WIDTH)
    wrapped("Question:", task["question"])
    wrapped("Reference answer (from the teacher):", task["reference_answer"])
    wrapped("Student answer:", task["student_answer"])
    say(f"  Teacher's grade (hidden from the system): {LABEL_WORDS[task['gold_label']]}")

    seen_task, regime, rec = run(tasks, i, "none", "on", judges, threshold, cache, text_cache, frozenset())
    heading("1. Planner: worked examples")
    if regime == "seen":
        say("  This question has teacher-graded answers, so the graders see 8 graded answers to the SAME question.")
    else:
        say("  New question: the graders see graded answers to SIMILAR questions, and also grade with no examples.")
    heading("2. Graders vote (3 local LLMs, a TF-IDF classifier and a fine-tuned DeBERTa model)")
    show_votes(rec)
    heading("3. Judge combines the votes; safety layer checks; Feedback writes the message")
    show_outcome(rec)
    if not rec["escalated"] and rec.get("feedback"):
        wrapped("Feedback the student receives:", rec["feedback"])

    if fault:
        say()
        say("=" * WIDTH)
        say(f"NOW WITH A FAULT: {FAULTS_V2[fault]}")
        say("=" * WIDTH)
        benched_graders = canary_benched(fault, cache)
        for ft in ["off", "on"]:
            seen_task, regime, rec = run(tasks, i, fault, ft, judges, threshold, cache, text_cache, benched_graders)
            heading(f"Safety layer {ft.upper()}")
            if seen_task["student_answer"] != task["student_answer"]:
                wrapped("Student answer the system received:", seen_task["student_answer"])
            if seen_task["reference_answer"] != task["reference_answer"]:
                wrapped("Reference answer the system received:", seen_task["reference_answer"])
            if ft == "on" and benched_graders:
                say("  Canary check before the session benched: "
                    + ", ".join(f"{m} ({mode})" for m, mode in sorted(benched_graders)))
            show_votes(rec)
            show_outcome(rec)

    if "--save" in args:
        os.makedirs(os.path.join("results", "demo"), exist_ok=True)
        path = os.path.join("results", "demo", f"demo_v2_task{i}_{fault or 'none'}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(transcript) + "\n")
        print(f"\nTranscript saved to {path}")


if __name__ == "__main__":
    main()
