# Live demo: one student answer goes through Planner -> Tutor -> Grader -> Feedback, showing
# each agent's output; then one injected fault is run on the same answer to show what changes
# and whether the two safety checks would stop wrong feedback reaching the student.
# Outputs are deterministic (temperature 0), so the demo gives the same result every time.
#
# Usage:
#   python -m scripts.demo --suggest                        list good demo answers from saved results
#   python -m scripts.demo --task 12 --fault tutor_overclaim
#   python -m scripts.demo --task 12 --fault tutor_overclaim --save    also save a transcript
#   python -m scripts.demo --task 12 --fault grader_label_flip --model gemma2:9b
import json
import os
import sys
import textwrap
import time

from src.faults import FAULTS, donor_index
from src.llm import DEFAULT_MODEL
from scripts.run_pipeline import load_jsonl, run_one

CONF_THRESHOLD = 0.9
WIDTH = 92
transcript = []


def say(text=""):
    """Print a line and keep a copy for the saved transcript."""
    print(text, flush=True)
    transcript.append(text)


def wrapped(label, text, indent=4):
    """Print a labelled block of text wrapped to the screen width."""
    say(f"  {label}")
    for para in (text or "(empty)").splitlines():
        for line in textwrap.wrap(para, WIDTH - indent) or [""]:
            say(" " * indent + line)


def verdict_line(rec):
    """One line saying what the student is told and whether it is right."""
    ok = "RIGHT" if rec["delivered"] == rec["gold_label"] else "WRONG"
    return f"Student is told: {rec['delivered']}   (correct label: {rec['gold_label']})  ->  {ok}"


def show_grader(rec):
    """Show the Grader's label and its probability for each label as a small text bar chart."""
    say(f"  Grader label: {rec['grader_label']}   (confidence {rec['grader_conf']:.2f})")
    for label, p in sorted((rec["grader_probs"] or {}).items(), key=lambda kv: -kv[1]):
        say(f"      {label:30} {p:5.2f}  " + "#" * round(p * 30))


def show_checks(rec):
    """Show what each safety check would do with this feedback."""
    conf_held = rec["grader_conf"] is None or rec["grader_conf"] < CONF_THRESHOLD
    agree_held = rec["delivered"] != rec["check_label"]
    say(f"  Safety check 1 - confidence below {CONF_THRESHOLD}: "
        + ("HELD BACK for a teacher" if conf_held else "sent to student"))
    say(f"  Safety check 2 - second grader says '{rec['check_label']}': "
        + ("disagrees -> HELD BACK for a teacher" if agree_held else "agrees -> sent to student"))


def run_and_show(task, fault, model, base=None, donor_task=None, donor_base=None):
    """Run the pipeline once (fault-free or with one fault) and print every stage."""
    start = time.time()
    rec = run_one(task, fault, model, base, donor_task, donor_base)
    say(f"  ({time.time() - start:.1f} s)")
    wrapped("1. Planner - key points a full answer needs:", rec["plan"])
    wrapped("2. Tutor - how the answer compares:", rec["notes"])
    say("  3. Grader")
    show_grader(rec)
    if rec["sent_label"] != rec["grader_label"]:
        say(f"  !! Label reaching the Feedback agent was corrupted to: {rec['sent_label']}")
    wrapped("4. Feedback - what the student reads:", rec["feedback"])
    say("  " + verdict_line(rec))
    show_checks(rec)
    return rec


def suggest(model):
    """List answers where the fault-free pipeline is right and a fault makes it wrong, and say
    whether the agreement check catches it - good demo material."""
    folder = os.path.join("results", "pipeline", model.replace(":", "-"))
    tasks = json.load(open("data/tasks.json", encoding="utf-8"))
    index = {t["task_id"]: i for i, t in enumerate(tasks)}
    base = {r["task_id"]: r for r in load_jsonl(os.path.join(folder, "none.jsonl"))}
    print(f"Answers where the fault-free pipeline is RIGHT but the fault makes it WRONG ({model}):")
    for fault in FAULTS:
        if fault == "none":
            continue
        hits = []
        for r in load_jsonl(os.path.join(folder, f"{fault}.jsonl")):
            b = base[r["task_id"]]
            if b["delivered"] == b["gold_label"] and r["delivered"] != r["gold_label"]:
                caught = "caught by agreement check" if r["delivered"] != r["check_label"] else "NOT caught"
                hits.append(f"--task {index[r['task_id']]} ({caught})")
        print(f"  {fault:20} " + (", ".join(hits[:4]) if hits else "none"))


def main():
    """Read the options and run the demo."""
    args = sys.argv[1:]

    def option(name, default=None):
        """Value that follows a command-line flag, or the default."""
        return args[args.index(name) + 1] if name in args else default

    model = option("--model", DEFAULT_MODEL)
    if "--suggest" in args:
        suggest(model)
        return
    fault = option("--fault")
    if fault is not None and fault not in FAULTS:
        sys.exit("Unknown fault. Choose from: " + ", ".join(f for f in FAULTS if f != "none"))
    tasks = json.load(open("data/tasks.json", encoding="utf-8"))
    i = int(option("--task", "0"))
    task = tasks[i]

    say("=" * WIDTH)
    say(f"MULTI-AGENT GRADING DEMO   model: {model}   task #{i} ({task['task_id']})")
    say("=" * WIDTH)
    wrapped("Question:", task["question"])
    wrapped("Reference answer:", task["reference_answer"])
    wrapped("Student answer:", task["student_answer"])

    say("\n" + "-" * WIDTH)
    say("RUN 1: no fault")
    say("-" * WIDTH)
    base = run_and_show(task, "none", model)

    if fault and fault != "none":
        donor_task = donor_base = None
        if fault in ("planner_wrong_task", "tutor_wrong_task", "reference_swap"):
            donor_task = tasks[donor_index(tasks, i)]
            if fault != "reference_swap":
                say(f"\n  (preparing the misrouted message from task {donor_task['task_id']} ...)")
                donor_base = run_one(donor_task, "none", model, None, None, None)
        say("\n" + "-" * WIDTH)
        say(f"RUN 2: injected fault '{fault}': {FAULTS[fault]}")
        say("-" * WIDTH)
        rec = run_and_show(task, fault, model, base, donor_task, donor_base)
        say("\n" + "=" * WIDTH)
        say("WHAT THE FAULT CHANGED")
        say(f"  Without fault: {base['delivered']}   With fault: {rec['delivered']}   Correct: {task['gold_label']}")
        if base["delivered"] == rec["delivered"]:
            say("  The verdict the student reads did not change.")
        say("=" * WIDTH)

    if "--save" in args:
        os.makedirs(os.path.join("results", "demo"), exist_ok=True)
        path = os.path.join("results", "demo", f"demo_{model.replace(':', '-')}_task{i}_{fault or 'none'}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(transcript) + "\n")
        print(f"\nTranscript saved to {path}")


if __name__ == "__main__":
    main()
