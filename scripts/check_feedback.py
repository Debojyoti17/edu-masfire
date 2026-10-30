# Does the feedback praise wrong answers? Compares the feedback of the final run (prompt v2) with
# the per-verdict prompt (v3, src/pipeline_v2.py FEEDBACK_GUIDE) on the 120-task set with no fault:
# same grade, same Tutor notes, only the Feedback prompt differs. The new feedback is stored in the
# pipeline's text cache, so later runs and the demo reuse it.
#
# Praise is detected with a fixed list of praising phrases ("great", "well done", "you correctly",
# ...). It is a simple, transparent check, not a full judgement of quality: the side-by-side file lets
# a person read the actual messages.
# Writes results/feedback_check/summary.md and side_by_side.csv.
# Usage: python -m scripts.check_feedback
import csv
import json
import os
import re

from scripts.run_pipeline import load_jsonl
from scripts.run_pipeline_v2 import CACHE_DIR, FEEDBACK_MODEL, key_of, load_cache, save_to_cache
from src.pipeline_v2 import FEEDBACK_VERSION, write_feedback

RUN = os.path.join("results", "pipeline_v2", "120_robust_full_refsim_deberta-v3-base-nli", "none_on.jsonl")
OUT_DIR = os.path.join("results", "feedback_check")
PRAISE = ["great", "good job", "well done", "excellent", "correctly", "you're right", "you are right", "you've got it",
          "nice", "good observation", "good understanding", "good point", "good work", "keep up", "you identified",
          "you recognized", "you noted", "good thinking", "on the right track", "fantastic", "awesome"]
WRONG = ["contradictory", "irrelevant"]


def advice(feedback):
    """The advice text of a feedback message (without the fixed Verdict line)."""
    return feedback.split("\n", 1)[1] if "\n" in feedback else feedback


def praises(feedback):
    """True if the advice contains a praising phrase."""
    text = advice(feedback).lower()
    return any(re.search(r"\b" + re.escape(p) + r"\b", text) for p in PRAISE)


def main():
    """Write the new feedback where it is missing, then compare old and new."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = {t["task_id"]: t for t in json.load(open(os.path.join("data", "tasks.json"), encoding="utf-8"))}
    text_cache = load_cache("text")
    rows = []
    for n, rec in enumerate(r for r in load_jsonl(RUN) if r.get("feedback")):
        task, sent = tasks[rec["task_id"]], rec["label_sent"]
        notes = text_cache[key_of("tutor", FEEDBACK_MODEL, task["question"], task["reference_answer"], task["student_answer"])]
        key = key_of(FEEDBACK_VERSION, FEEDBACK_MODEL, task["question"], task["reference_answer"],
                     task["student_answer"], str(sent), notes)
        if key not in text_cache:
            save_to_cache("text", key, list(write_feedback(task, sent, notes, FEEDBACK_MODEL)), text_cache)
        new, matches = text_cache[key]
        rows.append({"task_id": rec["task_id"], "verdict": sent, "teacher_grade": rec["gold_label"],
                     "student_answer": task["student_answer"], "old_feedback": rec["feedback"], "new_feedback": new,
                     "old_praises": praises(rec["feedback"]), "new_praises": praises(new), "verdict_kept": matches})
        if (n + 1) % 20 == 0:
            print(f"  {n + 1} done", flush=True)

    lines = [f"# Does the feedback praise wrong answers? (120-task set, no fault, {len(rows)} messages)", "",
             "| Verdict | Messages | Praise, old prompt (v2) | Praise, new prompt (v3) |", "|---|---|---|---|"]
    for group, verdicts in [("contradictory", ["contradictory"]), ("irrelevant", ["irrelevant"]),
                            ("both wrong-answer verdicts", WRONG),
                            ("partially correct", ["partially_correct_incomplete"]), ("correct", ["correct"])]:
        g = [r for r in rows if r["verdict"] in verdicts]
        if g:
            lines.append(f"| {group} | {len(g)} | {sum(r['old_praises'] for r in g)} ({sum(r['old_praises'] for r in g) / len(g):.0%}) "
                         f"| {sum(r['new_praises'] for r in g)} ({sum(r['new_praises'] for r in g) / len(g):.0%}) |")
    lines += ["", f"Verdict line kept in every new message: {all(r['verdict_kept'] for r in rows)}",
              "Praise for correct and partially correct answers is expected; for contradictory and irrelevant answers it is the problem.",
              "Read results/feedback_check/side_by_side.csv for the actual messages."]
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(OUT_DIR, "side_by_side.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["verdict"] not in WRONG, r["verdict"])))


if __name__ == "__main__":
    main()
