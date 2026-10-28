# Human rating of feedback quality: picks 60 feedback messages written by the final pipeline (no
# faults, 120-task set; 15 per teacher grade, seed 42) and writes one rating sheet per rater plus the
# instructions. Two team members fill in their sheet separately; scripts/analyze_rating.py then
# measures how good the feedback is and how much the two raters agree.
#
# Usage: python -m scripts.make_rating_sheet
import csv
import json
import os
import random

from scripts.run_pipeline import load_jsonl

RUN = os.path.join("results", "pipeline_v2", "120_robust_full_refsim_deberta-v3-base-nli", "none_off.jsonl")
OUT_DIR = os.path.join("results", "human_rating")
PER_LABEL = 15
RATERS = ["rater_A", "rater_B"]
INSTRUCTIONS = """# How to rate the feedback (about 1 hour)

Each row is one student answer and the feedback our system would give the student.
Fill in the `rating` column with ONE word, and add a short `comment` if you like.
Work alone: do not look at the other rater's sheet or discuss rows until both are finished.

- **helpful**: accurate about the science, fits the student's actual answer, and gives a clear,
  specific next step. A student reading it would learn something.
- **neutral**: not wrong, but vague or generic ("keep up the good work", "add more detail"), or
  only partly fits the answer.
- **harmful**: says something scientifically wrong, contradicts the reference answer, praises a
  wrong idea, tells the student to change something that was correct, or is unkind or confusing.

Rate the advice text. The first line ("Verdict: ...") is the grade the system gave; if that grade
looks wrong to you, write "verdict wrong" in the comment, but rate the advice on its own merits.

Save the file as CSV (keep the same name) when you are done.
"""


def main():
    """Pick the messages and write rater_A.csv, rater_B.csv and INSTRUCTIONS.md."""
    tasks = {t["task_id"]: t for t in json.load(open(os.path.join("data", "tasks.json"), encoding="utf-8"))}
    recs = [r for r in load_jsonl(RUN) if r.get("feedback")]
    rng = random.Random(42)
    picked = []
    for label in ["correct", "partially_correct_incomplete", "contradictory", "irrelevant"]:
        pool = sorted((r for r in recs if r["gold_label"] == label), key=lambda r: r["task_id"])
        picked += rng.sample(pool, min(PER_LABEL, len(pool)))
    rng.shuffle(picked)
    os.makedirs(OUT_DIR, exist_ok=True)
    for rater in RATERS:
        with open(os.path.join(OUT_DIR, f"{rater}.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row", "task_id", "question", "reference_answer", "student_answer", "feedback", "rating", "comment"])
            for k, r in enumerate(picked, start=1):
                t = tasks[r["task_id"]]
                w.writerow([k, r["task_id"], t["question"], t["reference_answer"], t["student_answer"], r["feedback"], "", ""])
    with open(os.path.join(OUT_DIR, "INSTRUCTIONS.md"), "w", encoding="utf-8") as f:
        f.write(INSTRUCTIONS)
    print(f"Wrote {len(picked)} rows to {OUT_DIR}/rater_A.csv and rater_B.csv, plus INSTRUCTIONS.md")


if __name__ == "__main__":
    main()
