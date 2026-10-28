# Test material for the photo (handwritten answer sheet) prototype: picks 30 student answers from
# the 120-task set (about 7-8 per teacher grade, seed 42) for team members to copy by hand and
# photograph. Because we know the exact text of each answer, the photos let us measure how well the
# vision model reads handwriting and whether grading from a photo gives the same grade as from text.
#
# Writes results/handwriting/to_write.csv (code + answer to copy) and INSTRUCTIONS.md.
# Usage: python -m scripts.make_handwriting_sheet
import csv
import json
import os
import random

OUT_DIR = os.path.join("results", "handwriting")
COUNTS = {"correct": 8, "partially_correct_incomplete": 8, "contradictory": 7, "irrelevant": 7}
INSTRUCTIONS = """# Handwritten answer sheets (about 1 hour for the team)

1. Open `to_write.csv`. Each row has a code (H01-H30) and a student answer.
2. On paper, write the code at the top, then copy the student answer exactly as it is (keep the
   student's own spelling mistakes). Use your normal handwriting, blue or black pen, one answer per
   piece of paper (half an A4 sheet is fine). Different team members can write different rows - that
   makes the test more realistic.
3. Photograph each sheet with a phone: good light, no shadow across the text, paper filling most of
   the picture, not tilted much.
4. Name each photo by its code (`H01.jpg`, `H02.jpg`, ...) and put them all in
   `results/handwriting/photos/` on this laptop.
5. Write in `who_wrote.csv` (two columns: code, writer) which team member wrote each one, so we can
   see if some handwriting is harder to read.
"""


def main():
    """Pick the 30 answers and write the sheet and instructions."""
    tasks = json.load(open(os.path.join("data", "tasks.json"), encoding="utf-8"))
    rng = random.Random(42)
    picked = []
    for label, n in COUNTS.items():
        pool = [(i, t) for i, t in enumerate(tasks) if t["gold_label"] == label]
        picked += rng.sample(pool, n)
    rng.shuffle(picked)
    os.makedirs(os.path.join(OUT_DIR, "photos"), exist_ok=True)
    with open(os.path.join(OUT_DIR, "to_write.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "task_number", "task_id", "student_answer_to_copy"])
        for k, (i, t) in enumerate(picked, start=1):
            w.writerow([f"H{k:02d}", i, t["task_id"], t["student_answer"]])
    with open(os.path.join(OUT_DIR, "INSTRUCTIONS.md"), "w", encoding="utf-8") as f:
        f.write(INSTRUCTIONS)
    with open(os.path.join(OUT_DIR, "who_wrote.csv"), "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows([["code", "writer"]] + [[f"H{k:02d}", ""] for k in range(1, len(picked) + 1)])
    print(f"Wrote {len(picked)} answers to {OUT_DIR}/to_write.csv; photos go in {OUT_DIR}/photos/")


if __name__ == "__main__":
    main()
