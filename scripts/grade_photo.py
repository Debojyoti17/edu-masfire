# Grade one handwritten answer from a photo, live (for the demo: someone writes an answer on paper,
# we photograph it and the system reads and grades it). The photo is copied into examples/live_photo/,
# a one-row class spreadsheet is written for it, and scripts/grade_csv.py grades it as usual.
#
# Usage:
#   python -m scripts.grade_photo --questions              show the questions to choose from
#   python -m scripts.grade_photo PHOTO --question 1       grade the photo as an answer to question 1
#   python -m scripts.grade_photo latest --question 1      use the newest photo in Downloads / Pictures / Desktop
#   python -m scripts.grade_photo --code H16               one of the team's test photos, with its own question
#                                                          (from results/handwriting/to_write.csv) and the teacher's grade
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time

OUT_DIR = os.path.join("examples", "live_photo")
QUESTIONS = [92, 75, 4, 30]  # task numbers in data/tasks.json: harp strings, tree shadow, sound and movement, water volume
IMAGE_TYPES = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def newest_photo():
    """The most recently saved image in the usual places a phone photo lands on this laptop."""
    home = os.path.expanduser("~")
    folders = [os.path.join(home, "Downloads"), os.path.join(home, "Pictures"), os.path.join(home, "Pictures", "Camera Roll"),
               os.path.join(home, "OneDrive", "Pictures"), os.path.join(home, "Desktop")]
    files = [f for d in folders for t in IMAGE_TYPES for f in glob.glob(os.path.join(d, t))]
    if not files:
        sys.exit("No photo found in Downloads, Pictures or Desktop. Give the photo's path instead of 'latest'.")
    return max(files, key=os.path.getmtime)


def main():
    """Grade the photo and print what was read, the grade and the feedback."""
    tasks = json.load(open(os.path.join("data", "tasks.json"), encoding="utf-8"))
    args = sys.argv[1:]
    if not args or "--questions" in args:
        for k, n in enumerate(QUESTIONS, start=1):
            print(f"Question {k}: {tasks[n]['question']}\n")
        return
    teacher = None
    if "--code" in args:  # a team test photo: its question is found through the handwriting test sheet
        code = args[args.index("--code") + 1].upper()
        with open(os.path.join("results", "handwriting", "to_write.csv"), encoding="utf-8-sig", newline="") as f:
            row = next((r for r in csv.DictReader(f) if r["code"] == code), None)
        if row is None or not glob.glob(os.path.join("results", "handwriting", "photos", code + ".*")):
            sys.exit(f"No test photo {code} (see results/handwriting/photos/)")
        task, number = tasks[int(row["task_number"])], code
        source = glob.glob(os.path.join("results", "handwriting", "photos", code + ".*"))[0]
        teacher = task["gold_label"].replace("_", " ")
    else:
        number = int(args[args.index("--question") + 1]) if "--question" in args else 1
        task = tasks[QUESTIONS[number - 1]]
        source = newest_photo() if args[0] == "latest" else args[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    name = time.strftime("answer_%H%M%S") + os.path.splitext(source)[1].lower()
    shutil.copy(source, os.path.join(OUT_DIR, name))
    print(f"Photo: {source}\nQuestion {number}: {task['question']}\n", flush=True)
    sheet = os.path.join(OUT_DIR, "live_photo.csv")
    with open(sheet, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["student", "question", "reference_answer", "student_answer", "photo"])
        w.writerow(["Live photo", task["question"], task["reference_answer"], "", name])
    subprocess.run([sys.executable, "-m", "scripts.grade_csv", sheet], check=True)
    with open(os.path.join(OUT_DIR, "live_photo_graded.csv"), encoding="utf-8-sig", newline="") as f:
        r = next(csv.DictReader(f))
    print("\n" + "=" * 70)
    print(f"What the model read:  {r['student_answer']}")
    print(f"Reading confidence:   {r.get('reading_confidence', '')}")
    print(f"Grade:                {r['grade']}  (confidence {r['confidence']})")
    print(f"Status:               {r['status']}  {r['reason']}")
    print(f"Feedback:\n{r['feedback']}")
    if teacher:
        print(f"\nTeacher's grade for this answer: {teacher}")


if __name__ == "__main__":
    main()
