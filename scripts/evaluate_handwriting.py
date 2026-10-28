# Test the photo prototype on the team's handwritten answer sheets (results/handwriting/photos/,
# made with the instructions from scripts/make_handwriting_sheet.py):
#   1. the vision model reads every photo; the reading is compared with the true text
#      (character error rate: share of characters wrong; 0 = perfect)
#   2. the same 30 answers are graded twice with scripts/grade_csv.py - once typed, once from the
#      photos - and the grades are compared
# Writes results/handwriting/readings.csv, class_typed.csv, class_photos.csv (and their _graded /
# _review files) and summary.md.
#
# Usage: python -m scripts.evaluate_handwriting
import csv
import json
import os
import subprocess
import sys

from src.ocr import VISION_MODEL, character_error_rate, read_photo

DIR = os.path.join("results", "handwriting")


def read_csv(path):
    """Rows of a CSV file as dictionaries."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    """Write rows (dicts) to a CSV file."""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def photo_of(code):
    """The photo file for a code (any common image extension), or None."""
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".heic"]:
        if os.path.exists(os.path.join(DIR, "photos", code + ext)):
            return f"photos/{code}{ext}"
    return None


def main():
    """Read the photos, grade typed vs photo versions and write the summary."""
    tasks = json.load(open(os.path.join("data", "tasks.json"), encoding="utf-8"))
    sheet = [r for r in read_csv(os.path.join(DIR, "to_write.csv")) if photo_of(r["code"])]
    if not sheet:
        sys.exit(f"No photos found in {DIR}/photos (expected H01.jpg, H02.jpg, ...)")
    # Where the writer did not copy the answer exactly, the reading is compared with what is really on
    # the paper (results/handwriting/as_written.csv, typed in by looking at the photo).
    written = {}
    if os.path.exists(os.path.join(DIR, "as_written.csv")):
        written = {r["code"]: r["text_on_paper"] for r in read_csv(os.path.join(DIR, "as_written.csv"))}
    readings = []
    for r in sheet:
        text, confidence = read_photo(os.path.join(DIR, photo_of(r["code"])))
        truth = written.get(r["code"], r["student_answer_to_copy"])
        cer = character_error_rate(truth, text)
        readings.append({"code": r["code"], "truth": truth, "reading": text,
                         "reading_confidence": "" if confidence is None else round(confidence, 3), "cer": round(cer, 3)})
        print(f"{r['code']}  CER {cer:.2f}  {text[:70]}", flush=True)
    write_csv(os.path.join(DIR, "readings.csv"), readings, ["code", "truth", "reading", "reading_confidence", "cer"])

    typed, photos = [], []
    for r in sheet:
        t = tasks[int(r["task_number"])]
        base = {"student": r["code"], "question": t["question"], "reference_answer": t["reference_answer"]}
        typed.append(dict(base, student_answer=r["student_answer_to_copy"]))
        photos.append(dict(base, student_answer="", photo=photo_of(r["code"])))
    write_csv(os.path.join(DIR, "class_typed.csv"), typed, ["student", "question", "reference_answer", "student_answer"])
    write_csv(os.path.join(DIR, "class_photos.csv"), photos,
              ["student", "question", "reference_answer", "student_answer", "photo"])
    for name in ["class_typed", "class_photos"]:
        subprocess.run([sys.executable, "-m", "scripts.grade_csv", os.path.join(DIR, name + ".csv"), "--no-feedback"],
                       check=True)

    graded_typed = {r["student"]: r for r in read_csv(os.path.join(DIR, "class_typed_graded.csv"))}
    graded_photo = {r["student"]: r for r in read_csv(os.path.join(DIR, "class_photos_graded.csv"))}
    gold = {r["code"]: tasks[int(r["task_number"])]["gold_label"] for r in sheet}
    words = {"correct": "correct", "partially_correct_incomplete": "partially correct",
             "contradictory": "contradictory", "irrelevant": "irrelevant"}
    n = len(sheet)
    same = sum(graded_typed[c]["grade"] == graded_photo[c]["grade"] for c in gold)
    right_typed = sum(graded_typed[c]["grade"] == words[gold[c]] for c in gold)
    right_photo = sum(graded_photo[c]["grade"] == words[gold[c]] for c in gold)
    held_photo = sum(graded_photo[c]["status"] == "teacher" for c in gold)
    flagged = sum("misread" in graded_photo[c]["reason"] for c in gold)
    wrong_sent = sum(graded_photo[c]["status"] == "auto" and graded_photo[c]["grade"] != words[gold[c]] for c in gold)
    wrong_sent_typed = sum(graded_typed[c]["status"] == "auto" and graded_typed[c]["grade"] != words[gold[c]] for c in gold)
    cers = [r["cer"] for r in readings]
    lines = [f"# Photo prototype on {n} handwritten answers (vision model {VISION_MODEL})", "",
             f"- Reading: average character error rate {sum(cers) / n:.3f}; perfect readings {sum(c == 0 for c in cers)} of {n}; "
             f"readings with more than 10% of characters wrong {sum(c > 0.1 for c in cers)}",
             f"- Same grade from the photo as from the typed text: {same} of {n}",
             f"- Grade matches the teacher's grade: typed {right_typed} of {n}, photo {right_photo} of {n}",
             f"- Photo version: {held_photo} held for the teacher ({flagged} flagged as possibly misread); "
             f"wrong grades sent automatically: {wrong_sent} (typed version: {wrong_sent_typed})", "",
             "Worst readings: " + "; ".join(f"{r['code']} (CER {r['cer']}): '{r['reading'][:60]}'"
                                            for r in sorted(readings, key=lambda r: -r["cer"])[:3])]
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
