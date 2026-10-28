# Backup example for the photo (handwritten answer sheet) demo, for use until the team's real photos
# exist: 6 student answers from the handwriting test sheet (results/handwriting/to_write.csv) are
# drawn in handwriting-style Windows fonts on lined "paper", slightly tilted, and saved as images.
# These are SYNTHETIC - a computer font is easier to read than real handwriting - so they only show
# that the photo step works, not how well it reads real handwriting.
#
# Writes examples/photo_quiz/P01.png ... P06.png and examples/photo_quiz/photo_quiz.csv (photo column
# filled, student_answer empty), which scripts/grade_csv.py can grade directly.
# Usage: python -m scripts.make_photo_example
#        python -m scripts.grade_csv examples/photo_quiz/photo_quiz.csv
import csv
import json
import os
import random
import textwrap

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT_DIR = os.path.join("examples", "photo_quiz")
FONTS = ["Inkfree.ttf", "segoepr.ttf", "LHANDW.TTF", "segoesc.ttf", "comic.ttf", "Inkfree.ttf"]
HOW_MANY = 6


def draw_sheet(code, text, font_file, rng, tilt=2.5, blur=0.8, ink=(20, 30, 110), size=44):
    """One answer drawn on lined paper with the code at the top, slightly tilted, as an image.
    tilt (degrees), blur, ink colour and font size can be changed to make the sheet harder to read."""
    width, line_gap = 1400, 70
    lines = textwrap.wrap(text, width=48)
    height = 220 + line_gap * max(3, len(lines))
    page = Image.new("RGB", (width, height), (250, 248, 240))
    pen = ImageDraw.Draw(page)
    for y in range(160, height - 20, line_gap):  # the ruled lines of the paper
        pen.line([(40, y + 50), (width - 40, y + 50)], fill=(170, 195, 225), width=2)
    font = ImageFont.truetype(os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", font_file), size)
    pen.text((60, 40), code, font=font, fill=ink)
    for k, line in enumerate(lines):
        pen.text((70 + rng.randint(-6, 6), 160 + k * line_gap), line, font=font, fill=ink)
    page = page.rotate(rng.uniform(-tilt, tilt), expand=True, fillcolor=(90, 90, 90))  # phone held at a slight angle
    return page.filter(ImageFilter.GaussianBlur(blur))  # a little camera blur


def main():
    """Draw the example sheets and write the CSV that points at them."""
    tasks = json.load(open(os.path.join("data", "tasks.json"), encoding="utf-8"))
    with open(os.path.join("results", "handwriting", "to_write.csv"), encoding="utf-8-sig", newline="") as f:
        sheet = list(csv.DictReader(f))[:HOW_MANY]
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(42)
    rows = []
    for k, (r, font_file) in enumerate(zip(sheet, FONTS), start=1):
        code = f"P{k:02d}"
        draw_sheet(code, r["student_answer_to_copy"], font_file, rng).save(os.path.join(OUT_DIR, code + ".png"))
        t = tasks[int(r["task_number"])]
        rows.append({"student": code, "question": t["question"], "reference_answer": t["reference_answer"],
                     "student_answer": "", "photo": code + ".png", "true_text": r["student_answer_to_copy"],
                     "teacher_grade_for_checking": t["gold_label"]})
    with open(os.path.join(OUT_DIR, "photo_quiz.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} synthetic answer-sheet images and {OUT_DIR}/photo_quiz.csv")


if __name__ == "__main__":
    main()
