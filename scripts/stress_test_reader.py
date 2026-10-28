# Stress test of the photo reader (src/ocr.py) on synthetic answer sheets: the 30 answers of the
# handwriting test sheet are drawn in 5 handwriting-style fonts, once clean and once "hard" (blurrier,
# more tilted, faded pencil-grey ink, smaller), and the vision model reads all 300 images.
# Measures how often the reading is wrong (character error rate) and whether the misread-word check
# (src/health.py looks_misread) catches the REAL reading mistakes - until now it was only tested on
# simulated mistakes. Fonts are easier than real handwriting, so this complements the team's photos.
#
# Writes results/handwriting/stress/*.png, readings.csv and summary.md.
# Usage: python -m scripts.stress_test_reader
import csv
import json
import os
import random

from scripts.make_photo_example import draw_sheet
from src.health import looks_misread
from src.ocr import VISION_MODEL, character_error_rate, read_photo

OUT_DIR = os.path.join("results", "handwriting", "stress")
FONTS = ["Inkfree.ttf", "segoepr.ttf", "LHANDW.TTF", "segoesc.ttf", "MISTRAL.TTF"]
LEVELS = {"clean": {}, "hard": {"tilt": 6, "blur": 1.8, "ink": (120, 120, 130), "size": 34}}


def main():
    """Draw the sheets, read them all, and write readings.csv and summary.md."""
    tasks = json.load(open(os.path.join("data", "tasks.json"), encoding="utf-8"))
    with open(os.path.join("results", "handwriting", "to_write.csv"), encoding="utf-8-sig", newline="") as f:
        sheet = list(csv.DictReader(f))
    os.makedirs(OUT_DIR, exist_ok=True)
    done = {}
    out_csv = os.path.join(OUT_DIR, "readings.csv")
    if os.path.exists(out_csv):  # resume after an interruption
        with open(out_csv, encoding="utf-8-sig", newline="") as f:
            done = {r["image"]: r for r in csv.DictReader(f)}
    rows = []
    for level, settings in LEVELS.items():
        for font_file in FONTS:
            rng = random.Random(42)
            for r in sheet:
                image = f"{r['code']}_{font_file.split('.')[0]}_{level}.png"
                if image in done:
                    rows.append(done[image])
                    continue
                path = os.path.join(OUT_DIR, image)
                draw_sheet(r["code"], r["student_answer_to_copy"], font_file, rng, **settings).save(path)
                text, confidence = read_photo(path)
                rows.append({"image": image, "level": level, "font": font_file, "truth": r["student_answer_to_copy"],
                             "reading": text, "reading_confidence": "" if confidence is None else round(confidence, 3),
                             "cer": round(character_error_rate(r["student_answer_to_copy"], text), 3),
                             "flagged_misread": looks_misread(dict(tasks[int(r["task_number"])], student_answer=text))})
                print(f"{image}  CER {rows[-1]['cer']}  {text[:60]}", flush=True)
                with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)

    lines = [f"# Photo reader stress test ({VISION_MODEL}, {len(rows)} synthetic sheets)", "",
             "| Level | Font | Avg CER | Perfect | CER > 10% | Flagged as misread |", "|---|---|---|---|---|---|"]
    for level in LEVELS:
        for font_file in FONTS + ["all"]:
            group = [r for r in rows if r["level"] == level and font_file in ("all", r["font"])]
            cers = [float(r["cer"]) for r in group]
            flagged = sum(str(r["flagged_misread"]) == "True" for r in group)
            lines.append(f"| {level} | {font_file.split('.')[0]} | {sum(cers) / len(cers):.3f} | "
                         f"{sum(c == 0 for c in cers)}/{len(cers)} | {sum(c > 0.1 for c in cers)} | {flagged} |")
    bad = [r for r in rows if float(r["cer"]) > 0.1]
    good = [r for r in rows if float(r["cer"]) <= 0.1]
    caught = sum(str(r["flagged_misread"]) == "True" for r in bad)
    false_alarm = sum(str(r["flagged_misread"]) == "True" for r in good)
    lines += ["", f"Misread-word check on real reading mistakes: caught {caught} of {len(bad)} readings with more than "
              f"10% of characters wrong; false alarms on {false_alarm} of {len(good)} good readings."]
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
