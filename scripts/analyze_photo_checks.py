# Photo reading on the team's 28 handwriting photos: (1) reading errors before and after the rotation
# fix (results/handwriting/readings_v1_upright_only.csv vs readings.csv), split by writer and by how
# the photo was taken; (2) how well a low reading confidence picks out bad readings (more than 10% of
# characters wrong), and the confidence threshold for the photo check in src/health.py.
#
# Threshold rule (decided before looking at the second writer): the lowest threshold that catches
# every bad reading of the photos it is chosen on. Because there are only 28 photos, its quality is
# estimated across writers: choose it on one writer's photos, test it on the other's, and swap.
# Writes results/handwriting/photo_checks.md.
# Usage: python -m scripts.analyze_photo_checks
import csv
import os

from PIL import Image
from sklearn.metrics import roc_auc_score

DIR = os.path.join("results", "handwriting")
BAD = 0.1  # a reading with more than 10% of characters wrong counts as bad


def load(name):
    """Readings from one results file, with writer and photo orientation added."""
    with open(os.path.join(DIR, name), encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        w, h = Image.open(os.path.join(DIR, "photos", r["code"] + ".jpg")).size
        r["writer"] = "Sammo" if int(r["code"][1:]) <= 15 else "Prottoy"
        r["taken"] = "sideways" if h > w else "upright"
        r["cer"], r["conf"] = float(r["cer"]), float(r["reading_confidence"])
    return rows


def threshold_for(rows):
    """Lowest threshold that flags every bad reading: just above the most confident bad reading."""
    bad = [r["conf"] for r in rows if r["cer"] > BAD]
    return round(max(bad) + 0.001, 3) if bad else 0.0


def caught(rows, t):
    """(bad readings flagged, bad readings, good readings flagged, good readings)."""
    bad, good = [r for r in rows if r["cer"] > BAD], [r for r in rows if r["cer"] <= BAD]
    return sum(r["conf"] < t for r in bad), len(bad), sum(r["conf"] < t for r in good), len(good)


def main():
    """Print and save the comparison and the threshold."""
    before, after = load("readings_v1_upright_only.csv"), load("readings.csv")
    lines = ["# Photo reading: rotation fix and reading-confidence check (28 photos, 2 writers)", "",
             "| Photos | Characters wrong, upright reading only | Characters wrong, best of 4 rotations |", "|---|---|---|"]
    for name, keep in [("all 28", lambda r: True), ("taken upright", lambda r: r["taken"] == "upright"),
                       ("taken sideways", lambda r: r["taken"] == "sideways"),
                       ("writer Sammo", lambda r: r["writer"] == "Sammo"), ("writer Prottoy", lambda r: r["writer"] == "Prottoy")]:
        b, a = [r["cer"] for r in before if keep(r)], [r["cer"] for r in after if keep(r)]
        lines.append(f"| {name} ({len(a)}) | {sum(b) / len(b):.1%} | {sum(a) / len(a):.1%} |")
    bad = [r["cer"] > BAD for r in after]
    lines += ["", f"Bad readings (more than 10% of characters wrong) after the fix: {sum(bad)} of {len(after)}"]
    if 0 < sum(bad) < len(bad):
        lines.append(f"Low reading confidence picks out bad readings: AUROC {roc_auc_score(bad, [-r['conf'] for r in after]):.2f} "
                     "(1 = perfect, 0.5 = chance)")
    lines += ["", "| Threshold chosen on | Threshold | Tested on | Bad readings caught | Good readings flagged |",
              "|---|---|---|---|---|"]
    for fit, test in [("Sammo", "Prottoy"), ("Prottoy", "Sammo")]:
        t = threshold_for([r for r in after if r["writer"] == fit])
        c, nb, f, ng = caught([r for r in after if r["writer"] == test], t)
        lines.append(f"| {fit} | {t} | {test} | {c} of {nb} | {f} of {ng} |")
    final = threshold_for(after)
    c, nb, f, ng = caught(after, final)
    lines += ["", f"Threshold from all 28 photos (same rule): **{final}** -> set READING_CONFIDENCE_MIN in src/health.py. "
              f"On these 28 photos it flags {c} of {nb} bad and {f} of {ng} good readings; the cross-writer rows above "
              "are the fairer estimate for new handwriting."]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(DIR, "photo_checks.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
