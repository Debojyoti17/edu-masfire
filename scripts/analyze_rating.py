# Analyse the human rating of feedback (results/human_rating/rater_A.csv and rater_B.csv, filled in
# by two team members; the cleaned versions from scripts/clean_ratings.py are used when they exist):
# share of helpful / neutral / harmful per rater, how often the raters agree, and Cohen's kappa
# (agreement beyond chance: 0 = chance, 1 = perfect; above 0.6 is usually called substantial).
# A rater who only said helpful / not helpful (not helpful = neutral or harmful) is compared with the
# other rater on that two-level scale. Rows where the raters disagree are listed so the team can
# discuss them afterwards.
#
# Usage: python -m scripts.analyze_rating
import csv
import os
from collections import Counter

from sklearn.metrics import cohen_kappa_score

DIR = os.path.join("results", "human_rating")
RATINGS = ["helpful", "neutral", "harmful", "not helpful"]


def read(rater):
    """{row: rating} for one rater's sheet (cleaned version if there is one); blank rows left out."""
    path = os.path.join(DIR, f"{rater}_clean.csv")
    if not os.path.exists(path):
        path = os.path.join(DIR, f"{rater}.csv")
    with open(path, encoding="utf-8-sig", newline="") as f:
        out = {}
        for r in csv.DictReader(f):
            rating = (r["rating"] or "").strip().lower()
            if rating:
                if rating not in RATINGS:
                    raise SystemExit(f"{rater} row {r['row']}: '{r['rating']}' is not one of {RATINGS}")
                out[r["row"]] = rating
        return out


def two_level(rating):
    """helpful stays helpful; neutral, harmful and not helpful all become 'not helpful'."""
    return "helpful" if rating == "helpful" else "not helpful"


def main():
    """Print the summary and save it to results/human_rating/summary.md."""
    a, b = read("rater_A"), read("rater_B")
    lines = ["# Human rating of feedback", ""]
    for name, ratings in [("Rater A", a), ("Rater B", b)]:
        c = Counter(ratings.values())
        lines.append(f"- {name}: {len(ratings)} of 60 rated; " + ", ".join(f"{k} {c[k]} ({c[k] / len(ratings):.0%})"
                                                                         for k in RATINGS if c[k]))
    rows = sorted(set(a) & set(b), key=int)
    coarse = any(v == "not helpful" for v in list(a.values()) + list(b.values()))
    if coarse:  # compare on the two-level scale
        x, y = [two_level(a[r]) for r in rows], [two_level(b[r]) for r in rows]
        scale = "two-level scale (helpful / not helpful)"
    else:
        x, y = [a[r] for r in rows], [b[r] for r in rows]
        scale = "three-level scale"
    agree = sum(p == q for p, q in zip(x, y))
    kappa = cohen_kappa_score(x, y)
    lines += [f"- Rated by both: {len(rows)} messages; compared on the {scale}",
              f"- Same rating: {agree} of {len(rows)} ({agree / len(rows):.0%}); Cohen's kappa {kappa:.2f}",
              f"- Helpful according to both: {sum(p == q == 'helpful' for p, q in zip(x, y))}; "
              f"not helpful according to both: {sum(p == q != 'helpful' for p, q in zip(x, y))}",
              "", "Rows where the raters disagree: " + ", ".join(f"{r} ({p}/{q})" for r, p, q in zip(rows, x, y) if p != q)]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
