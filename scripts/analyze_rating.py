# Analyse the human rating of feedback (results/human_rating/rater_A.csv and rater_B.csv, filled in
# by two team members): share of helpful / neutral / harmful per rater, how often the raters agree,
# and Cohen's kappa (agreement beyond chance: 0 = chance, 1 = perfect; above 0.6 is usually called
# substantial). Rows where the raters disagree are listed so the team can discuss them afterwards.
#
# Usage: python -m scripts.analyze_rating
import csv
import os
from collections import Counter

from sklearn.metrics import cohen_kappa_score

DIR = os.path.join("results", "human_rating")
RATINGS = ["helpful", "neutral", "harmful"]


def read(rater):
    """{row: rating} for one rater's sheet; ratings are lower-cased, blank rows left out."""
    with open(os.path.join(DIR, f"{rater}.csv"), encoding="utf-8-sig", newline="") as f:
        out = {}
        for r in csv.DictReader(f):
            rating = (r["rating"] or "").strip().lower()
            if rating:
                if rating not in RATINGS:
                    raise SystemExit(f"{rater} row {r['row']}: '{r['rating']}' is not one of {RATINGS}")
                out[r["row"]] = rating
        return out


def main():
    """Print the summary and save it to results/human_rating/summary.md."""
    a, b = read("rater_A"), read("rater_B")
    rows = sorted(set(a) & set(b), key=int)
    lines = [f"# Human rating of feedback ({len(rows)} messages rated by both raters)", ""]
    for name, ratings in [("Rater A", a), ("Rater B", b)]:
        c = Counter(ratings[r] for r in rows)
        lines.append(f"- {name}: " + ", ".join(f"{k} {c[k]} ({c[k] / len(rows):.0%})" for k in RATINGS))
    agree = sum(a[r] == b[r] for r in rows)
    kappa = cohen_kappa_score([a[r] for r in rows], [b[r] for r in rows], labels=RATINGS)
    lines += [f"- Same rating: {agree} of {len(rows)} ({agree / len(rows):.0%}); Cohen's kappa {kappa:.2f}",
              f"- Rated harmful by at least one rater: {sum('harmful' in (a[r], b[r]) for r in rows)}",
              "", "Rows where the raters disagree: " + ", ".join(f"{r} ({a[r]}/{b[r]})" for r in rows if a[r] != b[r])]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
