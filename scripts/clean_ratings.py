# Turn the raters' free-text answers into rating words, following the team's instructions (29 Oct):
#   rater_A (Sammo) rated whether he understood the feedback: "understand" (any spelling) means helpful,
#     "don't understand" / "not understand" means neutral or harmful, recorded as "not helpful";
#     anything else (e.g. "yes", "exactly") is left out.
#   rater_B (Prottoy): misspellings are corrected ("hamrful" -> harmful) and extra words ignored
#     ("helpful maybe" -> helpful); entries with no rating word ("no idea") are left out.
# The raters' own words are kept in the original_rating column.
# Reads results/human_rating/rater_A.csv and rater_B.csv; writes rater_A_clean.csv and rater_B_clean.csv.
# Usage: python -m scripts.clean_ratings
import csv
import difflib
import os

DIR = os.path.join("results", "human_rating")
WORDS = ["helpful", "neutral", "harmful"]


def clean_a(text):
    """Sammo's 'understand' answers: helpful, not helpful, or '' if it is neither."""
    t = text.lower().replace("’", "'")
    proper = clean_b(t)  # a proper rating word (added later) is kept as it is
    if proper and "underst" not in t:
        return "not helpful" if "not" in t.split() else proper
    if "not" in t or "don't" in t or "dont" in t:
        return "not helpful"
    if difflib.get_close_matches("understand", t.split(), n=1, cutoff=0.75):
        return "helpful"
    return ""


def clean_b(text):
    """Prottoy's answers: the first word that is (a misspelling of) helpful/neutral/harmful, or ''."""
    for word in text.lower().split():
        match = difflib.get_close_matches(word, WORDS, n=1, cutoff=0.75)
        if match:
            return match[0]
    return ""


def main():
    """Clean both sheets and print every change so it can be checked."""
    for rater, clean in [("rater_A", clean_a), ("rater_B", clean_b)]:
        with open(os.path.join(DIR, f"{rater}.csv"), encoding="utf-8-sig", newline="") as f:
            rows = [{k: v for k, v in r.items() if k} for r in csv.DictReader(f)]
        for r in rows:
            r["original_rating"] = (r["rating"] or "").strip()
            r["rating"] = clean(r["original_rating"]) if r["original_rating"] else ""
            if r["original_rating"] and r["original_rating"].lower() != r["rating"]:
                print(f"{rater} row {r['row']}: '{r['original_rating']}' -> '{r['rating'] or '(left out)'}'")
        with open(os.path.join(DIR, f"{rater}_clean.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{rater}: {sum(bool(r['rating']) for r in rows)} of {len(rows)} rows rated after cleaning\n")


if __name__ == "__main__":
    main()
