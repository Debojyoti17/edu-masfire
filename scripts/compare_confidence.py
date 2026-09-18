# Compare confidence-estimation methods (results/confidence_{method}.json,
# from scripts/run_confidence_method.py): for each, report the gap between
# confidence when the grader is right vs wrong, AUROC for predicting
# correctness from confidence, and how many wrong answers were confidently
# wrong (confidence above 0.9).
#
# Usage: python -m scripts.compare_confidence
import json

from scripts.run_confidence_method import METHODS

CONFIDENT_THRESHOLD = 0.9


def auroc(scored):
    """AUROC via the rank-sum (Mann-Whitney U) method: how often a random
    correct prediction has higher confidence than a random wrong one, with
    ties counted as a half-win. scored is a list of (confidence, is_correct).
    Returns None if there's only one class present (AUROC is undefined)."""
    ranked = sorted(scored, key=lambda pair: pair[0])
    n = len(ranked)

    # Average-rank tied confidence values together (standard Mann-Whitney tie handling).
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        average_rank = (i + j) / 2 + 1  # ranks are 1-indexed
        for k in range(i, j + 1):
            ranks[k] = average_rank
        i = j + 1

    n_correct = sum(1 for _, is_correct in ranked if is_correct)
    n_wrong = n - n_correct
    if n_correct == 0 or n_wrong == 0:
        return None

    rank_sum_correct = sum(rank for rank, (_, is_correct) in zip(ranks, ranked) if is_correct)
    u = rank_sum_correct - n_correct * (n_correct + 1) / 2
    return u / (n_correct * n_wrong)


def summarize(path):
    records = json.load(open(path, encoding="utf-8"))
    usable = [r for r in records if r["confidence"] is not None]
    missing = len(records) - len(usable)

    correct = [r for r in usable if r["predicted_label"] == r["gold_label"]]
    wrong = [r for r in usable if r["predicted_label"] != r["gold_label"]]

    accuracy = sum(1 for r in records if r["predicted_label"] == r["gold_label"]) / len(records)
    avg_conf_correct = sum(r["confidence"] for r in correct) / len(correct) if correct else float("nan")
    avg_conf_wrong = sum(r["confidence"] for r in wrong) / len(wrong) if wrong else float("nan")
    gap = avg_conf_correct - avg_conf_wrong

    scored = [(r["confidence"], r["predicted_label"] == r["gold_label"]) for r in usable]
    auc = auroc(scored)

    confidently_wrong = sum(1 for r in wrong if r["confidence"] > CONFIDENT_THRESHOLD)

    return {
        "accuracy": accuracy,
        "gap": gap,
        "auroc": auc,
        "confidently_wrong": confidently_wrong,
        "n_wrong": len(wrong),
        "missing": missing,
    }


def main():
    print(f'{"method":18}{"accuracy":10}{"gap":10}{"auroc":10}{"confidently wrong":20}{"missing"}')
    for method in METHODS:
        path = f"results/confidence_{method}.json"
        stats = summarize(path)
        auc_str = f"{stats['auroc']:.3f}" if stats["auroc"] is not None else "n/a"
        print(f"{method:18}{stats['accuracy']:<10.3f}{stats['gap']:<10.3f}{auc_str:<10}"
              f"{stats['confidently_wrong']}/{stats['n_wrong']:<17}{stats['missing']}")


if __name__ == "__main__":
    main()
