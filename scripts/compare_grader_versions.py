# Compare grader results across prompt versions (results/grader_v1.json,
# grader_v2.json, grader_v3.json, ...) in one table: accuracy, average top
# probability when correct vs wrong, and how many confidently wrong cases.
#
# Usage: python -m scripts.compare_grader_versions
import json

from scripts.analyze_grader import CONFIDENT_THRESHOLD, is_correct, top_prob
from scripts.run_grader import VERSIONS


def summarize(version):
    """Load one version's results and compute its row of the table."""
    results = json.load(open(f"results/grader_{version}.json", encoding="utf-8"))
    correct = [r for r in results if is_correct(r)]
    wrong = [r for r in results if not is_correct(r)]

    accuracy = len(correct) / len(results)
    avg_p_correct = sum(top_prob(r) for r in correct) / len(correct) if correct else float("nan")
    avg_p_wrong = sum(top_prob(r) for r in wrong) / len(wrong) if wrong else float("nan")
    confidently_wrong = sum(1 for r in wrong if top_prob(r) > CONFIDENT_THRESHOLD)

    return version, accuracy, avg_p_correct, avg_p_wrong, confidently_wrong


def main():
    rows = [summarize(version) for version in VERSIONS]

    print(f'{"version":8}{"accuracy":10}{"avg p (correct)":18}{"avg p (wrong)":16}{"confidently wrong"}')
    for version, accuracy, avg_p_correct, avg_p_wrong, confidently_wrong in rows:
        print(f"{version:8}{accuracy:<10.3f}{avg_p_correct:<18.3f}{avg_p_wrong:<16.3f}{confidently_wrong}")


if __name__ == "__main__":
    main()
