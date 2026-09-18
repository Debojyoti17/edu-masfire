# Compare grader results across prompt versions and models, in one table:
# accuracy, average top probability when correct vs wrong, and how many
# confidently wrong cases. Finds every results/grader_v*.json file, so
# results from any model that's been run show up automatically.
#
# Usage: python -m scripts.compare_grader_versions
import glob
import json
import os
import re

from scripts.analyze_grader import CONFIDENT_THRESHOLD, is_correct, top_prob
from src.llm import DEFAULT_MODEL

FILENAME_PATTERN = re.compile(r"^grader_(v\d+)(?:_.+)?\.json$")


def find_result_files():
    """Every results/grader_v*.json file, as (version, path) pairs."""
    files = []
    for path in sorted(glob.glob("results/grader_v*.json")):
        match = FILENAME_PATTERN.match(os.path.basename(path))
        if match:
            files.append((match.group(1), path))
    return files


def summarize(version, path):
    """Load one results file and compute its row of the table."""
    results = json.load(open(path, encoding="utf-8"))
    # Older results files were saved before we started recording which
    # model produced them - those were always run on the default model.
    model = results[0].get("model", DEFAULT_MODEL) if results else DEFAULT_MODEL

    correct = [r for r in results if is_correct(r)]
    wrong = [r for r in results if not is_correct(r)]

    accuracy = len(correct) / len(results)
    avg_p_correct = sum(top_prob(r) for r in correct) / len(correct) if correct else float("nan")
    avg_p_wrong = sum(top_prob(r) for r in wrong) / len(wrong) if wrong else float("nan")
    confidently_wrong = sum(1 for r in wrong if top_prob(r) > CONFIDENT_THRESHOLD)

    return version, model, accuracy, avg_p_correct, avg_p_wrong, confidently_wrong


def main():
    rows = [summarize(version, path) for version, path in find_result_files()]

    header = f'{"version":8}{"model":16}{"accuracy":10}{"avg p (correct)":18}{"avg p (wrong)":16}{"confidently wrong"}'
    print(header)
    for version, model, accuracy, avg_p_correct, avg_p_wrong, confidently_wrong in rows:
        print(f"{version:8}{model:16}{accuracy:<10.3f}{avg_p_correct:<18.3f}{avg_p_wrong:<16.3f}{confidently_wrong}")


if __name__ == "__main__":
    main()
