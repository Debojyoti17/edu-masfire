# Build the larger evaluation sets for the paper from the raw splits in data/raw/:
#   data/eval_test_ua.json      every test_ua answer (unseen answers to seen questions)
#   data/eval_test_uq.json      every test_uq answer (unseen questions)
#   data/eval_test_ud_600.json  600 random test_ud answers, seed 42 (unseen domains)
# The rare non_domain label is dropped, as in make_tasks.py.
import json
import random

LABELS = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant"]


def to_tasks(split):
    """Read one raw split and convert each row to the task format the graders use."""
    with open(f"data/raw/scientsbank_{split}.json", encoding="utf-8") as f:
        rows = json.load(f)
    tasks = [{"task_id": r["id"], "question": r["question"], "reference_answer": r["reference_answer"],
              "student_answer": r["student_answer"], "gold_label": r["label_name"]}
             for r in rows if r["label_name"] in LABELS]
    assert len({t["task_id"] for t in tasks}) == len(tasks), f"duplicate task_id in {split}"
    return tasks


def save(tasks, path):
    """Write a task list and print how many of each label it has."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=1)
    counts = {label: sum(t["gold_label"] == label for t in tasks) for label in LABELS}
    print(f"{path}: {len(tasks)} tasks {counts}")


def main():
    """Build and save the three evaluation sets, plus a 2000-answer train sample used to train
    the judge model of the redesigned pipeline."""
    save(to_tasks("test_ua"), "data/eval_test_ua.json")
    save(to_tasks("test_uq"), "data/eval_test_uq.json")
    ud = to_tasks("test_ud")
    random.seed(42)
    save(random.sample(ud, 600), "data/eval_test_ud_600.json")
    save(random.Random(42).sample(to_tasks("train"), 2000), "data/eval_train_sample_2000.json")


if __name__ == "__main__":
    main()
