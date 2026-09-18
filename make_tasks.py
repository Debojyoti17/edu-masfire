# Download rows from SciEntsBank and save 48 balanced tasks to data/tasks.json
import json, random, requests

DATASET = "nkazi/SciEntsBank"
SPLIT = "test_ua"
BASE = "https://datasets-server.huggingface.co"
ALL_LABELS = ["correct", "contradictory", "partially_correct_incomplete",
              "irrelevant", "non_domain"]
# non_domain is dropped: too rare (only 3 examples) and not a real grading case
USE_LABELS = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant"]
PER_LABEL = 30                      # 4 labels x 30 = 120 tasks (fewer if a label runs short)

rows = []
for offset in range(0, 600, 100):
    r = requests.get(f"{BASE}/rows",
                     params={"dataset": DATASET, "config": "default",
                             "split": SPLIT, "offset": offset, "length": 100},
                     timeout=60).json()
    batch = [item["row"] for item in r["rows"]]
    if not batch:
        break
    rows.extend(batch)
    print(f"downloaded {len(rows)} rows")

by_label = {name: [] for name in ALL_LABELS}
for row in rows:
    by_label[ALL_LABELS[row["label"]]].append(row)

print("\nHOW MANY OF EACH LABEL:")
for name in ALL_LABELS:
    print(f"  {name}: {len(by_label[name])}")

random.seed(42)                     # same tasks every time we run this
tasks = []
actual_counts = {}
for name in USE_LABELS:
    available = by_label[name]
    take = min(PER_LABEL, len(available))   # take fewer if the label runs short
    actual_counts[name] = take
    for row in random.sample(available, take):
        tasks.append({
            "task_id": row["id"],
            "question": row["question"],
            "reference_answer": row["reference_answer"],
            "student_answer": row["student_answer"],
            "gold_label": name,
        })

random.shuffle(tasks)               # mix the labels so runs are not grouped
with open("data/tasks.json", "w", encoding="utf-8") as f:
    json.dump(tasks, f, indent=2)

print(f"\nSaved {len(tasks)} tasks to data/tasks.json")
print("ACTUAL COUNT PER LABEL (target was", PER_LABEL, "each):")
for name in USE_LABELS:
    short = "  <- fewer than target, label ran short" if actual_counts[name] < PER_LABEL else ""
    print(f"  {name}: {actual_counts[name]}{short}")
print("EXAMPLE TASK:")
print(json.dumps(tasks[0], indent=2))