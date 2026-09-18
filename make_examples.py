# Download 8 example rows (2 per label) from the TRAIN split of SciEntsBank.
# These are for showing the model what each label looks like - they must
# never overlap with data/tasks.json, which comes from the test_ua split.
import json, random, requests

DATASET = "nkazi/SciEntsBank"
SPLIT = "train"                     # different split than data/tasks.json (test_ua)
BASE = "https://datasets-server.huggingface.co"
ALL_LABELS = ["correct", "contradictory", "partially_correct_incomplete",
              "irrelevant", "non_domain"]
USE_LABELS = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant"]
PER_LABEL = 2                       # 4 labels x 2 = 8 examples

rows = []
for offset in range(0, 300, 100):
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

print("\nHOW MANY OF EACH LABEL (in downloaded rows):")
for name in USE_LABELS:
    print(f"  {name}: {len(by_label[name])}")

random.seed(42)                     # same 8 examples every time we run this
examples = []
for name in USE_LABELS:
    for row in random.sample(by_label[name], PER_LABEL):
        examples.append({
            "task_id": row["id"],
            "question": row["question"],
            "reference_answer": row["reference_answer"],
            "student_answer": row["student_answer"],
            "gold_label": name,
        })

with open("data/examples.json", "w", encoding="utf-8") as f:
    json.dump(examples, f, indent=2)

print(f"\nSaved {len(examples)} examples to data/examples.json")

# Verify no task_id overlaps with data/tasks.json, so the test tasks stay unseen.
task_ids = {t["task_id"] for t in json.load(open("data/tasks.json", encoding="utf-8"))}
example_ids = {e["task_id"] for e in examples}
overlap = task_ids & example_ids
if overlap:
    print(f"\nWARNING: {len(overlap)} task_id(s) overlap with data/tasks.json: {overlap}")
else:
    print("\nVerified: no task_id overlap with data/tasks.json")
