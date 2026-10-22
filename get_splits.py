# Download every split of SciEntsBank (train, test_ua, test_uq, test_ud) and save each one to
# data/raw/scientsbank_<split>.json, so later experiments can run without internet.
# Uses the same Hugging Face datasets-server HTTP API as make_tasks.py.
import json
import os
import time

import requests

DATASET = "nkazi/SciEntsBank"
BASE = "https://datasets-server.huggingface.co"
SPLITS = ["train", "test_ua", "test_uq", "test_ud"]
ALL_LABELS = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant", "non_domain"]


def get_page(split, offset):
    """Download one page of 100 rows. If the server says 'too many requests', wait and try
    again, waiting longer each time."""
    wait = 5
    for _ in range(8):
        r = requests.get(f"{BASE}/rows",
                         params={"dataset": DATASET, "config": "default", "split": split,
                                 "offset": offset, "length": 100},
                         timeout=60)
        if r.status_code != 429:
            r.raise_for_status()
            return r.json()
        print(f"  rate-limited at {split} offset {offset}, waiting {wait}s", flush=True)
        time.sleep(wait)
        wait *= 2
    r.raise_for_status()


def fetch_split(split):
    """Download all rows of one split, 100 at a time, and return them as a list."""
    rows = []
    offset = 0
    while True:
        time.sleep(0.5)
        batch = [item["row"] for item in get_page(split, offset)["rows"]]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
    return rows


def main():
    """Fetch each split, add the label name next to its number, and save it."""
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    for split in SPLITS:
        rows = fetch_split(split)
        for row in rows:
            row["label_name"] = ALL_LABELS[row["label"]]
        path = os.path.join("data", "raw", f"scientsbank_{split}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
        counts = {name: sum(1 for r in rows if r["label_name"] == name) for name in ALL_LABELS}
        print(f"{split}: {len(rows)} rows saved to {path}  {counts}")


if __name__ == "__main__":
    main()
