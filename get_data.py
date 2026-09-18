# Fetch dataset info over plain HTTP - avoids the blocked library
import json
import requests

DATASET = "nkazi/SciEntsBank"
BASE = "https://datasets-server.huggingface.co"

# 1. What configs and splits does this dataset have?
splits = requests.get(f"{BASE}/splits", params={"dataset": DATASET}, timeout=60).json()
print("SPLITS AVAILABLE:")
print(json.dumps(splits, indent=2)[:1500])

# 2. Grab the first rows of the first split to see the columns
first = splits["splits"][0]
rows = requests.get(f"{BASE}/first-rows",
                    params={"dataset": DATASET,
                            "config": first["config"],
                            "split": first["split"]},
                    timeout=60).json()

print("\nCOLUMNS:")
for col in rows["features"]:
    print(" -", col["name"], ":", col["type"])

print("\nFIRST EXAMPLE:")
print(json.dumps(rows["rows"][0]["row"], indent=2)[:2000])