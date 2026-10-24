# Fine-tuned grader: a compact pre-trained text model (DeBERTa-v3) trained on the SciEntsBank
# train split to read "question + reference answer" next to a student answer and output one of
# the four labels. It is fast enough to run on a laptop CPU, and it becomes another grader that
# the judge combines with the LLMs and the TF-IDF classifier.
#
# For the judge's training data, each train answer is predicted by a version trained on the other
# 4/5 of the data (5 folds), once with folds mixing questions ("seen") and once with folds split by
# question ("unseen", to simulate new questions). A final model trained on all train answers
# predicts the test sets and is saved to models/finetuned/<name>/ for live use in the pipeline.
#
# Usage: python -m scripts.train_finetuned_grader deberta-v3-base-nli
import json
import os
import random
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, StratifiedKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from scripts.run_tfidf_baseline import load_split

LABELS = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant"]
SETTINGS = {
    "deberta-v3-base-nli": {"lr": 2e-5, "batch": 16, "epochs": 3},
    "deberta-v3-base": {"lr": 2e-5, "batch": 16, "epochs": 3},
    "deberta-v3-xsmall": {"lr": 5e-5, "batch": 32, "epochs": 4},
}
MAX_LEN = 256
TEST_SETS = {"120": "data/tasks.json", "test_ua": "data/eval_test_ua.json",
             "test_uq": "data/eval_test_uq.json", "test_ud_600": "data/eval_test_ud_600.json"}


def pair(row):
    """The two pieces of text the model reads: question + reference answer, and the student answer."""
    return f"Question: {row['question']} Reference answer: {row['reference_answer']}", row["student_answer"]


def batches(rows, size, shuffle, rng=None):
    """Split rows into batches (shuffled for training)."""
    order = list(range(len(rows)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), size):
        yield [rows[j] for j in order[i:i + size]]


def encode(tokenizer, rows, device):
    """Turn a batch of rows into model inputs, padded to the longest in the batch."""
    firsts, seconds = zip(*(pair(r) for r in rows))
    enc = tokenizer(list(firsts), list(seconds), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def train_model(name, rows, seed=42):
    """Fine-tune one model on the given train rows and return (model, tokenizer)."""
    cfg = SETTINGS[name]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    path = os.path.join("models", name)
    tokenizer = AutoTokenizer.from_pretrained(path)
    # dtype=float32: transformers 5 otherwise keeps the checkpoint's float16 weights, and training
    # float16 weights directly loses the small updates (the model then only learns the majority label).
    model = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=len(LABELS), ignore_mismatched_sizes=True, dtype=torch.float32,
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
    device = "cuda"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    steps = cfg["epochs"] * ((len(rows) + cfg["batch"] - 1) // cfg["batch"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    rng = random.Random(seed)
    model.train()
    for _ in range(cfg["epochs"]):
        for batch in batches(rows, cfg["batch"], True, rng):
            inputs = encode(tokenizer, batch, device)
            labels = torch.tensor([LABELS.index(r["label_name"]) for r in batch], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(**inputs).logits.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, rows, batch=64):
    """Probability of each label (in LABELS order) for every row."""
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for part in batches(rows, batch, False):
            inputs = encode(tokenizer, part, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**inputs).logits.float()
            out += torch.softmax(logits, dim=-1).cpu().tolist()
    return out


def accuracy(probs, rows, key="label_name"):
    """Share of rows where the most likely label is the right one."""
    return sum(LABELS[int(np.argmax(p))] == r[key] for p, r in zip(probs, rows)) / len(rows)


def main():
    """Train the out-of-fold models for the judge, then the final model; save every prediction."""
    name = sys.argv[1] if len(sys.argv) > 1 else "deberta-v3-base-nli"
    out_dir = os.path.join("results", "finetuned", name)
    os.makedirs(out_dir, exist_ok=True)
    train = load_split("train")
    y = [r["label_name"] for r in train]

    for regime in ["seen", "unseen"]:
        path = os.path.join(out_dir, f"train_oof_{regime}.json")
        if os.path.exists(path):
            continue
        if regime == "seen":
            folds = StratifiedKFold(5, shuffle=True, random_state=42).split(train, y)
        else:
            folds = GroupKFold(5).split(train, y, groups=[r["question"] for r in train])
        oof = {}
        for k, (tr, va) in enumerate(folds, start=1):
            start = time.time()
            model, tok = train_model(name, [train[i] for i in tr])
            probs = predict(model, tok, [train[i] for i in va])
            for i, p in zip(va, probs):
                oof[train[i]["id"]] = p
            acc = accuracy(probs, [train[i] for i in va])
            print(f"{name} {regime} fold {k}/5: held-out accuracy {acc:.3f} ({time.time() - start:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(oof, f)

    start = time.time()
    model, tok = train_model(name, train)
    save_to = os.path.join("models", "finetuned", name)
    model.save_pretrained(save_to)
    tok.save_pretrained(save_to)
    print(f"{name}: final model trained on all {len(train)} answers in {time.time() - start:.0f}s, saved to {save_to}", flush=True)
    for set_name, task_path in TEST_SETS.items():
        tasks = json.load(open(task_path, encoding="utf-8"))
        probs = predict(model, tok, tasks)
        with open(os.path.join(out_dir, f"{set_name}.json"), "w", encoding="utf-8") as f:
            json.dump({t["task_id"]: p for t, p in zip(tasks, probs)}, f)
        print(f"  {set_name}: accuracy {accuracy(probs, tasks, 'gold_label'):.3f} on {len(tasks)} answers", flush=True)


if __name__ == "__main__":
    main()
