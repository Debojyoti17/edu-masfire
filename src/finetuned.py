# Live use of the fine-tuned grader (trained by scripts/train_finetuned_grader.py) inside the
# pipeline. It runs on the CPU, so it does not compete with Ollama for the GPU and works on a
# laptop without a GPU. Each distinct answer text is graded once and remembered.
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant"]
MAX_LEN = 256
_models = {}
_memo = {}


def load(name):
    """Load a saved fine-tuned grader once (CPU)."""
    if name not in _models:
        path = os.path.join("models", "finetuned", name)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.eval()
        _models[name] = (model, tokenizer)
    return _models[name]


def finetuned_probs(task, name):
    """Probability of each label (LABELS order) for one task, from the named fine-tuned grader."""
    key = (name, task["question"], task["reference_answer"], task["student_answer"])
    if key not in _memo:
        model, tokenizer = load(name)
        first = f"Question: {task['question']} Reference answer: {task['reference_answer']}"
        enc = tokenizer(first, task["student_answer"], truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()
        order = [model.config.id2label[i] for i in range(len(probs))]
        _memo[key] = [probs[order.index(label)] for label in LABELS]
    return _memo[key]
