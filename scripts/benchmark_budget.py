# Budget-laptop tier (idea 8): how well does the grading system work with smaller, cheaper parts?
# Compares judges built from different sets of graders, all trained the same way (robust judge,
# train sample, seen / new-question regimes):
#   full        3 x 7-9B LLMs (qwen2.5:7b, mistral:7b, gemma2:9b) + TF-IDF       needs an 8 GB GPU
#   budget      3 x 2-3B LLMs (qwen2.5:3b, gemma2:2b, llama3.2:3b) + TF-IDF      ~4 GB GPU, or CPU
#   +finetuned  the same, plus a fine-tuned DeBERTa grader (if trained)
#   no LLM      TF-IDF + fine-tuned DeBERTa only                                  any laptop CPU
# For each: accuracy on test_ua (questions with graded examples) and on test_uq / test_ud_600 (new
# questions), the share graded automatically at the escalation threshold, and the accuracy of those.
# Also lists every single grader's accuracy and its average seconds per answer on the GPU.
#
# --cpu-timing additionally measures seconds per answer with no graphics card at all
# (Ollama option num_gpu 0) on 10 answers per LLM, and for the fine-tuned models on 50 answers.
#
# Usage: python -m scripts.benchmark_budget [--cpu-timing]
import json
import os
import sys
import time

import numpy as np

import src.judge as judge
from src.graders import grade_with_prompt, rag_prompt
from src.retrieval import retrieve
from scripts.run_pipeline import load_jsonl

FULL = ["qwen2.5:7b", "mistral:7b", "gemma2:9b"]
BUDGET = ["qwen2.5:3b", "gemma2:2b", "llama3.2:3b"]
FINETUNED = ["deberta-v3-xsmall", "deberta-v3-base-nli"]
TESTS = [("test_ua", "seen"), ("test_uq", "unseen"), ("test_ud_600", "unseen")]


def slug(model):
    """File-name form of a model name."""
    return model.replace(":", "-")


def single_graders():
    """Accuracy and average GPU seconds per answer of every LLM grader run that exists."""
    out = []
    for model in FULL + BUDGET:
        for set_name, regime in TESTS:
            for mode in judge.REGIME_MODES[regime]:
                path = os.path.join("results", "rag", f"{set_name}_{mode}_{slug(model)}.jsonl")
                recs = load_jsonl(path)
                if not recs:
                    continue
                out.append({"model": model, "set": set_name, "mode": mode, "n": len(recs),
                            "accuracy": round(sum(r["predicted_label"] == r["gold_label"] for r in recs) / len(recs), 3),
                            "seconds_per_answer": round(float(np.mean([r["seconds"] for r in recs])), 3)})
    for name in FINETUNED:
        for set_name, _ in TESTS:
            path = os.path.join("results", "finetuned", name, f"{set_name}.json")
            if os.path.exists(path):
                saved = json.load(open(path, encoding="utf-8"))
                tasks = json.load(open(os.path.join("data", f"eval_{set_name}.json"), encoding="utf-8"))
                right = [judge.LABELS[int(np.argmax(saved[t["task_id"]]))] == t["gold_label"] for t in tasks]
                out.append({"model": name, "set": set_name, "mode": "fine-tuned", "n": len(tasks),
                            "accuracy": round(float(np.mean(right)), 3)})
    return out


def evaluate_judge(name, models, extras):
    """Train the robust judges with these graders and score them on the three test sets.
    The judge module reads its grader list from judge.MODELS, so it is swapped for this run."""
    judge.MODELS = models
    rows, trained = [], {}
    try:
        for set_name, regime in TESTS:
            if regime not in trained:
                model, cv_acc, c = judge.train_judge(regime, True, extras)
                trained[regime] = (model, cv_acc, judge.choose_threshold(regime, c, True, extras))
            model, cv_acc, threshold = trained[regime]
            tasks = json.load(open(os.path.join("data", f"eval_{set_name}.json"), encoding="utf-8"))
            X, _ = judge.features(tasks, set_name, regime, extras)
            labels, probs = judge.judge_predict(model, X)
            right = np.array(labels) == np.array([t["gold_label"] for t in tasks])
            auto = probs.max(axis=1) >= threshold
            rows.append({"judge": name, "set": set_name, "cv_accuracy": round(float(cv_acc), 3),
                         "accuracy": round(float(right.mean()), 3), "auto_share": round(float(auto.mean()), 3),
                         "auto_accuracy": round(float(right[auto].mean()), 3) if auto.any() else None})
            print(rows[-1], flush=True)
    finally:
        judge.MODELS = FULL
    return rows


def cpu_timing():
    """Seconds per answer with no graphics card: 10 test_ua answers per LLM (same-question
    examples, the usual school setting), 50 per fine-tuned model."""
    tasks = json.load(open(os.path.join("data", "eval_test_ua.json"), encoding="utf-8"))[:50]
    out = []
    for model in BUDGET + ["qwen2.5:7b"]:
        grade_with_prompt(rag_prompt(tasks[0], retrieve(tasks[0], "same"), "same"), model, cpu_only=True)  # load
        start = time.time()
        for t in tasks[:10]:
            grade_with_prompt(rag_prompt(t, retrieve(t, "same"), "same"), model, cpu_only=True)
        out.append({"model": model, "cpu_seconds_per_answer": round((time.time() - start) / 10, 2)})
        print(out[-1], flush=True)
    from src.finetuned import finetuned_probs
    for name in FINETUNED:
        if not os.path.exists(os.path.join("models", "finetuned", name)):
            continue
        finetuned_probs(tasks[0], name)  # load
        start = time.time()
        for t in tasks[1:]:
            finetuned_probs(t, name)
        out.append({"model": name, "cpu_seconds_per_answer": round((time.time() - start) / 49, 3)})
        print(out[-1], flush=True)
    return out


def main():
    """Run the comparison and save results/budget/benchmark.json."""
    os.makedirs(os.path.join("results", "budget"), exist_ok=True)
    trained = [n for n in FINETUNED if os.path.exists(os.path.join("results", "finetuned", n, "test_uq.json"))]
    result = {"single_graders": single_graders(), "judges": []}
    setups = [("full", FULL, []), ("budget", BUDGET, [])]
    for name in trained:
        setups += [(f"full + {name}", FULL, [name]), (f"budget + {name}", BUDGET, [name]),
                   (f"no LLM: TF-IDF + {name}", [], [name])]
    for name, models, extras in setups:
        result["judges"] += evaluate_judge(name, models, extras)
    if "--cpu-timing" in sys.argv:
        result["cpu_timing"] = cpu_timing()
    with open(os.path.join("results", "budget", "benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print("Saved results/budget/benchmark.json")


if __name__ == "__main__":
    main()
