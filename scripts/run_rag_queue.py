# Run every grader job the redesigned pipeline needs, one after another, logging to
# results/rag/queue_log.txt. Each job resumes where it stopped, so the queue is safe to restart.
# Headline test numbers come first, then the train-sample runs the judge model learns from.
#
# Usage: python -m scripts.run_rag_queue
import os
import subprocess
import sys
import time

PY = sys.executable
LOG = os.path.join("results", "rag", "queue_log.txt")
MODELS = ["qwen2.5:7b", "mistral:7b", "gemma2:9b"]
TRAIN = "data/eval_train_sample_2000.json"
UA, UQ, UD = "data/eval_test_ua.json", "data/eval_test_uq.json", "data/eval_test_ud_600.json"


def log(message):
    """Print a message and append it, with the time, to the queue log."""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_jobs():
    """The ordered (model, task file, mode) jobs. Grouped by model where possible so the GPU
    does not keep swapping models."""
    jobs = [(m, UA, "same") for m in MODELS]
    for m in MODELS:
        jobs += [(m, TRAIN, "same"), (m, TRAIN, "similar"), (m, TRAIN, "zeroshot"),
                 (m, UQ, "similar"), (m, UQ, "zeroshot"), (m, UD, "similar"), (m, UD, "zeroshot"),
                 (m, UA, "zeroshot")]
    return jobs


def main():
    """Run every job in order, logging start, finish and failures."""
    os.makedirs(os.path.join("results", "rag", "logs"), exist_ok=True)
    jobs = build_jobs()
    log(f"rag queue start: {len(jobs)} jobs")
    for k, (model, tasks, mode) in enumerate(jobs, start=1):
        name = f"{model} {os.path.basename(tasks)} {mode}"
        log(f"[{k}/{len(jobs)}] start: {name}")
        start = time.time()
        logfile = os.path.join("results", "rag", "logs", f"{k:02d}_{model.replace(':', '-')}_{os.path.basename(tasks)}_{mode}.txt")
        with open(logfile, "w", encoding="utf-8") as out:
            code = subprocess.call([PY, "-u", "-m", "scripts.run_rag_grader", model, tasks, mode],
                                   stdout=out, stderr=subprocess.STDOUT)
        with open(logfile, encoding="utf-8") as out:
            last = [l.strip() for l in out if l.startswith("accuracy")]
        status = "done" if code == 0 else f"FAILED (exit {code})"
        log(f"[{k}/{len(jobs)}] {status}: {name} in {(time.time() - start) / 60:.1f} min  {last[-1] if last else ''}")
    log("rag queue finished")


if __name__ == "__main__":
    main()
