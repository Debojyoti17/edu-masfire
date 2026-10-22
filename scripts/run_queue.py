# Run the planned experiments one after another and log each step to results/queue_log.txt.
# If a job fails, it is logged and the queue moves on. Rerunning the queue skips finished
# grader jobs, and pipeline jobs resume where they stopped, so it is safe to restart.
#
# Usage: python -m scripts.run_queue [--workers N]
#   --workers N is passed to pipeline runs (only use N > 1 with an Ollama server started
#   with OLLAMA_NUM_PARALLEL >= N).
import os
import subprocess
import sys
import time

PY = sys.executable
LOG = os.path.join("results", "queue_log.txt")
MODELS = ["qwen2.5:7b", "llama3.1:8b", "mistral:7b", "gemma2:9b"]
NEW_MODELS = ["mistral:7b", "gemma2:9b"]
EVAL_SETS = ["data/eval_test_ua.json", "data/eval_test_uq.json", "data/eval_test_ud_600.json"]


def log(message):
    """Print a message and append it, with the time, to the queue log."""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def grader_output(model, tasks_path=None):
    """The results file run_grader.py writes for v2 + prefill with this model and task file."""
    parts = ["grader_v2_prefill"]
    if model != "qwen2.5:7b":
        parts.append(model.replace(":", "-"))
    if tasks_path:
        parts.append(os.path.splitext(os.path.basename(tasks_path))[0].replace("eval_", ""))
    return os.path.join("results", "_".join(parts) + ".json")


def build_jobs(workers):
    """The full ordered job list: presentation-critical runs first, then the paper runs."""
    jobs = []
    # Phase 1: the two new models on the 120-task set (single Grader, then pipeline + faults).
    for m in NEW_MODELS:
        jobs.append((f"grader v2 120 {m}", [PY, "-m", "scripts.run_grader", "v2", m, "--prefill"],
                     grader_output(m)))
    for m in NEW_MODELS:
        jobs.append((f"pipeline 120 {m}", [PY, "-m", "scripts.run_pipeline", "all", m, "--workers", str(workers)],
                     None))
    # Phase 2: single Grader on the larger evaluation sets, every model.
    for m in MODELS:
        for t in EVAL_SETS:
            jobs.append((f"grader v2 {os.path.basename(t)} {m}",
                         [PY, "-m", "scripts.run_grader", "v2", m, "--prefill", "--tasks", t], grader_output(m, t)))
    # Phase 3: pipeline + faults on the full test_ua set, every model, then test_uq for the main model.
    for m in MODELS:
        jobs.append((f"pipeline test_ua {m}", [PY, "-m", "scripts.run_pipeline", "all", m, "--tasks",
                                               "data/eval_test_ua.json", "--workers", str(workers)], None))
    jobs.append(("pipeline test_uq qwen2.5:7b", [PY, "-m", "scripts.run_pipeline", "all", "qwen2.5:7b", "--tasks",
                                                 "data/eval_test_uq.json", "--workers", str(workers)], None))
    return jobs


def main():
    """Run every job in order, skipping grader jobs whose results already exist."""
    workers = 1
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    jobs = build_jobs(workers)
    log(f"queue start: {len(jobs)} jobs, workers={workers}")
    for k, (name, cmd, output) in enumerate(jobs, start=1):
        if output and os.path.exists(output):
            log(f"[{k}/{len(jobs)}] skip (already done): {name}")
            continue
        log(f"[{k}/{len(jobs)}] start: {name}")
        start = time.time()
        safe_name = name.replace(" ", "_").replace(":", "-").replace(".json", "")
        with open(os.path.join("results", "queue_logs", f"{k:02d}_{safe_name}.txt"), "w", encoding="utf-8") as out:
            code = subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT)
        status = "done" if code == 0 else f"FAILED (exit {code})"
        log(f"[{k}/{len(jobs)}] {status}: {name} in {(time.time() - start) / 60:.1f} min")
    log("queue finished")


if __name__ == "__main__":
    os.makedirs(os.path.join("results", "queue_logs"), exist_ok=True)
    main()
