# Run the redesigned pipeline on a task file, fault-free and with each injected fault, with the
# fault-tolerance layer switched off and on:
#   Planner (retrieval) -> 3 LLM graders + TF-IDF -> judge -> Tutor -> Feedback (verdict fixed to the label)
# Fault-tolerance layer (ft=on): message checks with re-request, reference-answer check against
# the question bank, input check, canary check that benches a misbehaving grader, and escalation
# of low-confidence or flagged answers to a teacher.
#
# Work is done stage by stage (all qwen grading, then mistral, then gemma2, then judge, then
# Tutor + Feedback) so the GPU never swaps models per answer. Every model answer is cached by
# its exact prompt, so repeated conditions cost nothing. Results resume if stopped.
#
# Usage:
#   python -m scripts.run_pipeline_v2 data/tasks.json                    all faults, ft off and on
#   python -m scripts.run_pipeline_v2 data/tasks.json --faults none,plan_lost --ft on --limit 5
#   --judge plain uses a judge trained without fault-aware copies (saved in <set>/, for comparison);
#   the default robust judge saves in <set>_robust/.
#   --verdict-only skips writing the advice text (the verdict is fixed to the label anyway).
#   --checks basic leaves out the input check, canary check and new-question reference check (saved
#   without the _full_refsim suffix; folders named _full are from before the new-question reference check).
#   Extra (fine-tuned) graders used by the judge (src/judge.py EXTRA_GRADERS) are added to the folder name.
import hashlib
import json
import os
import sys
import time

from src.agents import parse_verdict
from src.faults import OVERCLAIM_TEXT, corrupt_task, donor_index, flip_label
from src.graders import grade_with_prompt, rag_prompt
from src.health import (benched, canary_set, check_input, expected_canary_accuracy, fit_reference_check,
                        looks_misread, reference_looks_wrong)
from src.ocr import simulate_misreading
from src.judge import EXTRA_GRADERS, LABELS, MODELS, REGIME_MODES, judge_predict, train_judge, choose_threshold
from src.finetuned import finetuned_probs
from src.pipeline_v2 import (FEEDBACK_VERSION, check_message, check_reference, make_message, write_feedback,
                             tfidf_probs, tutor)
from src.retrieval import has_same_question_examples, retrieve
from scripts.run_grader import build_prompt
from scripts.run_pipeline import load_jsonl

FEEDBACK_MODEL = "qwen2.5:7b"
FAULTS_V2 = {
    "none": "No fault",
    "plan_lost": "Planner's examples never arrive",
    "plan_misrouted": "Planner sends the examples for a different question",
    "grader_crash": "One grader (qwen) returns nothing",
    "grader_lying": "One grader (qwen) confidently reports a wrong label",
    "label_corrupted": "The judge's label is corrupted on its way to Feedback",
    "reference_swap": "Data corruption: the reference answer belongs to a different question",
    "student_truncate": "Input noise: the student answer is cut to its first half",
    "tutor_overclaim": "Tutor claims the answer covers every key point with no errors",
    "handwriting_misread": "The answer was read from a photo of handwriting, with reading mistakes",
}
CACHE_DIR = os.path.join("results", "pipeline_v2", "cache")
_reference_checks = {}  # one fitted reference check per task list


# ---------------------------------------------------------------- caching of model answers
def key_of(*parts):
    """A cache key: fingerprint of everything that determines a model's (deterministic) answer."""
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()



def load_cache(name):
    """Load a cache file of {key: value} lines."""
    return {r["key"]: r["value"] for r in load_jsonl(os.path.join(CACHE_DIR, name + ".jsonl"))}


def save_to_cache(name, key, value, cache):
    """Store one cached model answer in memory and on disk."""
    cache[key] = value
    with open(os.path.join(CACHE_DIR, name + ".jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}) + "\n")


def seed_grader_cache(model, set_name, tasks, cache):
    """Put the grader runs already done by the rag queue into the cache, so the fault-free
    pipeline reuses them instead of asking the model again."""
    slug = model.replace(":", "-")
    for mode in ["same", "similar", "zeroshot"]:
        runs = {r["task_id"]: r for r in load_jsonl(os.path.join("results", "rag", f"{set_name}_{mode}_{slug}.jsonl"))}
        for t in tasks:
            r = runs.get(t["task_id"])
            if r is None:
                continue
            prompt = build_prompt(t, "v2") if mode == "zeroshot" else rag_prompt(t, retrieve(t, mode), mode)
            cache.setdefault(key_of(model, prompt), [r["predicted_label"], r["probabilities"], r["mass"]])


# ---------------------------------------------------------------- one task, stage by stage
def plan_stage(task, fault, ft, donor, notes):
    """Planner + its message to the graders. Returns {mode: examples} for the modes this task's
    regime needs, after any fault and (with ft on) the message check and re-request."""
    regime = "seen" if has_same_question_examples(task) else "unseen"
    plans = {}
    for mode in REGIME_MODES[regime]:
        if mode == "zeroshot":
            plans[mode] = []
            continue
        correct = make_message(task, "planner", {"ids": [e["id"] for e in retrieve(task, mode)]})
        if fault == "plan_lost":
            message = None
        elif fault == "plan_misrouted":
            message = make_message(donor, "planner", {"ids": [e["id"] for e in retrieve(donor, mode)]})
        else:
            message = correct
        status = check_message(message, task) if ft == "on" else "unchecked"
        if status in ("lost", "misrouted", "corrupted"):
            notes.append(f"planner message {status} -> re-requested")
            message = correct
        examples = retrieve(task, mode) if message is correct else (
            [] if message is None else retrieve(donor, mode))
        plans[mode] = examples
    return regime, plans


def grader_prompts(task, regime, plans):
    """Every (model, mode, prompt) this task needs from the LLM graders."""
    out = []
    for model in MODELS:
        for mode in REGIME_MODES[regime]:
            prompt = build_prompt(task, "v2") if mode == "zeroshot" else rag_prompt(task, plans[mode], mode)
            out.append((model, mode, prompt))
    return out


def lie(label):
    """What the 'lying grader' fault sends: a confident wrong label (the next label in a fixed order)."""
    wrong = flip_label(label)
    return wrong, {l: (0.97 if l == wrong else 0.01) for l in LABELS}


def finish_task(task, original, fault, ft, regime, prompts, grader_cache, judges, threshold, text_cache, notes,
                verdict_only=False, benched_graders=frozenset(), input_flagged=False, explain=False):
    """Judge, Tutor and Feedback for one task, using cached grader answers. Returns the record.
    verdict_only=True skips writing the advice text: the Feedback verdict line is fixed to the label
    it is handed (0 mismatches in over 25,000 generated cases), so the delivered verdict is that label.
    benched_graders: (model, mode) pairs the canary check benched; their votes are left out.
    input_flagged: the input check found a problem, so the answer goes to a teacher.
    explain=True also returns what every grader told the judge and the escalation threshold (for the demo)."""
    start = time.time()
    columns, grader_labels, inputs = [], {}, {}
    for model, mode, prompt in prompts:
        label, probs, mass = grader_cache[key_of(model, prompt)]
        message = make_message(task, f"grader:{model}:{mode}", {"label": label, "probs": probs})
        if model == "qwen2.5:7b" and fault == "grader_crash":
            message = None
        if model == "qwen2.5:7b" and fault == "grader_lying" and label is not None:
            wrong, wrong_probs = lie(label)
            message = make_message(task, f"grader:{model}:{mode}", {"label": wrong, "probs": wrong_probs})
        status = check_message(message, task) if ft == "on" else "unchecked"
        if status in ("lost", "misrouted", "corrupted"):
            notes.append(f"grader {model}/{mode} message {status} -> re-requested")
            message = make_message(task, f"grader:{model}:{mode}", {"label": label, "probs": probs})
        payload = message["payload"] if message else {"label": None, "probs": None}
        if (model, mode) in benched_graders:
            notes.append(f"grader {model}/{mode} benched by canary check")
            payload = {"label": None, "probs": None}
        p = payload["probs"]
        columns += [p[l] for l in LABELS] if p else [0.25] * 4
        grader_labels[f"{model}|{mode}"] = payload["label"]
        inputs[f"{model} ({mode} examples)"] = [p[l] for l in LABELS] if p else None
    inputs["TF-IDF classifier"] = tfidf_probs(task)
    columns += inputs["TF-IDF classifier"]
    for name in EXTRA_GRADERS:
        inputs[f"{name} (fine-tuned)"] = finetuned_probs(task, name)
        columns += inputs[f"{name} (fine-tuned)"]
    labels, probs = judge_predict(judges[regime], [columns])
    final, confidence = labels[0], float(max(probs[0]))

    label_message = make_message(task, "judge", {"label": final, "confidence": confidence})
    if fault == "label_corrupted":
        label_message = dict(label_message, payload={"label": flip_label(final), "confidence": confidence})
    status = check_message(label_message, task) if ft == "on" else "unchecked"
    if status in ("lost", "misrouted", "corrupted"):
        notes.append(f"judge label message {status} -> re-sent")
        label_message = make_message(task, "judge", {"label": final, "confidence": confidence})
    sent = label_message["payload"]["label"]

    details = {"inputs": inputs, "threshold": threshold[regime]} if explain else {}
    if verdict_only:
        escalated = ft == "on" and (confidence < threshold[regime] or input_flagged)
        return details | {"task_id": original["task_id"], "gold_label": original["gold_label"], "fault": fault, "ft": ft,
                "regime": regime, "grader_labels": grader_labels, "final_label": final, "confidence": confidence,
                "label_sent": sent, "feedback": None, "feedback_generated": False, "delivered": sent,
                "escalated": escalated, "detections": list(notes), "seconds": round(time.time() - start, 2)}

    tutor_key = key_of("tutor", FEEDBACK_MODEL, task["question"], task["reference_answer"], task["student_answer"])
    if tutor_key not in text_cache:
        save_to_cache("text", tutor_key, tutor(task, FEEDBACK_MODEL), text_cache)
    tutor_notes = OVERCLAIM_TEXT if fault == "tutor_overclaim" else text_cache[tutor_key]

    fb_key = key_of(FEEDBACK_VERSION, FEEDBACK_MODEL, task["question"], task["reference_answer"],
                    task["student_answer"], str(sent), tutor_notes)
    if fb_key not in text_cache:
        save_to_cache("text", fb_key, list(write_feedback(task, sent, tutor_notes, FEEDBACK_MODEL)), text_cache)
    feedback_text, verdict_matches = text_cache[fb_key]
    if not verdict_matches:
        notes.append("feedback verdict could not be read back as the label it was given")
    delivered = parse_verdict(feedback_text)
    escalated = ft == "on" and (confidence < threshold[regime] or input_flagged)
    return details | {"task_id": original["task_id"], "gold_label": original["gold_label"], "fault": fault, "ft": ft,
            "regime": regime, "grader_labels": grader_labels, "final_label": final, "confidence": confidence,
            "label_sent": sent, "feedback": feedback_text, "delivered": delivered, "escalated": escalated,
            "detections": list(notes), "seconds": round(time.time() - start, 2)}


def prepare(task, tasks, i, fault, ft, full_checks=False):
    """Apply input faults, the reference check and (full checks) the input check and the reference
    check for new questions, then the Planner stage. Returns (task the agents see, regime, grader
    prompts, notes, input flagged)."""
    notes = []
    donor = tasks[donor_index(tasks, i)]
    seen_task = corrupt_task(task, fault, donor) if fault in ("reference_swap", "student_truncate") else dict(task)
    if fault == "handwriting_misread":
        seen_task["student_answer"] = simulate_misreading(task["student_answer"], task["task_id"])
    if ft == "on":
        status, bank = check_reference(seen_task)
        if status == "mismatch":
            notes.append("reference answer did not match the question bank -> restored")
            seen_task["reference_answer"] = bank
    input_flagged = False
    if ft == "on" and full_checks:
        for problem in check_input(seen_task):
            notes.append(f"input check: {problem} -> sent to a teacher")
            input_flagged = True
        # The system knows which answers it read from photos (here: all answers in this condition),
        # and checks those readings for misread words.
        if fault == "handwriting_misread" and looks_misread(seen_task):
            notes.append("input check: many unknown words in the photo reading (possibly misread) -> sent to a teacher")
            input_flagged = True
        if status == "unknown question":
            if id(tasks) not in _reference_checks:
                _reference_checks[id(tasks)] = fit_reference_check(tasks)
            if reference_looks_wrong(seen_task, _reference_checks[id(tasks)]):
                notes.append("reference answer does not seem to match the question -> sent to a teacher")
                input_flagged = True
    regime, plans = plan_stage(seen_task, fault, ft, donor, notes)
    return seen_task, regime, grader_prompts(seen_task, regime, plans), notes, input_flagged


# ---------------------------------------------------------------- driver
def main():
    """Read the options, prepare every condition, fill the grader cache model by model, then
    finish every task and save results/pipeline_v2/<set>/<fault>_<ft>.jsonl."""
    args = sys.argv[1:]
    if not args:
        sys.exit("Usage: python -m scripts.run_pipeline_v2 <task file> [--faults a,b] [--ft on|off|both] "
                 "[--judge robust|plain] [--limit N]")

    def option(name, default):
        """Value after a command-line flag, or the default."""
        return args[args.index(name) + 1] if name in args else default

    tasks_path = args[0]
    faults = list(FAULTS_V2) if option("--faults", "all") == "all" else option("--faults", "").split(",")
    fts = ["off", "on"] if option("--ft", "both") == "both" else [option("--ft", "both")]
    tasks = json.load(open(tasks_path, encoding="utf-8"))
    limit = option("--limit", None)
    if limit:
        tasks = tasks[:int(limit)]
    set_name = os.path.splitext(os.path.basename(tasks_path))[0].replace("eval_", "")
    if tasks_path == "data/tasks.json":
        set_name = "120"
    robust = option("--judge", "robust") == "robust"
    verdict_only = "--verdict-only" in args
    full_checks = option("--checks", "full") == "full"
    suffix = (("_robust" if robust else "") + ("_full_refsim" if full_checks else "")
              + ("_" + "_".join(EXTRA_GRADERS) if EXTRA_GRADERS else "") + (f"_limit{limit}" if limit else ""))
    out_dir = os.path.join("results", "pipeline_v2", set_name + suffix)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"Training {'robust' if robust else 'plain'} judges on the train sample ...", flush=True)
    judges, threshold = {}, {}
    for regime in ["seen", "unseen"]:
        judges[regime], cv_acc, c = train_judge(regime, robust)
        threshold[regime] = choose_threshold(regime, c, robust)
        print(f"  {regime}: cross-validated accuracy {cv_acc:.3f} (C={c}), escalation threshold {threshold[regime]:.3f}", flush=True)

    grader_cache = {}
    for model in MODELS:
        seed_grader_cache(model, "test_ua" if set_name in ("120",) else set_name, tasks, grader_cache)
    grader_cache.update(load_cache("graders"))
    text_cache = load_cache("text")

    # 1. Prepare every condition and collect the grader prompts it needs.
    jobs = []
    for fault in faults:
        for ft in fts:
            path = os.path.join(out_dir, f"{fault}_{ft}.jsonl")
            done = {r["task_id"] for r in load_jsonl(path)}
            for i, task in enumerate(tasks):
                if task["task_id"] in done:
                    continue
                seen_task, regime, prompts, notes, flagged = prepare(task, tasks, i, fault, ft, full_checks)
                jobs.append((fault, ft, path, task, seen_task, regime, prompts, notes, flagged))
    print(f"{len(jobs)} task-conditions to run", flush=True)

    # Canary answers (known grades) that each grader/mode grades before the session (full checks).
    canary_prompts = []
    if full_checks and "on" in fts:
        canaries = canary_set()
        for model in MODELS:
            for mode in ["same", "similar", "zeroshot"]:
                for c in canaries:
                    prompt = build_prompt(c, "v2") if mode == "zeroshot" else rag_prompt(c, retrieve(c, mode), mode)
                    canary_prompts.append((model, mode, c["gold_label"], prompt))

    # 2. Fill the grader cache, one model at a time.
    for model in MODELS:
        needed = {}
        for m, mode, _, prompt in canary_prompts:
            if m == model and key_of(m, prompt) not in grader_cache:
                needed[key_of(m, prompt)] = prompt
        for job in jobs:
            for m, mode, prompt in job[6]:
                if m == model and key_of(m, prompt) not in grader_cache:
                    needed[key_of(m, prompt)] = prompt
        print(f"{model}: {len(needed)} new grader calls", flush=True)
        start = time.time()
        for k, (key, prompt) in enumerate(needed.items(), start=1):
            save_to_cache("graders", key, list(grade_with_prompt(prompt, model)), grader_cache)
            if k % 200 == 0 or k == len(needed):
                print(f"  {model}: {k}/{len(needed)}, about {(len(needed) - k) * (time.time() - start) / k / 60:.0f} min left", flush=True)

    # Canary check per fault: which grader/modes would be benched this session.
    benched_by_fault = {}
    for fault in faults:
        results = {}
        for model, mode, gold, prompt in canary_prompts:
            label = grader_cache[key_of(model, prompt)][0]
            if model == "qwen2.5:7b" and fault == "grader_lying" and label is not None:
                label = lie(label)[0]
            results.setdefault((model, mode), ([], []))
            results[(model, mode)][0].append(label)
            results[(model, mode)][1].append(gold)
        benched_by_fault[fault] = frozenset(k for k, (labels, golds) in results.items()
                                            if benched(labels, golds, expected_canary_accuracy(*k)))
        if canary_prompts:
            print(f"  canary check, {fault}: benched {sorted(benched_by_fault[fault]) or 'none'}", flush=True)

    # 3. Judge, Tutor and Feedback for every task-condition.
    start = time.time()
    for k, (fault, ft, path, task, seen_task, regime, prompts, notes, flagged) in enumerate(jobs, start=1):
        rec = finish_task(seen_task, task, fault, ft, regime, prompts, grader_cache, judges, threshold, text_cache, notes,
                          verdict_only, benched_by_fault[fault] if ft == "on" else frozenset(), flagged)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if k % 50 == 0 or k == len(jobs):
            print(f"  finished {k}/{len(jobs)}, about {(len(jobs) - k) * (time.time() - start) / k / 60:.0f} min left", flush=True)
    print("All requested conditions finished.", flush=True)


if __name__ == "__main__":
    main()
