# Spreadsheet workflow for a teacher (idea 7): grade a whole class from one CSV file on a laptop.
#
# Input CSV (one row per student answer; Excel can save this format), columns:
#   student, question, reference_answer, student_answer, teacher_label
# teacher_label is optional. Fill it for answers the teacher has already graded (for example the
# first few answers to a new question): those grades are kept as they are AND become worked
# examples for grading the rest ("answer bank"). Accepted labels: correct, partially correct,
# contradictory, irrelevant.
#
# What happens:
#   1. canary check: each LLM grader grades 32 answers with known grades; a grader that scores
#      far below its normal level is left out for this session
#   2. input check: empty or cut-off answers go straight to the teacher; for a question the system
#      does not know, a reference answer that shares almost no words with its question is reported
#   3. three LLM graders + TF-IDF (+ the fine-tuned grader if the judge uses it) -> judge
#   4. grades the judge is not confident about go to the teacher; the rest get short feedback
# Output, next to the input file:
#   <name>_graded.csv  every answer with grade, confidence, status (auto / teacher / teacher-graded)
#                      and feedback text
#   <name>_review.csv  only the answers the teacher must check, with the two most likely grades
#                      and an empty teacher_label column.
#                      Fill it in, copy those grades into the input file's teacher_label column and
#                      run again: they join the answer bank and improve the next grades.
#
# Usage:
#   python -m scripts.grade_csv examples/class_quiz.csv
#   python -m scripts.grade_csv examples/class_quiz.csv --no-feedback     grades only (faster)
import csv
import os
import pickle
import sys
import time

from src.agents import LABEL_WORDS
from src.graders import grade_with_prompt, rag_prompt
from src.health import (benched, canary_set, check_input, expected_canary_accuracy, fit_reference_check,
                        reference_looks_wrong)
from src.judge import EXTRA_GRADERS, LABELS, MODELS, REGIME_MODES, choose_threshold, judge_predict, train_judge
from src.finetuned import finetuned_probs
from src.pipeline_v2 import FEEDBACK_VERSION, check_reference, tfidf_probs, tutor, write_feedback
from src.retrieval import has_same_question_examples, retrieve, use_answer_bank
from scripts.run_grader import build_prompt
from scripts.run_pipeline_v2 import CACHE_DIR, FEEDBACK_MODEL, key_of, load_cache, save_to_cache

JUDGE_DIR = os.path.join("models", "judge")
WORDS_TO_LABEL = {"correct": "correct", "partially correct": "partially_correct_incomplete",
                  "partially_correct_incomplete": "partially_correct_incomplete",
                  "contradictory": "contradictory", "irrelevant": "irrelevant"}


def read_rows(path):
    """Read the class CSV; every row gets an id and a cleaned teacher label (or None)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows, start=1):
        r["task_id"] = f"row{i}"
        given = (r.get("teacher_label") or "").strip().lower()
        if given and given not in WORDS_TO_LABEL:
            sys.exit(f"Row {i}: unknown teacher_label '{r['teacher_label']}'. Use: correct, partially correct, "
                     "contradictory or irrelevant.")
        r["label"] = WORDS_TO_LABEL.get(given)
    return rows


def load_judges():
    """The two judges (seen / new questions) and their escalation thresholds. Trained once on the
    train sample and then saved in models/judge/, so later sessions start in seconds."""
    name = "_".join(["robust"] + EXTRA_GRADERS) + ".pkl"
    path = os.path.join(JUDGE_DIR, name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    print("Training the judges (first run only, a few minutes) ...", flush=True)
    judges, threshold = {}, {}
    for regime in ["seen", "unseen"]:
        judges[regime], _, c = train_judge(regime)
        threshold[regime] = choose_threshold(regime, c)
    os.makedirs(JUDGE_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((judges, threshold), f)
    return judges, threshold


def grade(prompt, model, cache):
    """One LLM grader answer, from the cache if this exact prompt was graded before."""
    key = key_of(model, prompt)
    if key not in cache:
        save_to_cache("graders", key, list(grade_with_prompt(prompt, model)), cache)
    return cache[key]


def run_canaries(cache):
    """Canary check: the (model, mode) pairs to leave out this session."""
    canaries = canary_set()
    out = set()
    for model in MODELS:
        for mode in ["same", "similar", "zeroshot"]:
            labels = []
            for c in canaries:
                prompt = build_prompt(c, "v2") if mode == "zeroshot" else rag_prompt(c, retrieve(c, mode), mode)
                labels.append(grade(prompt, model, cache)[0])
            if benched(labels, [c["gold_label"] for c in canaries], expected_canary_accuracy(model, mode)):
                out.add((model, mode))
    return out


def write_csv(path, rows, columns):
    """Write rows (dicts) to a CSV file with the given columns."""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    """Grade every row of the class CSV and write the graded and review files."""
    args = sys.argv[1:]
    if not args:
        sys.exit("Usage: python -m scripts.grade_csv <class.csv> [--no-feedback]")
    path, with_feedback = args[0], "--no-feedback" not in args
    start = time.time()
    rows = read_rows(path)
    bank = [{"id": r["task_id"], "question": r["question"], "reference_answer": r["reference_answer"],
             "student_answer": r["student_answer"], "label_name": r["label"]} for r in rows if r["label"]]
    use_answer_bank(bank)
    todo = [r for r in rows if not r["label"]]
    print(f"{len(rows)} answers: {len(bank)} already graded by the teacher (answer bank), {len(todo)} to grade", flush=True)

    judges, threshold = load_judges()
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = load_cache("graders")
    print("Canary check ...", flush=True)
    benched_pairs = run_canaries(cache)
    print(f"  left out this session: {sorted(benched_pairs) or 'none'}", flush=True)

    # Graders, one model at a time so the GPU loads each model once.
    plans = {}
    for r in todo:
        r["regime"] = "seen" if has_same_question_examples(r) else "unseen"
        plans[r["task_id"]] = [(m, mode, build_prompt(r, "v2") if mode == "zeroshot" else
                                rag_prompt(r, retrieve(r, mode), mode))
                               for m in MODELS for mode in REGIME_MODES[r["regime"]]]
    for model in MODELS:
        print(f"Grading with {model} ...", flush=True)
        for r in todo:
            for m, mode, prompt in plans[r["task_id"]]:
                if m == model:
                    grade(prompt, model, cache)

    # Judge, checks and feedback.
    text_cache = load_cache("text")
    reference_check = fit_reference_check(rows)
    for n, r in enumerate(todo, start=1):
        columns = []
        for model, mode, prompt in plans[r["task_id"]]:
            probs = None if (model, mode) in benched_pairs else grade(prompt, model, cache)[1]
            columns += [probs[l] for l in LABELS] if probs else [0.25] * 4
        columns += tfidf_probs(r)
        for name in EXTRA_GRADERS:
            columns += finetuned_probs(r, name)
        labels, probs = judge_predict(judges[r["regime"]], [columns])
        r["label"], r["confidence"] = labels[0], round(float(max(probs[0])), 3)
        top = sorted(zip(LABELS, probs[0]), key=lambda lp: -lp[1])[:2]
        r["top_two"] = " / ".join(f"{LABEL_WORDS[l]} {p:.2f}" for l, p in top)
        problems = check_input(r)
        if check_reference(r)[0] == "unknown question" and reference_looks_wrong(r, reference_check):
            problems.append("reference answer shares almost no words with the question - please check it")
        if problems:
            r["status"], r["reason"] = "teacher", problems[0]
        elif r["confidence"] < threshold[r["regime"]]:
            r["status"], r["reason"] = "teacher", "system not confident enough"
        else:
            r["status"], r["reason"] = "auto", ""
        if with_feedback and r["status"] == "auto":
            key = key_of("tutor", FEEDBACK_MODEL, r["question"], r["reference_answer"], r["student_answer"])
            if key not in text_cache:
                save_to_cache("text", key, tutor(r, FEEDBACK_MODEL), text_cache)
            fb_key = key_of(FEEDBACK_VERSION, FEEDBACK_MODEL, r["question"], r["reference_answer"],
                            r["student_answer"], r["label"], text_cache[key])
            if fb_key not in text_cache:
                save_to_cache("text", fb_key, list(write_feedback(r, r["label"], text_cache[key], FEEDBACK_MODEL)),
                              text_cache)
            r["feedback"] = text_cache[fb_key][0]
        if n % 25 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)} graded", flush=True)
    use_answer_bank([])

    for r in rows:
        if "status" not in r:
            r["status"], r["confidence"] = "teacher-graded", ""
        r["grade"] = LABEL_WORDS[r["label"]]
    base = os.path.splitext(path)[0]
    out_columns = ["student", "question", "student_answer", "grade", "confidence", "status", "reason", "feedback"]
    write_csv(base + "_graded.csv", rows, out_columns)
    review = [dict(r, teacher_label="") for r in rows if r["status"] == "teacher"]
    write_csv(base + "_review.csv", review, ["student", "question", "reference_answer", "student_answer",
                                             "grade", "confidence", "top_two", "reason", "teacher_label"])
    auto = sum(r["status"] == "auto" for r in rows)
    print(f"\nDone in {time.time() - start:.0f}s: {auto} graded automatically, {len(review)} for the teacher to check.")
    print(f"  {base}_graded.csv\n  {base}_review.csv")


if __name__ == "__main__":
    main()
