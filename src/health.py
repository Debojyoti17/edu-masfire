# Two extra reliability checks for the redesigned pipeline:
#   - input check: flags student answers that look empty or cut off, so they go to a teacher
#     instead of being graded automatically
#   - canary check: before a grading session, each LLM grader grades 32 "canary" answers whose
#     teacher grades are known (8 per label). A grader is benched for the session if its canary
#     score is far below what it normally achieves (less than a 1% chance of being that low by bad
#     luck). This catches a grader that has gone wrong in a way message checks cannot see (for
#     example one that confidently gives wrong labels) without benching graders that are simply weak.
#   - reference check for new questions: a reference answer that shares almost no words with its
#     question was probably mixed up with another question's, so the answer goes to a teacher.
#   - word check for answers read from photos: if many words of the reading are not real words seen
#     before, the handwriting was probably misread, so the answer goes to a teacher.
import json
import math
import os
import random
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval import LABELS, load_train

CANARIES_PER_LABEL = 8
BENCH_IF_CHANCE_BELOW = 0.01
SENTENCE_ENDINGS = ".!?)\"'"
_known = {}
# Lowest word similarity between a question and its reference answer before the reference is
# suspected to belong to another question. Chosen on the train split so that 5% of genuine
# question/reference pairs are flagged (train: 94.9% of swapped references caught); on the test
# sets it flagged 5.5% / 4.5% of genuine pairs and caught 94.5% / 92.2% of swaps (test_uq / test_ud).
REFERENCE_SIMILARITY_MIN = 0.059
# Highest share of unknown words (not in the train answers, questions and reference answers, nor in
# this question and its reference answer) before a photo reading is suspected to be misread. Chosen
# so that 5% of genuine train answers are flagged, scoring each half of the train questions with a
# vocabulary built from the other half. On simulated misreadings (about 7% of characters wrong) it
# caught 75% / 72% / 78% and flagged 1.1% / 1.8% / 10.2% of genuine answers (test_ua / test_uq / test_ud).
UNKNOWN_WORDS_MAX = 0.167


def check_input(task):
    """Problems with the student's answer that make automatic grading unsafe (empty list = fine).
    Note: in this dataset every real answer ends with punctuation; real pupils' typing may not,
    so in a school this check should be tuned before relying on it."""
    answer = task["student_answer"].strip()
    if not answer:
        return ["student answer is empty"]
    if answer[-1] not in SENTENCE_ENDINGS:
        return ["student answer does not end like a sentence (possibly cut off)"]
    return []


def fit_reference_check(tasks):
    """Learn the words used in this session's questions and reference answers (common English words
    left out), for reference_looks_wrong."""
    return TfidfVectorizer(stop_words="english").fit([t["question"] + " " + t["reference_answer"] for t in tasks])


def reference_looks_wrong(task, vectorizer):
    """True if the reference answer shares almost no words with the question (similarity below
    REFERENCE_SIMILARITY_MIN), which suggests it belongs to a different question."""
    sim = cosine_similarity(vectorizer.transform([task["question"]]), vectorizer.transform([task["reference_answer"]]))
    return float(sim[0, 0]) < REFERENCE_SIMILARITY_MIN


def words(text):
    """The words of a text, lower-cased, letters only."""
    return re.findall(r"[a-z]+", text.lower())


def known_words():
    """Every word in the train split's answers, questions and reference answers (built once)."""
    if "known" not in _known:
        rows, _, _, _ = load_train()
        _known["known"] = {w for r in rows for w in words(r["student_answer"] + " " + r["question"] + " " + r["reference_answer"])}
    return _known["known"]


def looks_misread(task):
    """True if the share of unknown words in the student answer is above UNKNOWN_WORDS_MAX (used for
    answers read from photos, where unknown words usually mean the handwriting was misread)."""
    answer = words(task["student_answer"])
    context = set(words(task["question"] + " " + task["reference_answer"]))
    unknown = sum(w not in known_words() and w not in context for w in answer)
    return unknown / max(1, len(answer)) > UNKNOWN_WORDS_MAX


def canary_set():
    """32 train answers with known grades (8 per label), chosen once with seed 42 from train
    answers that are NOT in the judge's 2000-answer training sample."""
    sample_ids = {t["task_id"] for t in json.load(open(os.path.join("data", "eval_train_sample_2000.json"),
                                                           encoding="utf-8"))}
    rows, _, _, _ = load_train()
    rng = random.Random(42)
    canaries = []
    for label in LABELS:
        pool = [r for r in rows if r["label_name"] == label and r["id"] not in sample_ids]
        for r in rng.sample(pool, CANARIES_PER_LABEL):
            canaries.append({"task_id": r["id"], "question": r["question"], "reference_answer": r["reference_answer"],
                             "student_answer": r["student_answer"], "gold_label": label})
    return canaries


def expected_canary_accuracy(model, mode):
    """How often this grader/mode is normally right on a label-balanced set: its average per-label
    accuracy on the 2000-answer train sample (results/rag)."""
    path = os.path.join("results", "rag", f"train_sample_2000_{mode}_{model.replace(':', '-')}.jsonl")
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    recalls = []
    for label in LABELS:
        of_label = [r for r in recs if r["gold_label"] == label]
        recalls.append(sum(r["predicted_label"] == label for r in of_label) / len(of_label))
    return sum(recalls) / len(recalls)


def chance_of_score_this_low(right, n, p):
    """Probability that a grader who is normally right with probability p gets `right` or fewer
    of n canaries right just by bad luck (binomial distribution)."""
    return sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(right + 1))


def benched(canary_labels, gold_labels, expected):
    """True if a grader's canary score is so far below its normal accuracy that bad luck is an
    unlikely explanation (less than BENCH_IF_CHANCE_BELOW)."""
    right = sum(p == g for p, g in zip(canary_labels, gold_labels))
    return chance_of_score_this_low(right, len(gold_labels), expected) < BENCH_IF_CHANCE_BELOW
