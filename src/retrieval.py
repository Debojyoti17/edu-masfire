# Retrieval Planner: finds teacher-graded student answers in the SciEntsBank train split to show
# the Grader as worked examples.
#   "same" mode:    graded answers to the same question, most similar to this student's answer
#   "similar" mode: most similar graded answers from *other* questions (for unseen questions)
# The task's own answer is never returned, so this can also be run on train answers
# (that is how the judge model gets its training data).
import json
import os
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TRAIN_PATH = os.path.join("data", "raw", "scientsbank_train.json")
LABELS = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant"]
K = 8

_cache = {}
_bank = []  # extra teacher-graded answers (see use_answer_bank)


def use_answer_bank(rows):
    """Add teacher-graded answers to the examples the Planner can retrieve, for example the first
    few answers a teacher graded to a new question. Rows look like train rows (id, question,
    reference_answer, student_answer, label_name). Pass [] to go back to the train split only."""
    global _bank
    _bank = list(rows)
    _cache.clear()


def load_train():
    """Load the teacher-graded train answers once (rare non_domain label dropped) plus any
    answer-bank rows, grouped by question, plus a word index over 'question + answer' for
    finding similar answers."""
    if "train" not in _cache:
        with open(TRAIN_PATH, encoding="utf-8") as f:
            rows = [r for r in json.load(f) if r["label_name"] in LABELS] + _bank
        by_question = defaultdict(list)
        for r in rows:
            by_question[r["question"]].append(r)
        texts = [r["question"] + " " + r["student_answer"] for r in rows]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit(texts)
        _cache["train"] = (rows, by_question, vectorizer, vectorizer.transform(texts))
    return _cache["train"]


def has_same_question_examples(task):
    """True if the train split has graded answers to this exact question (other than this one)."""
    _, by_question, _, _ = load_train()
    return any(r["id"] != task["task_id"] for r in by_question.get(task["question"], []))


def retrieve(task, mode, k=K):
    """Return up to k graded examples for this task in the given mode ('same' or 'similar')."""
    rows, by_question, vectorizer, matrix = load_train()
    if mode == "same":
        pool = [r for r in by_question.get(task["question"], []) if r["id"] != task["task_id"]]
        if not pool:
            return []
        local = TfidfVectorizer().fit([p["student_answer"] for p in pool] + [task["student_answer"]])
        sims = cosine_similarity(local.transform([task["student_answer"]]),
                                 local.transform([p["student_answer"] for p in pool]))[0]
        return [pool[i] for i in sims.argsort(kind="stable")[::-1][:k]]
    if mode == "similar":
        sims = cosine_similarity(vectorizer.transform([task["question"] + " " + task["student_answer"]]), matrix)[0]
        picked = []
        for i in sims.argsort(kind="stable")[::-1]:
            if rows[i]["question"] != task["question"]:
                picked.append(rows[i])
                if len(picked) == k:
                    break
        return picked
    raise ValueError(f"unknown retrieval mode: {mode}")
