# Pieces of the redesigned pipeline that are not graders or the judge:
#   - message envelopes, so a receiving agent can tell a lost, misrouted or corrupted message
#   - a reference-answer check against the question bank (train split)
#   - the Tutor (explains what the answer covers / misses; used only for feedback)
#   - the Feedback agent with a verdict check (its verdict must match the judge's label)
#   - the TF-IDF classifier as a live grader, so corrupted inputs reach it too
import hashlib
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.agents import LABEL_WORDS, feedback, parse_verdict
from src.llm import DEFAULT_MODEL, generate
from src.retrieval import load_train
from scripts.run_tfidf_baseline import build_features, load_split

LABELS = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant"]
TUTOR_MAX_TOKENS = 160
FEEDBACK_MAX_TOKENS = 120
_cache = {}


def fingerprint(text):
    """A short fingerprint of some text: changes completely if the text changes at all."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def make_message(task, sender, payload):
    """Wrap an agent's output with the task ID, a fingerprint of the question and a checksum
    of the content, so the receiver can check it arrived intact and for the right task."""
    body = json.dumps(payload, sort_keys=True)
    return {"task_id": task["task_id"], "question_fp": fingerprint(task["question"]), "sender": sender,
            "payload": payload, "checksum": fingerprint(task["task_id"] + body)}


def check_message(message, task):
    """What the receiving agent finds: 'ok', 'lost', 'misrouted' or 'corrupted'."""
    if message is None:
        return "lost"
    if message["task_id"] != task["task_id"] or message["question_fp"] != fingerprint(task["question"]):
        return "misrouted"
    if message["checksum"] != fingerprint(message["task_id"] + json.dumps(message["payload"], sort_keys=True)):
        return "corrupted"
    return "ok"


def reference_bank():
    """Question -> the reference answer the train split gives for it."""
    if "bank" not in _cache:
        rows, _, _, _ = load_train()
        _cache["bank"] = {r["question"]: r["reference_answer"] for r in rows}
    return _cache["bank"]


def check_reference(task):
    """Compare the task's reference answer with the question bank. Returns ('ok' | 'mismatch' |
    'unknown question', the bank's reference answer or None)."""
    known = reference_bank().get(task["question"])
    if known is None:
        return "unknown question", None
    return ("ok" if known.strip() == task["reference_answer"].strip() else "mismatch"), known


def tutor(task, model=DEFAULT_MODEL):
    """Tutor: in 2-3 sentences, what the student answer covers, misses or gets wrong compared
    with the reference answer. It does not grade; its notes are only used to write feedback."""
    prompt = f"""Question: {task['question']}
Reference answer: {task['reference_answer']}
Student answer: {task['student_answer']}

In 2-3 sentences, say what the student answer gets right, what it is missing, and whether
anything in it contradicts the reference answer. Do not give a grade.
"""
    return generate(prompt, model=model, max_tokens=TUTOR_MAX_TOKENS).strip()


def feedback_with_check(task, label, notes, model=DEFAULT_MODEL, check=True):
    """Feedback agent. Writes the student's message; with check=True, if the verdict it wrote
    differs from the label it was given, the message is rewritten with the verdict fixed.
    Returns (feedback text, whether the check had to step in)."""
    text = feedback(task, label, notes, model=model)
    if not check or parse_verdict(text) == label:
        return text, False
    fixed = f"Verdict: {LABEL_WORDS.get(label, label)}\n\n"
    prompt = f"""Question: {task['question']}
Reference answer: {task['reference_answer']}
Student answer: {task['student_answer']}

The verdict for this answer is: {LABEL_WORDS.get(label, label)}.
Tutor's comparison:
{notes or "(none provided)"}

Write 2-3 short, encouraging sentences for the student that are consistent with that verdict,
explaining what was good and what to improve.
"""
    rest = generate(prompt, model=model, max_tokens=FEEDBACK_MAX_TOKENS, prefill=fixed)
    return fixed + rest.strip(), True


def tfidf_grader():
    """The trained TF-IDF classifier (full train split), fitted once and kept in memory."""
    if "tfidf" not in _cache:
        rows = load_split("train")
        vs, vr = TfidfVectorizer(ngram_range=(1, 2), min_df=2), TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        X = build_features(vs, vr, rows, fit=True)
        model = LogisticRegression(C=3.0, max_iter=3000).fit(X, [r["label_name"] for r in rows])
        _cache["tfidf"] = (model, vs, vr)
    return _cache["tfidf"]


def tfidf_probs(task):
    """TF-IDF classifier probabilities for one (possibly corrupted) task, in LABELS order."""
    model, vs, vr = tfidf_grader()
    row = {"question": task["question"], "reference_answer": task["reference_answer"],
           "student_answer": task["student_answer"]}
    p = model.predict_proba(build_features(vs, vr, [row]))[0]
    order = list(model.classes_)
    return [float(p[order.index(label)]) for label in LABELS]
