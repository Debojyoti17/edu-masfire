# The four agents of the grading pipeline: Planner -> Tutor -> Grader -> Feedback.
# Each agent is one prompt sent to a local model. They pass plain text to each other.

from src.llm import DEFAULT_MODEL, choice_probabilities, generate
from src.rubric import RUBRIC

LETTERS = ["A", "B", "C", "D"]
LABELS = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant"]
LETTER_TO_LABEL = dict(zip(LETTERS, LABELS))

# How each label is written in feedback that a student reads.
LABEL_WORDS = {
    "correct": "correct",
    "contradictory": "contradictory",
    "partially_correct_incomplete": "partially correct",
    "irrelevant": "irrelevant",
}

PLANNER_MAX_TOKENS = 120
TUTOR_MAX_TOKENS = 160
FEEDBACK_MAX_TOKENS = 120


def task_block(task):
    """The question, reference answer and student answer, written the same way for every agent."""
    return (f"Question: {task['question']}\n"
            f"Reference answer: {task['reference_answer']}\n"
            f"Student answer: {task['student_answer']}\n")


def format_rubric():
    """The four labels with their rubric definitions, one per line, lettered A to D."""
    return "\n".join(f"{letter}) {label}: {RUBRIC[label]}" for letter, label in zip(LETTERS, LABELS))


def planner(task, model=DEFAULT_MODEL):
    """Planner: reads the question and reference answer (not the student's answer) and lists
    the key points a fully correct answer must contain."""
    prompt = f"""Question: {task['question']}
Reference answer: {task['reference_answer']}

List the 1 to 3 key points that a fully correct student answer must contain.
Write one short point per line, starting each line with "- ". Do not add anything else.
"""
    return generate(prompt, model=model, max_tokens=PLANNER_MAX_TOKENS).strip()


def tutor(task, plan, model=DEFAULT_MODEL):
    """Tutor: compares the student's answer with the Planner's key points and describes what
    is covered, missing or contradicted. It is told not to name a grade."""
    prompt = f"""{task_block(task)}
Key points a full answer needs:
{plan or "(none provided)"}

In 2-3 sentences, say which key points the student answer covers, which it misses,
and whether anything in it contradicts the reference answer.
Do not name or hint at a grading label. Only describe the comparison.
"""
    return generate(prompt, model=model, max_tokens=TUTOR_MAX_TOKENS).strip()


def grader(task, plan, notes, model=DEFAULT_MODEL, prefill="Label: "):
    """Grader: picks one of the four labels using the rubric plus the Planner's and Tutor's notes.
    Returns (label, probability of each label, captured mass). Label is None if the model
    gave no usable letter."""
    prompt = f"""{task_block(task)}
Key points a full answer needs:
{plan or "(none provided)"}

Tutor's comparison:
{notes or "(none provided)"}

Rubric:
{format_rubric()}

Based on all of the above, answer with one letter only: A, B, C, or D.
"""
    result = choice_probabilities(prompt, LETTERS, model=model, prefill=prefill)
    if result["probs"] is None:
        return None, None, result["mass"]
    probs = {LETTER_TO_LABEL[letter]: p for letter, p in result["probs"].items()}
    label = max(probs, key=probs.get)
    return label, probs, result["mass"]


def feedback(task, label, notes, model=DEFAULT_MODEL):
    """Feedback: writes the message the student actually sees. It must start with the verdict
    it was handed by the Grader, then give 2-3 sentences of advice."""
    prompt = f"""{task_block(task)}
The grader's verdict: {LABEL_WORDS.get(label, "unknown")}

Tutor's comparison:
{notes or "(none provided)"}

Write feedback to the student. Your first line must be "Verdict: " followed by exactly one of:
correct, partially correct, contradictory, irrelevant.
Then write 2-3 short, encouraging sentences explaining what was good and what to improve.
"""
    text = generate(prompt, model=model, max_tokens=FEEDBACK_MAX_TOKENS, prefill="Verdict: ")
    return "Verdict: " + text.strip()


def parse_verdict(feedback_text):
    """Reads the first line of the feedback and returns which label it tells the student,
    or None if the verdict is unclear."""
    lines = feedback_text.strip().splitlines()
    first = lines[0].lower().replace("verdict:", "") if lines else ""
    if "partial" in first:
        return "partially_correct_incomplete"
    if "contradict" in first:
        return "contradictory"
    if "irrelevant" in first:
        return "irrelevant"
    if "incorrect" in first or "not correct" in first:
        return None
    if "correct" in first:
        return "correct"
    return None
