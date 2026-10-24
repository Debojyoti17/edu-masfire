# Graders for the redesigned pipeline. Each LLM grader sees the rubric plus worked examples
# chosen by the retrieval Planner (src/retrieval.py) and picks a label; its probabilities for
# all four labels are kept so a judge model can combine several graders later.
from src.agents import LETTER_TO_LABEL, LETTERS, format_rubric, grader_prefill
from src.llm import choice_probabilities

LABEL_TO_LETTER = {label: letter for letter, label in LETTER_TO_LABEL.items()}


def rag_prompt(task, examples, mode):
    """The grading prompt with retrieved examples. 'same' examples are other answers to this
    question; 'similar' examples come from other questions, so they include their question."""
    if mode == "same":
        intro = "Here are other student answers to this same question, already graded by a teacher:"
        shown = "\n".join(f"Student answer: {e['student_answer']}\nLabel: {LABEL_TO_LETTER[e['label_name']]}\n"
                          for e in examples)
    else:
        intro = "Here are graded examples from similar questions, showing how a teacher applies the labels:"
        shown = "\n".join(f"Question: {e['question']}\nReference answer: {e['reference_answer']}\n"
                          f"Student answer: {e['student_answer']}\nLabel: {LABEL_TO_LETTER[e['label_name']]}\n"
                          for e in examples)
    return f"""Question: {task['question']}
Reference answer: {task['reference_answer']}

Rubric:
{format_rubric()}

{intro}
{shown}
Now grade this student answer the same way.
Question: {task['question']}
Student answer: {task['student_answer']}

Answer with one letter only: A, B, C, or D.
"""


def grade_with_prompt(prompt, model, cpu_only=False):
    """Send a grading prompt and return (label, probability of each label, captured mass).
    Label and probabilities are None if the model gave no usable letter.
    cpu_only=True runs the model without the graphics card (for laptop speed tests)."""
    result = choice_probabilities(prompt, LETTERS, model=model, prefill=grader_prefill(model), cpu_only=cpu_only)
    if result["probs"] is None:
        return None, None, result["mass"]
    probs = {LETTER_TO_LABEL[letter]: p for letter, p in result["probs"].items()}
    return max(probs, key=probs.get), probs, result["mass"]
