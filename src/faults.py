# Faults we inject on purpose into the pipeline, one at a time, to see how often a broken
# step leads to wrong feedback reaching the student.

FAULTS = {
    "none": "No fault (baseline)",
    "planner_omit": "Planner's message is lost: Tutor and Grader get an empty plan",
    "planner_wrong_task": "Planner's message is misrouted: it carries the plan for a different question",
    "tutor_wrong_task": "Tutor's message is misrouted: it carries the comparison for a different student answer",
    "tutor_overclaim": "Tutor hallucinates: it claims the answer covers every key point with no errors",
    "grader_label_flip": "Grader's label is corrupted on its way to the Feedback agent",
    "reference_swap": "Data corruption: the reference answer belongs to a different question",
    "student_truncate": "Input noise: the student answer is cut to its first half",
}

OVERCLAIM_TEXT = ("The student answer covers every key point of the reference answer "
                  "and contains no errors or contradictions.")

LABEL_ORDER = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant"]


def donor_index(tasks, i):
    """For faults that misroute content, pick the task to borrow it from: the next task in the
    list that has a different question (so the borrowed content is genuinely wrong)."""
    n = len(tasks)
    j = (i + 1) % n
    while tasks[j]["question"] == tasks[i]["question"] and j != i:
        j = (j + 1) % n
    return j


def corrupt_task(task, fault, donor):
    """Return the version of the task the agents will see. Only the two input faults change it."""
    seen = dict(task)
    if fault == "reference_swap":
        seen["reference_answer"] = donor["reference_answer"]
    elif fault == "student_truncate":
        words = task["student_answer"].split()
        seen["student_answer"] = " ".join(words[: max(1, len(words) // 2)])
    return seen


def flip_label(label):
    """Corrupt a label into a different one (the next label in a fixed order)."""
    if label not in LABEL_ORDER:
        return LABEL_ORDER[0]
    return LABEL_ORDER[(LABEL_ORDER.index(label) + 1) % len(LABEL_ORDER)]
