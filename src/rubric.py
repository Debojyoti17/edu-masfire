# A written grading rubric for the four labels the Grader must choose between.
# This is plain text, meant to be dropped into prompts so the model grades
# the same way a teacher following this rubric would.

RUBRIC = {
    "correct": (
        "The student's answer captures the key idea in the reference answer. "
        "It does not need to use the same words, and it can leave out minor "
        "detail, but it must not be missing any part of the core explanation "
        "and must not contain anything that is scientifically wrong."
    ),
    "contradictory": (
        "The student directly addresses the question, but states something "
        "that conflicts with the reference answer - a wrong fact, a reversed "
        "relationship, or a wrong cause. The key test: the student took a "
        "clear position on the question and that position is incorrect."
    ),
    "partially_correct_incomplete": (
        "The student's answer is on the right track and contains no incorrect "
        "statements, but it is missing a necessary part of the reference "
        "answer - for example the right conclusion without the reasoning "
        "behind it, or only one of two required points. Nothing the student "
        "wrote is wrong; something required is simply missing."
    ),
    "irrelevant": (
        "The student's answer does not address what the question is actually "
        "asking. This includes restating the question, describing an "
        "unrelated observation, or answering a different question. The key "
        "test: you cannot judge this answer as right or wrong on the "
        "question's own terms, because it never engages with the question."
    ),
}

# The distinction that matters most: "contradictory" means the student
# engaged with the question and got it wrong. "irrelevant" means the student
# never engaged with the question at all, so there is nothing to judge as
# right or wrong. If the student made a claim that answers the question but
# that claim is false, that is contradictory, not irrelevant.
