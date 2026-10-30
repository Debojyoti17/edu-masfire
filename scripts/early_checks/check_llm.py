# Test: use src/llm.py to grade the first task in data/tasks.json.
import json
from src.llm import choice_probabilities

tasks = json.load(open("data/tasks.json", encoding="utf-8"))
task = tasks[0]

# The four labels, in the same order as the letters we ask the model to pick.
labels = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant"]

prompt = f"""Question: {task['question']}
Reference answer: {task['reference_answer']}
Student answer: {task['student_answer']}

Is the student answer correct, contradictory, partially_correct_incomplete, or irrelevant?
Answer with one letter only.
A) correct
B) contradictory
C) partially_correct_incomplete
D) irrelevant
"""

result = choice_probabilities(prompt, ["A", "B", "C", "D"])

print("TASK:", task["task_id"])
print("GOLD LABEL:", task["gold_label"])
print("RAW MASS:", result["mass"])
print("PROBABILITIES:")
if result["probs"] is None:
    print("  none of the options showed up in the top candidate tokens")
else:
    for letter, label in zip(["A", "B", "C", "D"], labels):
        print(f"  {letter} ({label}): {result['probs'][letter]:.3f}")
