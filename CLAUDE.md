\# Project: Fault-tolerant multi-agent grading (MAS-FIRE extension)



\## Goal

A 4-agent pipeline (Planner -> Tutor -> Grader -> Feedback) grades short student

answers using small local models. We inject faults on purpose, measure how often

wrong feedback reaches the student, and test a confidence check that catches it.



\## Data

data/tasks.json holds 48 tasks. Each has: task\_id, question, reference\_answer,

student\_answer, gold\_label.

gold\_label is one of: correct, contradictory, partially\_correct\_incomplete, irrelevant.

The Grader's job is to pick the right label for the student's answer.



\## Environment

\- Windows, RTX 4060 8GB, Python 3.13, virtualenv at .venv

\- Models run locally through Ollama: qwen2.5:7b (main), llama3.1:8b

\- Use the native `ollama` Python client. Do NOT use the OpenAI-compatible endpoint:

&#x20; it drops the logprobs we need.

\- Always pass options: temperature=0, seed=42, num\_ctx=4096



\## Rules

\- Keep code simple: plain Python functions, no classes unless truly needed.

\- Comment every function in plain language. I am still learning Python.

\- Do not install new libraries without telling me why.

\- Never delete files in data/ or results/.

\- After each task, tell me exactly which command to run to test it.

\- Explain what you changed in simple terms before I run anything.



\## Layout

data/ (tasks), src/ (main code), scripts/ (experiment runners), results/ (outputs)

