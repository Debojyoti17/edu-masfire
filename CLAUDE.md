\# Project: Fault-tolerant multi-agent grading (MAS-FIRE extension)



\## Goal

A 4-agent pipeline (Planner -> Tutor -> Grader -> Feedback) grades short student

answers using small local models. We inject faults on purpose, measure how often

wrong feedback reaches the student, and test a confidence check that catches it.



\## Data

data/tasks.json holds 120 tasks (30 per label). Each has: task\_id, question, reference\_answer,

student\_answer, gold\_label.

gold\_label is one of: correct, contradictory, partially\_correct\_incomplete, irrelevant.

The Grader's job is to pick the right label for the student's answer.



\## Environment

\- Windows, RTX 4060 8GB, Python 3.13, virtualenv at .venv

\- Models run locally through Ollama: qwen2.5:7b (main), llama3.1:8b

\- Use the native `ollama` Python client. Do NOT use the OpenAI-compatible endpoint:

&#x20; it drops the logprobs we need.

\- Always pass options: temperature=0, seed=42, num\_ctx=4096



\## Decisions

\- Main config: v6 prompt (two-step - step 1 describes the comparison with no label hints, step 2 makes the letter decision) running on llama3.1:8b. qwen2.5:7b stays as the second model for side-by-side comparison.

\- Why: of every prompt version x model combination tried, v6 on llama3.1:8b gave the widest gap between confidence when the grader is right vs when it is wrong: avg top probability 0.819 when correct vs 0.636 when wrong (gap 0.183), and only 3 of 29 wrong predictions were "confidently wrong" (top probability above 0.9). The same v6 prompt on qwen2.5:7b showed almost no gap (0.984 correct vs 0.962 wrong) with 26 of 30 wrong predictions confidently wrong, so qwen's confidence numbers are close to meaningless as a safety signal in this setup. Accuracy was similar either way (llama3.1:8b 19/48 = 0.396, qwen2.5:7b 18/48 = 0.375), so the choice was made on confidence calibration, not raw accuracy.

\- Caveat to keep in mind: llama3.1:8b often does not answer with a bare letter as its literal first token in the v6 decision step (it wants to start writing a sentence). In about a third of v6/llama3.1:8b cases, the reported confidence is renormalized from a small slice of its real output distribution, which adds noise to the confidence numbers above.



\## Rules

\- Keep code simple: plain Python functions, no classes unless truly needed.

\- Comment every function in plain language. I am still learning Python.

\- Do not install new libraries without telling me why.

\- Never delete files in data/ or results/.

\- After each task, tell me exactly which command to run to test it.

\- Explain what you changed in simple terms before I run anything.



\## Layout

data/ (tasks), src/ (main code), scripts/ (experiment runners), results/ (outputs)

