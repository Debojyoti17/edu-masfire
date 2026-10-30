\# Project: Fault-tolerant multi-agent grading (MAS-FIRE extension)



\## Goal

A 4-agent pipeline (Planner -> Tutor -> Grader -> Feedback) grades short student

answers using small local models. We inject faults on purpose, measure how often

wrong feedback reaches the student, and test a confidence check that catches it.

(Since 23 Oct 2026 the system is the redesigned pipeline - see "Current state" under Decisions.)



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

\- Current state (final, Oct 2026): the redesigned pipeline (scripts/run_pipeline_v2.py) replaced the 4-agent pipeline, which was less accurate than a single Grader. Retrieval Planner (teacher-graded examples for the same question) -> five graders (qwen2.5:7b, mistral:7b, gemma2:9b via Ollama, TF-IDF classifier, fine-tuned DeBERTa-v3-base) -> logistic-regression judge (fault-aware, separate judges for seen and new questions) -> safety layer (message envelopes, reference checks, input check, canary check, misread-word check for photos, low-confidence escalation to a teacher) -> Feedback with the verdict line fixed to the judge's label. Final results: results/pipeline_v2/<set>_robust_full_refsim_deberta-v3-base-nli/, summarised in results/report_pack.html. Photo reader: src/ocr.py with qwen2.5vl:7b. HANDOFF.md has the latest status. The decisions below are the history of the original pipeline.

\- Main config of the original pipeline: v2 prompt (rubric definitions, no examples, single-step) running on qwen2.5:7b. llama3.1:8b stays as the second model for side-by-side comparison.

\- Why: after fixing the mass-capture measurement bug (see finding below) and rerunning every version x model combination on all 120 tasks with prefilling, v2+qwen2.5:7b is now the best on both accuracy (59/120 = 0.492, the highest of any combo tested) and confidence gap (avg top probability 0.792 when correct vs 0.727 when wrong, gap 0.065, also the highest of any combo tested). No combo tested has a strong gap in absolute terms, so treat this as "least bad," not "solved" - a confidence threshold gate built on this will still let a meaningful share of wrong answers through.

\- Superseded decision: v6 + llama3.1:8b was previously recorded as the main config, based on a 0.183 confidence gap measured on 48 tasks without prefilling. That decision is retracted; see finding below.

\- Finding - the mass-capture bug manufactured an apparent signal: llama3.1:8b often does not answer with a bare letter as its literal first token (it wants to start writing a sentence), so without prefilling, only a small slice of its real output distribution was being renormalized into the reported confidence. On v6+llama3.1:8b, the like-for-like comparison on the same 120 tasks (results/grader\_v6\_llama3.1-8b.json vs results/grader\_v6\_prefill\_llama3.1-8b.json) shows prefilling (planting "Label: " so the next token must be a letter) raised average captured probability mass from 0.657 to 0.994 (tasks below 0.5 mass: 28/120 -> 0/120), and the confidence gap went from 0.047 to 0.013. The 0.183 gap that first motivated the v6+llama choice came from an earlier 48-task run whose results file was later overwritten, so it cannot be re-derived and should not be compared directly with the 120-task numbers. Lesson: a confidence number is only trustworthy if you also check how much of the real output distribution it was computed from (see "mass" in src/llm.py's choice\_probabilities).

\- Finding - the verifier method's "No." tic on qwen2.5:7b is real, confirmed on 20 tasks (results/verifier\_tic\_full\_answers.json): the model was asked to verify a label with prefill "Answer: " and its full (untruncated) explanation was read alongside its first word. In 4 of 20 cases the first word ("No") directly contradicted the explanation's own conclusion (which affirmed the label was correct) - always in the same direction, opening negative then concluding positive, never the reverse. A further 3 cases were ambiguous because the explanation got cut off before stating an explicit verdict. So roughly 1 in 5 verifier confidence readings may be systematically too low, not just noisy. Also tried prefill "Answer: Yes/No because" as a literal alternative: it is degenerate (avg captured mass 0.000) because the token right after "because" is naturally a continuation word ("the", "you", ...), never Yes/No - so that specific prefill text does not work, though "Answer: " (mass 0.999, AUROC 0.611 on this 20-task check) does. Lesson: reading a verdict from only the first token assumes the first token IS the verdict, which is not always true for a model with contrastive-opener habits - worth keeping in mind for any first-token-only confidence method, not just verifier.

\- All four version x model combinations retested with prefilling on the full 120 tasks (accuracy, avg captured mass, confidence gap): v2+qwen2.5:7b 0.492 / 0.981 / 0.065; v2+llama3.1:8b 0.317 / 0.996 / 0.014; v6+qwen2.5:7b 0.408 / 0.991 / -0.000; v6+llama3.1:8b 0.358 / 0.994 / 0.013.

\- Confirmed main config: v2 + qwen2.5:7b, accuracy 0.492 (59/120), the best of everything tested. llama3.1:8b stays as the second model for comparison.

\- Conclusion - the confidence gate is a tested and rejected mitigation, not a proposed one: all four confidence-estimation methods tried on v2+qwen2.5:7b (first\_token, verbalized, self\_consistency, verifier) gave AUROC between 0.547 and 0.605 on the full 120 tasks - barely above 0.5 (chance) and far short of what a safety gate would need to reliably tell right answers from wrong ones before they reach the student. This applies on top of the earlier finding that raw confidence gaps are also weak (see above). No method or prompt version tested so far gives a usable confidence signal.



\## Rules

\- Keep code simple: plain Python functions, no classes unless truly needed.

\- Comment every function in plain language. I am still learning Python.

\- Do not install new libraries without telling me why.

\- Never delete files in data/ or results/.

\- After each task, tell me exactly which command to run to test it.

\- Explain what you changed in simple terms before I run anything.



\## Layout

data/ (tasks), src/ (main code), scripts/ (experiment runners), results/ (outputs)

