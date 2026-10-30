# Fault-tolerant multi-agent grading (MAS-FIRE extension)

A four-agent pipeline (Planner → Tutor → Grader → Feedback) grades short science answers with
small local language models. We inject faults into the pipeline on purpose, measure how often
wrong feedback reaches the student, and test safety checks that could stop it.

**Status:** work in progress. The original pipeline, a redesigned pipeline (v2) with a safety
layer, fault injection on both, and tools for classroom use are implemented.

## How it works

| Agent | Job |
|---|---|
| Planner | Lists the key points a fully correct answer must contain (sees question + reference answer). |
| Tutor | Compares the student answer with those key points, without naming a grade. |
| Grader | Picks one label (correct, contradictory, partially_correct_incomplete, irrelevant); its confidence is read from the model's first-token probabilities, with the reply prefilled so the next token must be the answer letter. |
| Feedback | Writes the message the student sees, starting with a verdict that is read back for scoring. |

Faults (`src/faults.py`), injected one at a time: lost or misrouted Planner message, misrouted
or over-claiming Tutor message, Grader label corrupted on its way to Feedback, swapped
reference answer, truncated student answer.

Safety checks evaluated: hold back low-confidence grades, and hold back feedback when an
independent second grader disagrees.

## Redesigned pipeline (v2)

The first pipeline turned out to be less accurate than a single Grader, so it was redesigned
(`scripts/run_pipeline_v2.py`):

1. **Planner (retrieval)** picks 8 teacher-graded answers as worked examples: answers to the same
   question when they exist (or from the teacher's answer bank), otherwise similar answers to other questions.
2. **Three LLM graders** (qwen2.5:7b, mistral:7b, gemma2:9b) and a **TF-IDF classifier** each give
   a probability for every label; optionally a **fine-tuned DeBERTa grader** too.
3. A **judge** (logistic regression trained on 2000 train answers, including copies where one
   grader lies or crashes) combines them into one label and a confidence.
4. **Tutor + Feedback** write the advice; the verdict line is fixed to the judge's label.

Safety layer: checked message envelopes (lost / misrouted / corrupted messages are re-requested),
reference-answer check against the question bank (and a question/reference word-match check for
new questions), input check (empty or cut-off answers), canary check (a grader that fails
known-grade answers is benched for the session), a misread-word check for answers read from
photos, and escalation of low-confidence grades to a teacher.

## Using it in a school

```
python -m scripts.grade_csv examples/class_quiz.csv          # grade a class from a spreadsheet
python -m scripts.simulate_answer_bank test_uq               # how much a few teacher-graded answers help
python -m scripts.benchmark_budget --cpu-timing              # smaller models / CPU-only laptop
python -m scripts.test_prefix_cache                          # faster grading with a shared prompt prefix
```

`grade_csv` reads a CSV (student, question, reference_answer, student_answer, teacher_label) and
writes `<name>_graded.csv` (grade, confidence, status, feedback) and `<name>_review.csv` (answers a
teacher must check). Answers the teacher has already graded become worked examples for the rest.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Model wrapper (`llm.py`), the original four agents (`agents.py`), the redesigned pipeline: Planner (`retrieval.py`), graders (`graders.py`, `finetuned.py`), judge (`judge.py`), safety layer (`pipeline_v2.py`, `health.py`), faults (`faults.py`), photo reader (`ocr.py`) |
| `scripts/` | Experiment runners, analysis, charts, slides and report pack; the demo (`demo_v2.py`); the teacher tools (`grade_csv.py`, `grade_photo.py`) |
| `tests/` | Automatic tests of the safety layer and fault injection (`python -m unittest discover -s tests -t .`) |
| `data/` | `tasks.json` (120-task evaluation set), `eval_test_*.json` (larger test sets), `examples.json` |
| `examples/` | Example class spreadsheets and answer-sheet images for the teacher tools |
| `results/` | Result files, summaries, charts, slides, report pack, handwriting and rating tests |
| `get_splits.py`, `make_tasks.py`, `make_eval_sets.py`, `make_examples.py` | Build the datasets |
| `scripts/early_checks/` | First quick checks of the Ollama connection and log-probabilities (kept for history) |

## Setup

1. Python 3.13 and a virtual environment: `pip install -r requirements.txt`
2. [Ollama](https://ollama.com) with the models used: `ollama pull qwen2.5:7b`, `mistral:7b`, `gemma2:9b`
   (redesigned pipeline), `qwen2.5vl:7b` (photo reader) and `llama3.1:8b` (original comparison)
3. Data: `python get_splits.py` then `python make_eval_sets.py` (downloads SciEntsBank from Hugging Face)

All model calls use temperature 0, seed 42 and a 4096-token context.

## Reproducing the main results

Redesigned pipeline (final results):

```
python -m scripts.run_pipeline_v2 data/tasks.json            # every fault, safety layer off and on (also test sets)
python -m scripts.analyze_pipeline_v2                        # tables -> results/pipeline_v2_summary.md
python -m scripts.make_report_pack                           # all tables -> results/report_pack.html
python -m scripts.evaluate_handwriting                       # photo test -> results/handwriting/summary.md
python -m unittest discover -s tests -t .                    # automatic tests of the safety layer
```

Original pipeline:

```
python -m scripts.run_grader v2 qwen2.5:7b --prefill        # single Grader, main configuration
python -m scripts.run_pipeline all qwen2.5:7b                # pipeline: no fault + every fault
python -m scripts.analyze_pipeline                           # summary tables -> results/pipeline_summary.md
python -m scripts.run_confidence_method verifier             # one of four confidence methods
python -m scripts.compare_confidence                         # compare the confidence methods
python -m scripts.run_tfidf_baseline                         # non-LLM baseline
python -m scripts.make_charts                                # charts -> results/charts/
```

Results summaries: `results/interim_findings.md`, `results/pipeline_summary.md`.

## Data

SciEntsBank (SemEval-2013 Task 7), via the Hugging Face dataset `nkazi/SciEntsBank`. The rare
`non_domain` label is excluded. Raw splits are downloaded by `get_splits.py` and are not
redistributed in this repository.

## AI assistance

Parts of the code and analysis were written with an AI coding assistant (Claude Code); commits
it co-authored carry a `Co-Authored-By` line.
