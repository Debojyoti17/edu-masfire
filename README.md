# Fault-tolerant multi-agent grading (MAS-FIRE extension)

A four-agent pipeline (Planner → Tutor → Grader → Feedback) grades short science answers with
small local language models. We inject faults into the pipeline on purpose, measure how often
wrong feedback reaches the student, and test safety checks that could stop it.

**Status:** work in progress. The pipeline, fault injection, safety checks, confidence-method
study and a non-LLM baseline are implemented; larger-scale runs are under way.

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

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Model wrapper (`llm.py`), rubric, the four agents, the fault definitions |
| `scripts/` | Experiment runners, analysis and chart scripts |
| `data/` | `tasks.json` (120-task evaluation set), `examples.json` (8 worked examples) |
| `results/` | Result files, summaries and charts |
| `get_splits.py`, `make_tasks.py`, `make_eval_sets.py`, `make_examples.py` | Build the datasets |

## Setup

1. Python 3.13 and a virtual environment: `pip install -r requirements.txt`
2. [Ollama](https://ollama.com) with the models used: `ollama pull qwen2.5:7b` and `ollama pull llama3.1:8b`
   (also `mistral:7b`, `gemma2:9b` for the extended runs)
3. Data: `python get_splits.py` then `python make_eval_sets.py` (downloads SciEntsBank from Hugging Face)

All model calls use temperature 0, seed 42 and a 4096-token context.

## Reproducing the main results

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
