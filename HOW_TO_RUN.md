# How to run the project

**Fault-tolerant multi-agent grading of short student answers (MAS-FIRE extension)**
Team 2: Debojyoti Barai Mitul (24BCE2951), Sammo Sarker (24BCE2962), Prottoy Saha (24BCE2939)

Everything runs locally on one laptop. No paid API or internet connection is needed after setup.

## 1. Requirements

- Windows, Linux or macOS with Python 3.13
- [Ollama](https://ollama.com) (tested with version 0.40.9) for the local language models
- An NVIDIA GPU with 8 GB of memory is recommended (we used an RTX 4060 8 GB). The no-LLM and
  small-model setups also run on a CPU only.
- About 30 GB of disk space for the models

## 2. Setup (one time, needs internet)

```
python -m venv .venv
.venv\Scripts\activate                 (Windows)   or   source .venv/bin/activate   (Linux/macOS)
pip install -r requirements.txt
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128     (GPU build of PyTorch)

ollama pull qwen2.5:7b
ollama pull mistral:7b
ollama pull gemma2:9b
ollama pull llama3.1:8b                 (second model of the original pipeline)
ollama pull qwen2.5:3b                  (budget-laptop tier, optional)
ollama pull llama3.2:3b                 (budget-laptop tier, optional)
ollama pull gemma2:2b                   (budget-laptop tier, optional)

python get_splits.py                    downloads the SciEntsBank dataset into data/raw/
python make_eval_sets.py                builds the test sets used in the experiments
```

The fine-tuned grader starts from two pre-trained DeBERTa models from Hugging Face:

```
hf download MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli --local-dir models/deberta-v3-base-nli
hf download microsoft/deberta-v3-xsmall --local-dir models/deberta-v3-xsmall
```

Keep Ollama running in the background (the Ollama app, or `ollama serve`). All model calls use
temperature 0 and seed 42, so results are repeatable.

## 3. Quick start: grade a class from a spreadsheet

```
python -m scripts.grade_csv examples/class_quiz.csv
```

The input CSV has the columns: student, question, reference_answer, student_answer, teacher_label
(teacher_label is optional). The tool writes two files next to the input:

- `class_quiz_graded.csv`: every answer with its grade, confidence, status (auto / teacher) and feedback
- `class_quiz_review.csv`: only the answers a teacher must check

The first run trains the judge (a few minutes); later runs start in seconds.

## 4. Reproducing the experiments

All commands are run from the project folder. Results are written to `results/`.

| Step | Command | Time on RTX 4060 |
|---|---|---|
| Single grader, original pipeline | `python -m scripts.run_grader v2 qwen2.5:7b --prefill` | ~5 min |
| Original 4-agent pipeline with faults | `python -m scripts.run_pipeline all qwen2.5:7b` then `python -m scripts.analyze_pipeline` | ~1 h |
| LLM grader runs for the redesigned pipeline | `python -m scripts.run_rag_queue` | ~10 h |
| TF-IDF baseline | `python -m scripts.run_tfidf_baseline` | ~1 min |
| Fine-tuned DeBERTa grader | `python -m scripts.train_finetuned_grader deberta-v3-base-nli` | ~35 min |
| Redesigned pipeline (v2) with all faults, safety layer off/on | `python -m scripts.run_pipeline_v2 data/tasks.json` (also `data/eval_test_ua.json`, and `data/eval_test_uq.json --verdict-only`, `data/eval_test_ud_600.json --verdict-only`) | 15 min to 1 h per set |
| Summary tables | `python -m scripts.analyze_pipeline_v2` -> `results/pipeline_v2_summary.md` | seconds |
| Teacher answer bank simulation | `python -m scripts.simulate_answer_bank test_uq` then `python -m scripts.analyze_answer_bank` | ~25 min |
| Faster grading with a shared prompt prefix | `python -m scripts.test_prefix_cache` | ~15 min |
| Budget-laptop comparison (small models, CPU only) | `python -m scripts.benchmark_budget --cpu-timing` | ~5 min |
| Charts | `python -m scripts.make_charts` -> `results/charts/` | seconds |

Every long run saves its progress and continues where it stopped if it is interrupted.

## 5. Where things are

| Path | Contents |
|---|---|
| `src/` | Model wrapper (`llm.py`), agents, graders, retrieval Planner, judge, safety checks, fine-tuned grader |
| `scripts/` | Experiment runners, analysis, charts, slides, the spreadsheet tool |
| `data/` | `tasks.json` (120-task evaluation set), `examples.json`; the full dataset is downloaded by `get_splits.py` |
| `examples/` | Example class spreadsheet for `grade_csv` |
| `README.md` | Project overview |

## 6. Notes

- If Ollama reports a GPU memory error, the code waits and retries automatically.
- Do not run several experiments at the same time on one GPU, and do not set `OLLAMA_NUM_PARALLEL`:
  both change the model outputs slightly.
- Parts of the code were written with an AI coding assistant (Claude Code); see README.md.
