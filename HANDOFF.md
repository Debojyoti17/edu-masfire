# Handoff notes (state on 2026-10-24, 15:00)

Read this together with CLAUDE.md before continuing the work in a new session.

## Deadlines
- Presentation was delayed (originally Sat 2026-10-24). Final report due 2026-10-30.
- Aim afterwards: a workshop paper (e.g. BEA).

## What is done
- Redesigned pipeline v2 (`scripts/run_pipeline_v2.py`): retrieval Planner, 3 LLM graders
  (qwen2.5:7b, mistral:7b, gemma2:9b) + TF-IDF, fault-aware ("robust") judge, Tutor + Feedback with
  the verdict fixed to the judge label, safety layer (message checks, reference check, input check,
  canary check, escalation).
- Final runs with robust judge + full checks: `results/pipeline_v2/<set>_robust_full/` for 120,
  test_ua, test_uq, test_ud_600. Summary: `results/pipeline_v2_summary.md`
  (`python -m scripts.analyze_pipeline_v2`). Charts 09/10 `_120_robust_full` in `results/charts/`.
- Answer-bank simulation (`scripts/simulate_answer_bank.py`, summary `results/answer_bank_summary.md`
  via `python -m scripts.analyze_answer_bank`). Finding: a few teacher-graded answers per new question
  raise accuracy (test_uq 0.651 -> 0.700, test_ud 0.533 -> 0.689) but do NOT reduce teacher workload
  at equal safety (the no-bank judge's confidence already filters as well).
- Spreadsheet workflow `scripts/grade_csv.py` (example `examples/class_quiz.csv`) - written, NOT yet run end to end.
- Small "budget laptop" models run through `scripts.run_rag_grader` (results in `results/rag/`):
  qwen2.5:3b 0.592 on test_ua/same (qwen2.5:7b 0.609), llama3.2:3b 0.547, gemma2:2b 0.376 (too weak).
  Prefill choices for them are in `src/agents.py` MODEL_PREFILLS.

## Running in the background when this was written (each step starts by itself)
Log: `results/budget/logs/chain2.txt`
1. llama3.2:3b on test_uq / test_ud_600 (similar, zeroshot)
2. fine-tuning check `results/finetuned/logs/debug_base_nli.txt` (RESULT line: should be well above 0.43)
3. `scripts.train_finetuned_grader` for deberta-v3-base-nli and deberta-v3-xsmall
   (logs `results/finetuned/logs/<name>.txt`, outputs `results/finetuned/<name>/`, model `models/finetuned/<name>/`)
4. `scripts.test_prefix_cache` -> `results/answer_bank/prefix_cache_test_uq.json`
5. `scripts.benchmark_budget --cpu-timing` -> `results/budget/benchmark.json`

## Known issue being fixed
Fine-tuning first collapsed to the majority label (0.406). Cause found: transformers 5 loads the
checkpoint in float16; fixed with `dtype=torch.float32` in `train_model` and `src/finetuned.py`.
A quick xsmall test after the fix still learned slowly (0.426 on 2000 answers / 2 epochs), so check
the debug RESULT and fold-1 accuracy before trusting the full run. If still stuck: try without bf16
autocast, more epochs, or lr 1e-5/3e-5.

## Update 2026-10-25
- Fine-tuning fixed and done: deberta-v3-base-nli alone 0.698 test_ua / 0.610 test_uq / 0.610 test_ud.
  The judge now uses it (EXTRA_GRADERS in src/judge.py); final runs in
  `results/pipeline_v2/<set>_robust_full_deberta-v3-base-nli/`. Budget results in results/budget/benchmark.json
  (TF-IDF + DeBERTa with no LLM: 0.708 / 0.593 / 0.622 at 0.12 s per answer on CPU).
  Speed trick (fixed example order): 3x faster for qwen/mistral, results/answer_bank/prefix_cache_test_uq.json.
- The user paused work until their weekly usage resets (Wed 2026-10-28). Remaining before 30 Oct:
  1. run grade_csv end to end; 2. charts + slides with final numbers (new judge's auto threshold holds
  back only ~6% on seen questions - compare judges at equal hold-back, see conversation); 3. tables and
  methods summary for the team's report; 4. optional: photo input (local vision model via Ollama, e.g.
  qwen2.5vl:3b, needs internet once + user approval) with "misread handwriting" as a new fault.

## Update 2026-10-28 (13:30)
- Final runs are in `results/pipeline_v2/<set>_robust_full_refsim_deberta-v3-base-nli/` (adds the
  new-question reference check and the handwriting_misread fault). Analysis, charts, the 20-slide
  `results/slides/final_presentation.pptx` and `results/report_pack.html` (Tables 1-9) are rebuilt from them.
- Photo prototype (src/ocr.py, qwen2.5vl:7b): stress test on 300 handwriting-font sheets
  (scripts/stress_test_reader.py) gives 0.2% (clean) / 0.6% (blurry, tilted, faded) of characters
  wrong, 0 false misread flags. Backup demo: examples/photo_quiz/ (6 font sheets, all read perfectly,
  5 of 6 graded right). Simulated misreading on test_ua: wrong to student 33.5% off -> 6.1% on.
- Saved demo transcripts in results/demo/ (tasks 23, 23 grader_lying, 1 student_truncate, 0 label_corrupted).
- Waiting on the team (sent 28 Oct): Sammo writes H01-H15 + rates rater_A.csv, Prottoy writes H16-H30 +
  rates rater_B.csv. When back: photos -> results/handwriting/photos/ (any names; rename by the code on
  the sheet), then `python -m scripts.evaluate_handwriting`; sheets -> results/human_rating/, then
  `python -m scripts.analyze_rating`; then rebuild slides and report pack.

## Next steps (older list)
1. If fine-tuning works: set `EXTRA_GRADERS` in `src/judge.py` only if it raises the judge's
   cross-validated accuracy (compare `train_judge(regime, extras=[name])`), then rerun the v2 pipeline.
2. Run `python -m scripts.grade_csv examples/class_quiz.csv` end to end and fix anything that breaks.
3. Read `results/budget/benchmark.json` and `results/answer_bank/prefix_cache_test_uq.json`; write
   up the budget tier and speed results.
4. Update charts (`scripts/make_charts.py`) and slides (`scripts/make_slides.py`) with the final
   `_robust_full` numbers, answer bank, budget tier; the plan slide is out of date.
5. Commit locally; push to GitHub only after asking the user.

## Rules to keep (also in CLAUDE.md and memory)
Never delete files in data/ or results/. Ask before installing libraries or pushing. Don't touch the
user's other project. Ask the user through popups. Ollama 0.40.9; never use OLLAMA_NUM_PARALLEL.
