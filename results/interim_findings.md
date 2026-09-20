# Interim Findings

Date: 2026-09-20

Every number below is computed directly from the JSON files in `results/` and `data/`; each table names its source files. No interpretation is included. Section 8 lists numbers that appear elsewhere in this repository but cannot be checked against a file.

## 1. What has been built so far

### src/

- `src/llm.py` - Ollama wrapper. `generate()` returns the model's text reply (optional `max_tokens`, `temperature` and `seed` overrides; defaults are temperature 0, seed 42, num_ctx 4096). `choice_probabilities()` asks a multiple-choice question and returns per-option probabilities read from the first generated token's logprobs, plus the raw probability mass those options captured; it has an optional `prefill` argument that adds a final assistant message for the model to continue from.
- `src/rubric.py` - Written definitions of the four labels (correct, contradictory, partially_correct_incomplete, irrelevant), used in prompt versions v2-v6.

### scripts/

- `scripts/run_grader.py` - Grades every task in `data/tasks.json` with a chosen prompt version (v1-v6), optional model argument and `--prefill` flag; saves to `results/grader_{version}[_prefill][_{model}].json`.
- `scripts/run_grader_only.py` - Earlier single-prompt grader run (label-names-only prompt, qwen2.5:7b); saves `results/grader_only.json` and prints overall and per-label accuracy.
- `scripts/analyze_grader.py` - Reads one results file; reports the confidently-wrong count (top probability > 0.9), average top probability for correct vs wrong predictions, and example cases. Its helper functions are imported by other scripts.
- `scripts/compare_grader_versions.py` - Prints accuracy, average top probability when correct/wrong and confidently-wrong count for every `results/grader_v*.json` file.
- `scripts/run_confidence_method.py` - Runs v2 grading with one of four confidence methods (first_token, verbalized, self_consistency, verifier); saves `results/confidence_{method}.json`.
- `scripts/compare_confidence.py` - Reports accuracy, confidence gap, AUROC and confidently-wrong count for the four confidence methods.
- `scripts/check_verifier_tic.py` - On 20 tasks, generates the verifier's full answer and compares two prefill variants (`"Answer: "` and `"Answer: Yes/No because"`) by captured mass and AUROC; saves `results/verifier_tic_*.json`.

### Repo root

- `get_data.py` - Fetches the SciEntsBank split list and first rows over HTTP (datasets-server).
- `make_tasks.py` - Builds `data/tasks.json` (see section 2).
- `make_examples.py` - Builds `data/examples.json` (8 rows from the train split).
- `test_logprobs.py` - Prototype: one Ollama call with logprobs, prints the raw response.
- `test_confidence.py` - Prototype: P(Yes) from the top-20 first-token logprobs for four questions.
- `test_llm.py` - Grades the first task in `data/tasks.json` through `src/llm.py` and prints the probabilities.

## 2. Dataset

- Source: Hugging Face dataset `nkazi/SciEntsBank`, fetched over HTTP from `datasets-server.huggingface.co` (`make_tasks.py`).
- Split used for `data/tasks.json`: `test_ua`. Split used for `data/examples.json`: `train`.
- The source has five labels (correct, contradictory, partially_correct_incomplete, irrelevant, non_domain). `non_domain` is excluded; the comment in `make_tasks.py` gives the reason as "too rare (only 3 examples) and not a real grading case".
- Tasks in `data/tasks.json`: 120 (120 unique task_ids, 80 distinct questions). Fields per task: task_id, question, reference_answer, student_answer, gold_label.
- Tasks per gold label: correct 30, contradictory 30, partially_correct_incomplete 30, irrelevant 30.
- Sampling (`make_tasks.py`): rows are requested in batches of 100 at offsets 0-500; for each of the four labels, `random.sample` draws 30 rows with `random.seed(42)`; the combined list is then shuffled with `random.shuffle`. An earlier version of the same script drew 12 per label (48 tasks).
- Worked examples in `data/examples.json`: 8 rows from the `train` split, correct 2, contradictory 2, partially_correct_incomplete 2, irrelevant 2. task_ids shared between `examples.json` and `tasks.json`: 0.

## 3. Baseline accuracy by prompt version and model

Prompt versions (from the header of `scripts/run_grader.py`): v1 = label names only; v2 = v1 plus the rubric definitions; v3 = v2 plus the 8 worked examples; v4 = two-step (reasoning, then letter); v5 = v4 with contradictory and irrelevant merged into "incorrect" (3 labels); v6 = two-step where step 1 may only describe the comparison and may not name a label.

### Runs on the 48-task set (results files with 48 records)

| version | model | prefill | correct/n | accuracy | avg captured mass | source file |
|---|---|---|---|---|---|---|
| v1 | qwen2.5:7b* | no | 20/48 | 0.417 | 1.000 | `results/grader_v1.json` |
| v1 (run_grader_only.py) | qwen2.5:7b* | no | 20/48 | 0.417 | 1.000 | `results/grader_only.json` |
| v2 | qwen2.5:7b* | no | 22/48 | 0.458 | 1.000 | `results/grader_v2.json` |
| v3 | qwen2.5:7b* | no | 21/48 | 0.438 | 1.000 | `results/grader_v3.json` |
| v4 | qwen2.5:7b* | no | 19/48 | 0.396 | 1.000 | `results/grader_v4.json` |
| v5 (3-label scoring) | qwen2.5:7b* | no | 20/48 | 0.417 | 1.000 | `results/grader_v5.json` |
| v6 | qwen2.5:7b | no | 18/48 | 0.375 | 1.000 | `results/grader_v6.json` |
| v2 | llama3.1:8b | no | 22/48 | 0.458 | 0.941 | `results/grader_v2_llama3.1-8b.json` |

### Runs on the 120-task set (results files with 120 records)

| version | model | prefill | correct/n | accuracy | avg captured mass | source file |
|---|---|---|---|---|---|---|
| v6 | llama3.1:8b | no | 50/120 | 0.417 | 0.657 | `results/grader_v6_llama3.1-8b.json` |
| v2 | qwen2.5:7b | yes | 59/120 | 0.492 | 0.981 | `results/grader_v2_prefill.json` |
| v2 | llama3.1:8b | yes | 38/120 | 0.317 | 0.996 | `results/grader_v2_prefill_llama3.1-8b.json` |
| v6 | qwen2.5:7b | yes | 49/120 | 0.408 | 0.991 | `results/grader_v6_prefill.json` |
| v6 | llama3.1:8b | yes | 43/120 | 0.358 | 0.994 | `results/grader_v6_prefill_llama3.1-8b.json` |

Notes:

- `*` = the results file has no `model` field (it predates the field); the model is taken from the filename convention (no suffix = qwen2.5:7b, `llama3.1-8b` suffix = llama3.1:8b). Files without a `prefill` field predate the flag and are listed as `no`.
- All 48-record files contain the same task_ids in the same order: True. 23 of those 48 task_ids appear in the current 120-task `data/tasks.json`.
- `grader_only.json` and `grader_v1.json` use the same prompt; predicted labels differ on 0 of 48 tasks.
- `results/baseline_v6_llama.json` is content-identical to `results/grader_v6_llama3.1-8b.json`: True. It is not listed separately.
- v5 is scored against gold labels with contradictory and irrelevant merged, so its accuracy is on a 3-label problem.

## 4. Per-label accuracy and predicted-label distribution

Main configuration: v2 + qwen2.5:7b with prefill, 120 tasks (`results/grader_v2_prefill.json`).

### Main configuration: per-label accuracy

| gold label | correct/n | accuracy |
|---|---|---|
| correct | 13/30 | 0.433 |
| contradictory | 11/30 | 0.367 |
| partially_correct_incomplete | 12/30 | 0.400 |
| irrelevant | 23/30 | 0.767 |

### Main configuration: confusion matrix (rows = gold label, columns = predicted label)

| gold \ predicted | correct | contradictory | partially_correct_incomplete | irrelevant |
|---|---|---|---|---|
| correct | 13 | 3 | 10 | 4 |
| contradictory | 1 | 11 | 4 | 14 |
| partially_correct_incomplete | 4 | 3 | 12 | 11 |
| irrelevant | 0 | 5 | 2 | 23 |

### All 120-task configurations: per-label accuracy

| gold label | v2 + qwen2.5:7b, prefill | v2 + llama3.1:8b, prefill | v6 + qwen2.5:7b, prefill | v6 + llama3.1:8b, prefill | v6 + llama3.1:8b, no prefill |
|---|---|---|---|---|---|
| correct | 13/30 = 0.433 | 11/30 = 0.367 | 0/30 = 0.000 | 7/30 = 0.233 | 26/30 = 0.867 |
| contradictory | 11/30 = 0.367 | 7/30 = 0.233 | 17/30 = 0.567 | 11/30 = 0.367 | 10/30 = 0.333 |
| partially_correct_incomplete | 12/30 = 0.400 | 20/30 = 0.667 | 28/30 = 0.933 | 22/30 = 0.733 | 3/30 = 0.100 |
| irrelevant | 23/30 = 0.767 | 0/30 = 0.000 | 4/30 = 0.133 | 3/30 = 0.100 | 11/30 = 0.367 |

### All 120-task configurations: number of times each label was predicted (30 gold tasks per label)

| configuration | correct | contradictory | partially_correct_incomplete | irrelevant |
|---|---|---|---|---|
| v2 + qwen2.5:7b, prefill | 18 | 22 | 28 | 52 |
| v2 + llama3.1:8b, prefill | 20 | 9 | 91 | 0 |
| v6 + qwen2.5:7b, prefill | 0 | 22 | 94 | 4 |
| v6 + llama3.1:8b, prefill | 16 | 18 | 82 | 4 |
| v6 + llama3.1:8b, no prefill | 76 | 21 | 7 | 16 |

## 5. Confidence methods (v2 + qwen2.5:7b, 120 tasks)

| method | accuracy | avg confidence when correct | avg confidence when wrong | gap | AUROC | confidently wrong (> 0.9) | records without confidence |
|---|---|---|---|---|---|---|---|
| first_token | 0.492 | 0.792 | 0.727 | 0.065 | 0.605 | 13/61 | 0 |
| verbalized | 0.375 | 0.960 | 0.941 | 0.019 | 0.547 | 53/75 | 0 |
| self_consistency | 0.483 | 0.983 | 0.958 | 0.025 | 0.554 | 50/62 | 0 |
| verifier | 0.492 | 0.271 | 0.182 | 0.089 | 0.582 | 6/61 | 0 |

Definitions (from `scripts/run_confidence_method.py`):

- first_token: probability of the predicted letter from prefilled logprobs (prefill `'Label: '`); read from `results/grader_v2_prefill.json`, not re-queried. Values equal that file's top probabilities: True.
- verbalized: the model is asked for a label and a 0-100 confidence in words; both are parsed from the reply.
- self_consistency: 5 samples per task at temperature 0.7 (seeds 42-46); label = majority letter, confidence = fraction of samples agreeing with it.
- verifier: a second call shows the label chosen by first_token and asks "Is this label correct? Yes or No"; confidence = P(Yes) from prefilled logprobs (prefill `'Answer: '`). Its predicted labels are identical to first_token's: True.

## 6. Mass-capture measurement (v6 + llama3.1:8b, same 120 tasks)

"Captured mass" is the total first-token probability that fell on the answer letters A-D before renormalizing. The prefill text is `'Label: '` (`scripts/run_grader.py`). Both files hold the same task_ids in the same order: True.

| | no prefill | prefill |
|---|---|---|
| source file | `results/grader_v6_llama3.1-8b.json` | `results/grader_v6_prefill_llama3.1-8b.json` |
| accuracy | 50/120 = 0.417 | 43/120 = 0.358 |
| avg captured mass | 0.657 | 0.994 |
| min captured mass | 0.137 | 0.947 |
| tasks with mass < 0.5 | 28/120 (23.3%) | 0/120 (0.0%) |
| avg top probability when correct | 0.756 | 0.867 |
| avg top probability when wrong | 0.708 | 0.854 |
| gap (correct - wrong) | 0.047 | 0.013 |
| confidently wrong (top prob > 0.9) | 9/70 | 40/77 |

The average captured mass for every other results file is in the last column of the tables in section 3.

## 7. Model calls represented by the results files

One call = one request to the Ollama chat endpoint. Counts are derived from the number of records in each file and the call structure in the scripts (a record with a `reasoning` field came from a generate call plus a decision call; other grader records came from one decision call; self_consistency uses `SELF_CONSISTENCY_RUNS = 5` generate calls per task).

| results file | records | generate calls | choice_probabilities calls | total | note |
|---|---|---|---|---|---|
| `results/grader_v1.json` | 48 | 0 | 48 | 48 | single decision call |
| `results/grader_only.json` | 48 | 0 | 48 | 48 | single decision call |
| `results/grader_v2.json` | 48 | 0 | 48 | 48 | single decision call |
| `results/grader_v3.json` | 48 | 0 | 48 | 48 | single decision call |
| `results/grader_v4.json` | 48 | 48 | 48 | 96 | reasoning step + decision step |
| `results/grader_v5.json` | 48 | 48 | 48 | 96 | reasoning step + decision step |
| `results/grader_v6.json` | 48 | 48 | 48 | 96 | reasoning step + decision step |
| `results/grader_v2_llama3.1-8b.json` | 48 | 0 | 48 | 48 | single decision call |
| `results/grader_v6_llama3.1-8b.json` | 120 | 120 | 120 | 240 | reasoning step + decision step |
| `results/grader_v2_prefill.json` | 120 | 0 | 120 | 120 | single decision call |
| `results/grader_v2_prefill_llama3.1-8b.json` | 120 | 0 | 120 | 120 | single decision call |
| `results/grader_v6_prefill.json` | 120 | 120 | 120 | 240 | reasoning step + decision step |
| `results/grader_v6_prefill_llama3.1-8b.json` | 120 | 120 | 120 | 240 | reasoning step + decision step |
| `results/confidence_first_token.json` | 120 | 0 | 0 | 0 | derived from grader_v2_prefill.json, no new calls |
| `results/confidence_verbalized.json` | 120 | 120 | 0 | 120 | 1 generate per task |
| `results/confidence_self_consistency.json` | 120 | 600 | 0 | 600 | 5 generate per task |
| `results/confidence_verifier.json` | 120 | 0 | 120 | 120 | 1 choice call per task with a predicted label |
| `results/verifier_tic_full_answers.json` | 20 | 20 | 0 | 20 | 1 generate per task |
| `results/verifier_tic_prefill_answer.json` | 20 | 0 | 20 | 20 | 1 choice call per task |
| `results/verifier_tic_prefill_answer_yes-no_because.json` | 20 | 0 | 20 | 20 | 1 choice call per task |
| **Total** | | **1244** | **1144** | **2388** | |

`results/baseline_v6_llama.json` is not counted separately because it is a content-identical copy of `results/grader_v6_llama3.1-8b.json`. Calls whose output files were later overwritten, and calls made in interactive checks that saved no file, are not counted; the total is a lower bound (see section 8).

## 8. Data availability notes

Numbers that appear elsewhere in this repository (CLAUDE.md or commit messages) but cannot be checked against any file in `results/`:

- The original 48-task `v6 + llama3.1:8b` run: `results/grader_v6_llama3.1-8b.json` was overwritten by the 120-task rerun (the file now holds 120 records). CLAUDE.md cites a confidence gap of 0.183 for that run and the commit message of `854d3fa` cites 3 confidently-wrong cases out of 29; neither can be recomputed.
- The 10-task prefill check (average captured mass 0.693 before, 0.994 after) was run interactively and no file was saved. Section 6 reports the same measurement on all 120 tasks from files instead.
- The comparison of the verifier's first word against its full explanation (CLAUDE.md: 4 of 20 disagreed, 3 ambiguous) was a manual reading of `results/verifier_tic_full_answers.json`; the classification itself is not stored in any file. The 20 full answers are in that file.
- The number of rows available per label in `test_ua` is printed by `make_tasks.py` when it runs and is not stored in any file.
- Interactive smoke tests and debugging calls are not represented in any results file.

