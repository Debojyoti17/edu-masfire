# Speed idea 3: prompt-prefix caching. Ollama remembers the work it did on the start of the last
# prompt; if the next prompt starts with exactly the same text, that part is not processed again.
# Our usual grading prompt puts the worked examples in a different order for every student answer
# (most similar first), so consecutive prompts differ early and little can be reused. If instead
# every answer to a question gets the same examples in the same order, the whole prompt up to the
# student's answer is shared, and only the new answer has to be read.
#
# Test on test_uq with the teacher's first 8 answers per question as the answer bank (the same
# setup as scripts/simulate_answer_bank.py, k=8), grading the same scored answers question by
# question, with both orders. Model answers are NOT taken from the cache here, because the point is
# to time them. Reports seconds per answer and accuracy for each model and order.
#
# Usage: python -m scripts.test_prefix_cache [test_uq]
import json
import os
import sys
import time

from src.graders import grade_with_prompt, rag_prompt
from src.judge import MODELS
from src.retrieval import retrieve, use_answer_bank
from scripts.simulate_answer_bank import bank_rows, split_pool


def main():
    """Time and score both example orders for every LLM grader; save results/answer_bank/prefix_cache_<set>.json."""
    set_name = sys.argv[1] if len(sys.argv) > 1 else "test_uq"
    tasks = json.load(open(os.path.join("data", f"eval_{set_name}.json"), encoding="utf-8"))
    pool, scored = split_pool(tasks)
    bank = bank_rows(pool, 8)
    use_answer_bank(bank)
    fixed = {q: [r for r in bank if r["question"] == q] for q in pool}
    results = []
    for model in MODELS:
        for order in ["similarity order (usual)", "fixed order (shared prefix)"]:
            grade_with_prompt(rag_prompt(scored[0], fixed[scored[0]["question"]], "same"), model)  # load the model
            right, start = 0, time.time()
            for t in scored:
                examples = retrieve(t, "same") if order.startswith("similarity") else fixed[t["question"]]
                label, _, _ = grade_with_prompt(rag_prompt(t, examples, "same"), model)
                right += label == t["gold_label"]
            results.append({"model": model, "order": order, "n": len(scored),
                            "seconds_per_answer": round((time.time() - start) / len(scored), 3),
                            "accuracy": round(right / len(scored), 3)})
            print(results[-1], flush=True)
    use_answer_bank([])
    with open(os.path.join("results", "answer_bank", f"prefix_cache_{set_name}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"Saved results/answer_bank/prefix_cache_{set_name}.json", flush=True)


if __name__ == "__main__":
    main()
