# Report pack for the team's final report: every result table, a short factual methods summary,
# the list of figures and the limitations, in one web page (results/report_pack.html). Open it in a
# browser and copy tables into Word; they keep their layout. All numbers are read from the result
# files. The summary text is reference material - the team writes the report prose in its own words.
#
# Usage: python -m scripts.make_report_pack
import html
import json
import os
from collections import Counter

from scripts.compare_eval_sets import grader_file
from scripts.run_pipeline import load_jsonl
from scripts.run_pipeline_v2 import FAULTS_V2

OUT = os.path.join("results", "report_pack.html")
FINAL = "_robust_full_deberta-v3-base-nli"
SETS = [("120", "120-task set"), ("test_ua", "test_ua: familiar questions"),
        ("test_uq", "test_uq: new questions"), ("test_ud_600", "test_ud: new topics (600 sample)")]
parts = []


def table(caption, headers, rows, source):
    """Add a table with a caption and the file(s) its numbers come from."""
    parts.append(f"<h3>{html.escape(caption)}</h3><table><tr>"
                 + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr>"
                 + "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
                 + f"</table><p class='src'>Source: {html.escape(source)}</p>")


def text(title, lines):
    """Add a heading and a bulleted list."""
    parts.append(f"<h2>{html.escape(title)}</h2><ul>" + "".join(f"<li>{html.escape(l)}</li>" for l in lines) + "</ul>")


def share(recs, test):
    """Share of records for which test(record) is true."""
    return sum(test(r) for r in recs) / len(recs)


def wrong_reaching(r):
    """A wrong grade that reached the student (not held back for the teacher)."""
    return not r["escalated"] and r["delivered"] != r["gold_label"]


def v2(set_name, fault, ft, suffix=FINAL):
    """Saved records of the redesigned pipeline."""
    return load_jsonl(os.path.join("results", "pipeline_v2", set_name + suffix, f"{fault}_{ft}.jsonl"))


def main():
    """Build every section and save the page."""
    sizes = {s: len(json.load(open(p, encoding="utf-8"))) for s, p in
             [("train sample", "data/eval_train_sample_2000.json"), ("120", "data/tasks.json"),
              ("test_ua", "data/eval_test_ua.json"), ("test_uq", "data/eval_test_uq.json"),
              ("test_ud_600", "data/eval_test_ud_600.json")]}
    text("Methods summary (reference notes)", [
        "Data: SciEntsBank (SemEval-2013 Task 7) from Hugging Face nkazi/SciEntsBank; 4 labels (correct, "
        "contradictory, partially_correct_incomplete, irrelevant); the rare non_domain label is excluded.",
        f"Evaluation sets: 120 tasks (30 per label, from test_ua); test_ua {sizes['test_ua']} answers (questions seen "
        f"in training); test_uq {sizes['test_uq']} (new questions); test_ud {sizes['test_ud_600']} (random sample, new "
        f"domains, seed 42). Judge training sample: {sizes['train sample']} train answers.",
        "Hardware and software: one laptop, RTX 4060 8 GB, Python 3.13, Ollama 0.40.9; temperature 0, seed 42, "
        "4096-token context for every model call.",
        "Original pipeline: Planner (key points) -> Tutor (compares) -> Grader (label + first-token probabilities, "
        "reply prefilled with 'Label: ') -> Feedback (verdict + advice). 7 faults injected one at a time.",
        "Redesigned pipeline: retrieval Planner (8 teacher-graded answers to the same question, else 8 similar answers "
        "from other questions plus a zero-shot prompt); graders qwen2.5:7b, mistral:7b, gemma2:9b, a TF-IDF + logistic "
        "regression classifier and a fine-tuned DeBERTa-v3-base (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, 3 epochs, "
        "lr 2e-5, on the 4,946-answer train split); a logistic-regression judge stacks their probabilities.",
        "Judge training: separate judges for seen and new questions; 5-fold cross-validation (grouped by question for "
        "new questions) chooses regularisation; fault-aware copies where one LLM grader gives a random wrong label "
        "(confidence 0.5-0.99) or crashes; escalation threshold = lowest confidence where cross-validated accuracy >= 0.80.",
        "Safety layer: message envelopes (task ID, question fingerprint, checksum; re-request on failure); "
        "reference-answer check against the question bank; input check (empty / not ending like a sentence); canary "
        "check (32 known-grade answers, a grader is benched if binomial P(score this low) < 0.01); confidence escalation; "
        "Feedback verdict line fixed to the judge's label by prefilling.",
        "Faults (redesigned pipeline): " + "; ".join(f"{k} = {v}" for k, v in FAULTS_V2.items() if k != "none") + ".",
        "Metrics: accuracy; wrong feedback reaching the student (wrong verdict and not held back); share held back "
        "for a teacher; share where a check fired. Exact McNemar tests with Holm correction compare each fault with "
        "the fault-free run (results/pipeline_v2_summary.md).",
    ])

    # Original pipeline vs single grader
    rows = []
    for m in ["qwen2.5:7b", "llama3.1:8b", "mistral:7b", "gemma2:9b"]:
        single = json.load(open(grader_file(m, "120"), encoding="utf-8"))
        pipe = load_jsonl(os.path.join("results", "pipeline", m.replace(":", "-"), "none.jsonl"))
        rows.append([m, f"{share(single, lambda r: r['predicted_label'] == r['gold_label']):.3f}",
                     f"{share(pipe, lambda r: r['delivered'] == r['gold_label']):.3f}" if len(pipe) == 120 else "-"])
    table("Table 1. Original pipeline vs a single Grader (120 tasks, no faults)",
          ["Model", "Single Grader accuracy", "Pipeline accuracy (verdict the student reads)"], rows,
          "results/grader_v2_prefill*.json, results/pipeline/<model>/none.jsonl")

    # Final pipeline, no faults
    rows = []
    for set_name, label in SETS:
        off, on = v2(set_name, "none", "off"), v2(set_name, "none", "on")
        rows.append([label, len(off), f"{share(off, lambda r: r['final_label'] == r['gold_label']):.3f}",
                     f"{share(off, wrong_reaching):.1%}", f"{share(on, wrong_reaching):.1%}",
                     f"{share(on, lambda r: r['escalated']):.1%}"])
    table("Table 2. Redesigned pipeline, no faults", ["Set", "Answers", "Judge accuracy", "Wrong to student (safety off)",
          "Wrong to student (safety on)", "Held for teacher"], rows, f"results/pipeline_v2/<set>{FINAL}/none_*.jsonl")

    # Faults
    for set_name, label in [("test_ua", "familiar questions, 537 answers"), ("test_uq", "new questions, 733 answers")]:
        rows = []
        for fault in FAULTS_V2:
            off, on = v2(set_name, fault, "off"), v2(set_name, fault, "on")
            rows.append([fault, f"{share(off, lambda r: r['final_label'] == r['gold_label']):.3f}",
                         f"{share(off, wrong_reaching):.1%}", f"{share(on, wrong_reaching):.1%}",
                         f"{share(on, lambda r: r['escalated']):.1%}",
                         f"{share(on, lambda r: any('->' in d or 'benched' in d for d in r['detections'])):.0%}"])
        table(f"Table 3{'a' if set_name == 'test_ua' else 'b'}. Every fault, safety layer off and on ({label})",
              ["Fault", "Judge accuracy (off)", "Wrong to student (off)", "Wrong to student (on)", "Held (on)",
               "A check fired (on)"], rows, f"results/pipeline_v2/{set_name}{FINAL}/; p-values in results/pipeline_v2_summary.md")

    # Equal hold-back comparison
    rows = []
    for set_name, label in SETS:
        row = [label]
        for suffix in ["_robust_full", FINAL]:
            recs = v2(set_name, "none", "on", suffix)
            ranked = sorted(recs, key=lambda r: r["confidence"])
            keep = ranked[round(0.25 * len(recs)):]
            row.append(f"{sum(r['final_label'] != r['gold_label'] for r in keep) / len(recs):.1%}")
        rows.append(row)
    table("Table 4. Wrong grades reaching students when the least confident 25% go to the teacher",
          ["Set", "Judge: 3 LLMs + TF-IDF", "Judge: + fine-tuned DeBERTa"], rows,
          "results/pipeline_v2/<set>_robust_full and <set>" + FINAL + " (none_on.jsonl)")

    # Budget tiers
    bench = json.load(open(os.path.join("results", "budget", "benchmark.json"), encoding="utf-8"))
    acc = {(j["judge"], j["set"]): j for j in bench["judges"]}
    names = [("full", "3 LLMs (7-9B) + TF-IDF"), ("full + deberta-v3-base-nli", "3 LLMs (7-9B) + TF-IDF + DeBERTa"),
             ("budget + deberta-v3-base-nli", "3 small LLMs (2-3B) + TF-IDF + DeBERTa"),
             ("no LLM: TF-IDF + deberta-v3-base-nli", "TF-IDF + DeBERTa (no LLM, CPU only)"),
             ("no LLM: TF-IDF + deberta-v3-xsmall", "TF-IDF + DeBERTa-xsmall (no LLM, CPU only)")]
    rows = [[name] + [f"{acc[(key, s)]['accuracy']:.3f}" for s in ["test_ua", "test_uq", "test_ud_600"]]
            for key, name in names if (key, "test_ua") in acc]
    table("Table 5. Judge accuracy by hardware tier", ["Graders combined by the judge", "test_ua", "test_uq", "test_ud"],
          rows, "results/budget/benchmark.json")
    rows = [[g["model"], g["set"], g["mode"], g["n"], f"{g['accuracy']:.3f}"] for g in bench["single_graders"]]
    table("Table 6. Single graders on their own", ["Grader", "Set", "Examples shown", "Answers", "Accuracy"], rows,
          "results/rag/*.jsonl, results/finetuned/<model>/<set>.json")
    rows = [[c["model"], f"{c['cpu_seconds_per_answer']} s"] for c in bench["cpu_timing"]]
    prefix = json.load(open(os.path.join("results", "answer_bank", "prefix_cache_test_uq.json"), encoding="utf-8"))
    rows += [[f"{p['model']} on GPU, {p['order']}", f"{p['seconds_per_answer']} s (accuracy {p['accuracy']:.3f})"]
             for p in prefix]
    table("Table 7. Speed per answer", ["Grader and setting", "Seconds per answer"], rows,
          "results/budget/benchmark.json (CPU, 10-50 test_ua answers), results/answer_bank/prefix_cache_test_uq.json (613 answers)")

    # Answer bank
    from scripts.analyze_answer_bank import workload
    from src.judge import LABELS
    import numpy as np
    rows = []
    for set_name in ["test_uq", "test_ud_600"]:
        run = json.load(open(os.path.join("results", "answer_bank", f"{set_name}.json"), encoding="utf-8"))
        tasks = json.load(open(os.path.join("data", f"eval_{set_name}.json"), encoding="utf-8"))
        gold = {t["task_id"]: t["gold_label"] for t in tasks}
        q = len(Counter(t["question"] for t in tasks))
        for k in ["0", "2", "4", "8"]:
            ans = run["per_answer"][k]
            probs = np.array(list(ans.values()))
            right = [LABELS[int(p.argmax())] == gold[i] for i, p in zip(ans, probs)]
            rows.append([set_name, k, len(ans), f"{np.mean(right):.3f}", workload(probs.max(axis=1), right, int(k) * q, 0.05)])
    table("Table 8. Teacher answer bank (simulation)", ["Set", "Answers per question graded first", "Answers scored",
          "Accuracy", "Answers graded by hand for <= 5% wrong"], rows, "results/answer_bank/<set>.json; results/answer_bank_summary.md")

    figures = [("01_grader_accuracy.png", "Single Grader accuracy, prompt versions and models"),
               ("02_confidence_auroc.png", "Confidence methods: AUROC (all 0.55-0.61)"),
               ("03_mass_capture.png", "Measurement finding: prefilling the answer"),
               ("05_pipeline_harm.png", "Original pipeline: harm done by each fault"),
               (f"09_v2_old_vs_new_120{FINAL}.png", "Old vs redesigned pipeline"),
               (f"10_v2_faults_120{FINAL}.png", "Faults with the safety layer off and on"),
               ("11_judge_tradeoff.png", "Teacher review vs wrong grades reaching students"),
               ("12_budget_tiers.png", "Accuracy by hardware tier"),
               ("13_answer_bank.png", "Teacher answer bank simulation")]
    text("Figures (results/charts/)", [f"{f}: {d}" for f, d in figures if os.path.exists(os.path.join("results", "charts", f))])
    text("Limitations", [
        "One dataset (SciEntsBank, English school science); typed answers only.",
        "Accuracy 0.72 on familiar questions and 0.63-0.64 on new ones: teacher review stays necessary.",
        "A wrong reference answer for a brand-new question is not detected.",
        "Feedback advice text has not been rated by people; only its verdict line is checked.",
        "Faults are injected one at a time; the input check relies on answers ending like a sentence, which suits this "
        "cleaned dataset but should be tuned for real pupils' typing.",
        "Results from before 23 Oct 2026 used Ollama 0.40.4; the redesigned pipeline ran on 0.40.9.",
    ])
    text("AI tool use", ["Claude Code (Anthropic) wrote most code, ran the experiments and produced the analysis scripts, "
                         "charts and slides; Google Antigravity produced a read-only review and pitch draft. State the "
                         "extent honestly in the report as the course requires."])

    style = ("body{font-family:Calibri,Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#111}"
             "table{border-collapse:collapse;margin:8px 0}th,td{border:1px solid #999;padding:4px 8px;font-size:14px}"
             "th{background:#d9e2f3}.src{color:#555;font-size:12px;margin-top:2px}h3{margin-top:28px}")
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>Report pack</title><style>{style}</style></head>"
            "<body><h1>Report pack: Fault-tolerant multi-agent grading (Team 2)</h1>"
            "<p>All numbers are generated from the result files by scripts/make_report_pack.py.</p>"
            + "".join(parts) + "</body></html>")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
