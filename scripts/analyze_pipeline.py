# Summarise pipeline runs from scripts/run_pipeline.py. For each model and fault it reports:
#   - how often wrong feedback reached the student,
#   - how the fault changed things compared with the fault-free run (harmed = right became wrong),
#   - how often the Feedback agent's verdict differed from the label the Grader handed it,
#   - two safety checks that hold back feedback for a human to review:
#       confidence check: hold back when the Grader's confidence is below a threshold
#       agreement check:  hold back when the delivered verdict differs from an independent second grader
# Prints the tables and writes them to results/pipeline_summary.md.
#
# Usage: python -m scripts.analyze_pipeline                     every folder in results/pipeline
#        python -m scripts.analyze_pipeline results/pipeline/qwen2.5-7b_limit3
import glob
import math
import os
import sys

from src.faults import FAULTS
from scripts.run_pipeline import load_jsonl

CONF_THRESHOLDS = [0.7, 0.9]
OUT = os.path.join("results", "pipeline_summary.md")


def pct(x):
    """Format a fraction as a percentage with one decimal place."""
    return f"{100 * x:.1f}%"


def mcnemar_p(b, c):
    """Exact two-sided McNemar test on paired results. b = tasks the fault turned from right to
    wrong, c = tasks it turned from wrong to right. A small p means the fault really changes
    how often feedback is right, not just by chance."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvals):
    """Holm correction: makes p-values stricter because we test several faults at once."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adjusted[i] = running
    return adjusted


def is_wrong(r):
    """Feedback is wrong when the verdict the student read is not the gold label (unclear counts as wrong)."""
    return r["delivered"] != r["gold_label"]


def check_stats(records, held_back):
    """For one safety check: share of tasks where wrong feedback still got through, share held back
    for a human, and share of correct feedback held back unnecessarily."""
    n = len(records)
    through = sum(1 for r in records if not held_back(r) and is_wrong(r))
    held = sum(1 for r in records if held_back(r))
    held_ok = sum(1 for r in records if held_back(r) and not is_wrong(r))
    return through / n, held / n, held_ok / n


def summarize(records, base_by_id):
    """All the numbers for one fault on one model."""
    n = len(records)
    harmed = fixed = changed = 0
    for r in records:
        b = base_by_id[r["task_id"]]
        changed += r["delivered"] != b["delivered"]
        harmed += (not is_wrong(b)) and is_wrong(r)
        fixed += is_wrong(b) and (not is_wrong(r))
    s = {
        "n": n,
        "wrong": sum(is_wrong(r) for r in records) / n,
        "changed": changed / n,
        "harmed": harmed / n,
        "fixed": fixed / n,
        "p": mcnemar_p(harmed, fixed),
        "mismatch": sum(r["delivered"] != r["sent_label"] for r in records) / n,
        "unclear": sum(r["delivered"] is None for r in records),
    }
    for t in CONF_THRESHOLDS:
        s[f"conf{t}"] = check_stats(records, lambda r, t=t: r["grader_conf"] is None or r["grader_conf"] < t)
    s["agree"] = check_stats(records, lambda r: r["delivered"] != r["check_label"])
    return s


def model_tables(folder):
    """Build the two markdown tables for one results folder (one model)."""
    base = load_jsonl(os.path.join(folder, "none.jsonl"))
    if not base:
        return None
    base_by_id = {r["task_id"]: r for r in base}
    rows = []
    for fault in FAULTS:
        recs = load_jsonl(os.path.join(folder, f"{fault}.jsonl"))
        if recs:
            rows.append((fault, summarize(recs, base_by_id)))

    faulty = [s for f, s in rows if f != "none"]
    for s, p in zip(faulty, holm([s["p"] for s in faulty])):
        s["p_holm"] = p

    model = base[0]["model"]
    lines = [f"## {model} ({len(base)} tasks per fault)", ""]
    lines += ["### Effect of each fault", "",
              "| fault | wrong feedback reached student | changed vs no fault | right became wrong | "
              "wrong became right | p (McNemar, Holm-adjusted) | Feedback verdict differs from Grader label | "
              "unclear verdicts |",
              "|---|---|---|---|---|---|---|---|"]
    for fault, s in rows:
        p = f"{s['p_holm']:.3g}" if "p_holm" in s else "-"
        lines.append(f"| {fault} | {pct(s['wrong'])} | {pct(s['changed'])} | {pct(s['harmed'])} | "
                     f"{pct(s['fixed'])} | {p} | {pct(s['mismatch'])} | {s['unclear']} |")
    lines += ["", "### Safety checks (wrong feedback that still reached the student / share held back for a human)", "",
              "| fault | no check | confidence < 0.7 | confidence < 0.9 | two graders must agree |",
              "|---|---|---|---|---|"]
    for fault, s in rows:
        cells = [f"{pct(s[k][0])} / {pct(s[k][1])}" for k in ("conf0.7", "conf0.9", "agree")]
        lines.append(f"| {fault} | {pct(s['wrong'])} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    """Summarise the folders given on the command line, or every pipeline folder by default."""
    folders = sys.argv[1:] or sorted(d for d in glob.glob(os.path.join("results", "pipeline", "*"))
                                     if os.path.isdir(d) and "_limit" not in d)
    parts = ["# Pipeline fault-injection results", "",
             "Faults: " + "; ".join(f"**{k}** = {v}" for k, v in FAULTS.items()), ""]
    for folder in folders:
        table = model_tables(folder)
        if table:
            parts.append(table)
    text = "\n".join(parts)
    print(text)
    if not sys.argv[1:]:
        with open(OUT, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
