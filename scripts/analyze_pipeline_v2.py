# Summarise the redesigned pipeline (results/pipeline_v2/<set>/<fault>_<ft>.jsonl): for each fault,
# with the fault-tolerance layer off and on:
#   - accuracy of the judge's final label
#   - wrong feedback reaching the student (with ft on, grades the judge is unsure of are held back
#     for a teacher, so they do not reach the student)
#   - share held back for a teacher, share where a check fired, unreadable feedback verdicts
#   - whether the fault changes how often feedback is right (paired McNemar test, Holm-corrected)
# Writes results/pipeline_v2_summary.md.
#
# Usage: python -m scripts.analyze_pipeline_v2
import glob
import os

from scripts.analyze_pipeline import holm, mcnemar_p, pct
from scripts.run_pipeline import load_jsonl
from scripts.run_pipeline_v2 import FAULTS_V2

OUT = os.path.join("results", "pipeline_v2_summary.md")


def reached_wrong(r):
    """Wrong feedback that actually reached the student (not held back, verdict not the gold label)."""
    return (not r["escalated"]) and r["delivered"] != r["gold_label"]


def summarize(records, base_by_id):
    """All numbers for one fault + ft setting, compared with the fault-free run at the same setting."""
    n = len(records)
    harmed = sum((base_by_id[r["task_id"]]["delivered"] == r["gold_label"]) and r["delivered"] != r["gold_label"]
                 for r in records)
    fixed = sum((base_by_id[r["task_id"]]["delivered"] != r["gold_label"]) and r["delivered"] == r["gold_label"]
                for r in records)
    return {
        "n": n,
        "judge_acc": sum(r["final_label"] == r["gold_label"] for r in records) / n,
        "wrong_delivered": sum(r["delivered"] != r["gold_label"] for r in records) / n,
        "wrong_reaching": sum(reached_wrong(r) for r in records) / n,
        "held": sum(r["escalated"] for r in records) / n,
        "detected": sum(any("->" in d for d in r["detections"]) for r in records) / n,
        "verdict_unreadable": sum(any(d.startswith("feedback") for d in r["detections"]) for r in records) / n,
        "p": mcnemar_p(harmed, fixed),
    }


def set_table(folder):
    """Markdown table for one task set."""
    rows = []
    for ft in ["off", "on"]:
        base = load_jsonl(os.path.join(folder, f"none_{ft}.jsonl"))
        if not base:
            continue
        base_by_id = {r["task_id"]: r for r in base}
        stats = {}
        for fault in FAULTS_V2:
            recs = load_jsonl(os.path.join(folder, f"{fault}_{ft}.jsonl"))
            if len(recs) == len(base):
                stats[fault] = summarize(recs, base_by_id)
        faulty = [f for f in stats if f != "none"]
        for f, p in zip(faulty, holm([stats[f]["p"] for f in faulty])):
            stats[f]["p_holm"] = p
        for fault, s in stats.items():
            rows.append((fault, ft, s))
    if not rows:
        return None
    lines = [f"## {os.path.basename(folder)} ({rows[0][2]['n']} answers)", "",
             "| fault | safety layer | judge accuracy | wrong verdict written | wrong feedback reaching student | "
             "held back for teacher | a check fired | feedback verdict unreadable | p (vs no fault, Holm) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for fault, ft, s in sorted(rows, key=lambda r: (list(FAULTS_V2).index(r[0]), r[1])):
        p = f"{s['p_holm']:.3g}" if "p_holm" in s else "-"
        lines.append(f"| {fault} | {ft} | {s['judge_acc']:.3f} | {pct(s['wrong_delivered'])} | "
                     f"{pct(s['wrong_reaching'])} | {pct(s['held'])} | {pct(s['detected'])} | "
                     f"{pct(s['verdict_unreadable'])} | {p} |")
    return "\n".join(lines) + "\n"


def main():
    """Summarise every task-set folder (test runs with --limit are skipped)."""
    parts = ["# Redesigned pipeline: fault injection with the safety layer off and on", "",
             "Faults: " + "; ".join(f"**{k}** = {v}" for k, v in FAULTS_V2.items()), ""]
    for folder in sorted(glob.glob(os.path.join("results", "pipeline_v2", "*"))):
        if os.path.basename(folder) == "cache" or "_limit" in folder or not os.path.isdir(folder):
            continue
        table = set_table(folder)
        if table:
            parts.append(table)
    text = "\n".join(parts)
    print(text)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
