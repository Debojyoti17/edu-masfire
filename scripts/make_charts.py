# Make the charts for the slides and paper from the saved result files. Saves PNGs in
# results/charts/. Pipeline charts are made only for models whose pipeline runs exist.
#
# Usage: python -m scripts.make_charts
import json
import math
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from scripts.analyze_pipeline import summarize
from scripts.compare_confidence import auroc
from scripts.run_pipeline import load_jsonl
from src.faults import FAULTS

OUT_DIR = os.path.join("results", "charts")
LABELS = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant"]

# Colours from the dataviz skill's validated reference palette (light mode). The first three
# categorical slots are safe together for colour-blind readers; ink colours are for text only.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
DEEMPH = "#c3c2b7"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

FAULT_NAMES = {
    "none": "No fault",
    "planner_omit": "Planner message lost",
    "planner_wrong_task": "Planner message misrouted",
    "tutor_wrong_task": "Tutor message misrouted",
    "tutor_overclaim": "Tutor overclaims",
    "grader_label_flip": "Label corrupted in transit",
    "reference_swap": "Wrong reference answer",
    "student_truncate": "Student answer truncated",
}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 11,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": False,
})


def load(path):
    """Read one JSON results file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k correct out of n."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def bootstrap_auroc(pairs, reps=2000, seed=42):
    """AUROC plus a 95% bootstrap interval: resample the tasks with replacement many times
    and take the middle 95% of the AUROC values."""
    rng = random.Random(seed)
    values = []
    for _ in range(reps):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        a = auroc(sample)
        if a is not None:
            values.append(a)
    values.sort()
    return auroc(pairs), values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def titles(fig, title, subtitle):
    """Put a bold title and a grey subtitle at the top left of the figure."""
    fig.text(0.02, 0.97, title, fontsize=14, fontweight="bold", va="top", color=INK)
    fig.text(0.02, 0.905, subtitle, fontsize=10.5, va="top", color=INK2)


def save(fig, name):
    """Save a figure as a PNG in the charts folder and close it."""
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)


def chart_accuracy():
    """Horizontal bars: accuracy of each 120-task configuration with 95% intervals.
    The main configuration is blue, the rest grey."""
    configs = [
        ("v2 + qwen2.5:7b (main)", "results/grader_v2_prefill.json"),
        ("v6 + qwen2.5:7b", "results/grader_v6_prefill.json"),
        ("v2 + llama3.1:8b", "results/grader_v2_prefill_llama3.1-8b.json"),
        ("v6 + llama3.1:8b", "results/grader_v6_prefill_llama3.1-8b.json"),
        ("v6 + llama3.1:8b, no prefill", "results/grader_v6_llama3.1-8b.json"),
    ]
    rows = []
    for name, path in configs:
        recs = load(path)
        k = sum(r["predicted_label"] == r["gold_label"] for r in recs)
        lo, hi = wilson(k, len(recs))
        rows.append((name, k / len(recs), lo, hi))
    if os.path.exists("results/baseline_tfidf_120.json"):
        recs = load("results/baseline_tfidf_120.json")
        k = sum(r["predicted_label"] == r["gold_label"] for r in recs)
        rows.append(("Trained TF-IDF baseline (non-LLM)",) + (k / len(recs),) + wilson(k, len(recs)))
    for model in ["qwen2.5:7b", "llama3.1:8b"]:
        recs = load_jsonl(os.path.join("results", "pipeline", model.replace(":", "-"), "none.jsonl"))
        if len(recs) == 120:
            k = sum(r["delivered"] == r["gold_label"] for r in recs)
            rows.append((f"Full pipeline, {model}, no faults",) + (k / len(recs),) + wilson(k, len(recs)))
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(9, 2.2 + 0.42 * len(rows)))
    fig.subplots_adjust(left=0.36, right=0.95, top=1 - 1.0 / (2.2 + 0.42 * len(rows)),
                        bottom=0.55 / (2.2 + 0.42 * len(rows)))
    y = range(len(rows))
    colors = [SERIES[0] if "main" in r[0] else SERIES[1] if "baseline" in r[0] else DEEMPH for r in rows]
    ax.barh(y, [r[1] for r in rows], height=0.5, color=colors)
    ax.errorbar([r[1] for r in rows], y, xerr=[[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]],
                fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    for i, r in enumerate(rows):
        ax.text(r[3] + 0.012, i, f"{r[1]:.3f}", va="center", fontsize=10, color=INK)
    ax.axvline(0.25, color=MUTED, linewidth=1)
    ax.text(0.252, len(rows) - 0.45, "chance (4 labels)", fontsize=9, color=MUTED)
    ax.set_yticks(list(y), [r[0] for r in rows])
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("Accuracy on the 120 tasks; bars show 95% Wilson intervals")
    ax.grid(axis="y", visible=False)
    titles(fig, "Grading accuracy on the 120-task set",
           "Blue: main LLM Grader. Orange: trained non-LLM baseline. Most intervals overlap.")
    save(fig, "01_grader_accuracy.png")


def chart_confidence_auroc():
    """Dot plot: AUROC of each confidence method with a 95% bootstrap interval, against chance (0.5)."""
    rows = []
    for m in ["first_token", "verbalized", "self_consistency", "verifier"]:
        recs = load(f"results/confidence_{m}.json")
        pairs = [(r["confidence"], r["predicted_label"] == r["gold_label"]) for r in recs if r["confidence"] is not None]
        rows.append((m,) + bootstrap_auroc(pairs))
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(9, 4.0))
    fig.subplots_adjust(left=0.20, right=0.95, top=0.78, bottom=0.15)
    y = list(range(len(rows)))
    ax.hlines(y, [r[2] for r in rows], [r[3] for r in rows], color=SERIES[0], linewidth=2)
    ax.scatter([r[1] for r in rows], y, s=70, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
    for i, r in enumerate(rows):
        ax.text(r[3] + 0.01, i, f"{r[1]:.3f}", va="center", fontsize=10, color=INK)
    ax.axvline(0.5, color=MUTED, linewidth=1)
    ax.text(0.503, len(rows) - 0.5, "chance", fontsize=9, color=MUTED)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0.35, 1.0)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlabel("AUROC for telling right grades from wrong ones (1.0 = perfect); lines show 95% bootstrap intervals")
    ax.grid(axis="y", visible=False)
    titles(fig, "No confidence method separates right from wrong grades well",
           "v2 + qwen2.5:7b, 120 tasks. Every method's interval reaches close to or below chance.")
    save(fig, "02_confidence_auroc.png")
    return rows


def chart_mass_capture():
    """Grouped histogram: how much of the model's first-token probability landed on the answer
    letters, without and with prefilling (v6 + llama3.1:8b, same 120 tasks)."""
    before = [r["mass"] for r in load("results/grader_v6_llama3.1-8b.json")]
    after = [r["mass"] for r in load("results/grader_v6_prefill_llama3.1-8b.json")]
    bins = [i / 10 for i in range(11)]

    def counts(values):
        """Number of tasks falling in each 0.1-wide bin."""
        c = [0] * 10
        for v in values:
            c[min(int(v * 10), 9)] += 1
        return c

    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.15)
    centers = [b + 0.05 for b in bins[:-1]]
    w = 0.038
    ax.bar([c - w / 2 - 0.001 for c in centers], counts(before), width=w, color=SERIES[1], label="No prefill")
    ax.bar([c + w / 2 + 0.001 for c in centers], counts(after), width=w, color=SERIES[0], label="Prefill \"Label: \"")
    ax.axvline(0.5, color=MUTED, linewidth=1)
    ax.text(0.505, ax.get_ylim()[1] * 0.92, "half the probability", fontsize=9, color=MUTED)
    ax.set_xticks(bins, [f"{b:.1f}" for b in bins])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability on the answer letters A-D (captured mass)")
    ax.set_ylabel("Tasks")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    low = sum(v < 0.5 for v in before)
    titles(fig, "Prefilling makes confidence readings measure the whole answer",
           f"v6 + llama3.1:8b, 120 tasks. Without prefill, {low} tasks had less than half their probability "
           f"on A-D (average {sum(before) / len(before):.2f}); with it, average {sum(after) / len(after):.2f}.")
    save(fig, "03_mass_capture.png")


def chart_confusion():
    """Heatmap: confusion matrix for the main configuration (rows = gold, columns = predicted)."""
    recs = load("results/grader_v2_prefill.json")
    m = [[sum(1 for r in recs if r["gold_label"] == g and r["predicted_label"] == p) for p in LABELS] for g in LABELS]
    short = ["correct", "contradictory", "partially\ncorrect", "irrelevant"]
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    fig.subplots_adjust(left=0.20, right=0.97, top=0.82, bottom=0.14)
    ax.imshow(m, cmap=cmap, vmin=0, vmax=30)
    for i in range(4):
        for j in range(4):
            v = m[i][j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=13,
                    color="white" if v >= 12 else INK, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(4), short)
    ax.set_yticks(range(4), short)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label (30 tasks each)")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    titles(fig, "Where the main Grader goes wrong",
           "v2 + qwen2.5:7b, 120 tasks. Diagonal = correct. It over-uses \"irrelevant\".")
    save(fig, "04_confusion_main.png")


def pipeline_stats(slug):
    """Summary numbers per fault for one model's pipeline run, or None if the run is incomplete."""
    folder = os.path.join("results", "pipeline", slug)
    base = load_jsonl(os.path.join(folder, "none.jsonl"))
    if len(base) < 120:
        return None
    base_by_id = {r["task_id"]: r for r in base}
    stats = {}
    for fault in FAULTS:
        recs = load_jsonl(os.path.join(folder, f"{fault}.jsonl"))
        if len(recs) == len(base):
            stats[fault] = summarize(recs, base_by_id)
    return stats


def chart_pipeline_harm(all_stats):
    """Grouped bars: share of tasks each fault turned from right to wrong, one bar per model."""
    models = [m for m in all_stats if all_stats[m]]
    faults = [f for f in FAULTS if f != "none" and all(f in all_stats[m] for m in models)]
    if not models or not faults:
        return
    fig, ax = plt.subplots(figsize=(9, 5.6))
    fig.subplots_adjust(left=0.27, right=0.95, top=0.82, bottom=0.20)
    h = 0.32
    for k, m in enumerate(models):
        ys = [i + (k - (len(models) - 1) / 2) * (h + 0.03) for i in range(len(faults))]
        ax.barh(ys, [100 * all_stats[m][f]["harmed"] for f in faults], height=h, color=SERIES[k], label=m)
    ax.set_yticks(range(len(faults)), [FAULT_NAMES[f] for f in faults])
    ax.invert_yaxis()
    ax.set_xlabel("Tasks where the fault turned correct feedback into wrong feedback (%)")
    ax.grid(axis="y", visible=False)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=len(models), bbox_to_anchor=(0.6, 0.0))
    titles(fig, "How much damage each injected fault does",
           "Compared with the same pipeline without the fault, 120 tasks per fault.")
    save(fig, "05_pipeline_harm.png")


def chart_pipeline_checks(model, stats):
    """Two panels for one model: wrong feedback that still reached the student, and the share
    of feedback held back for a human, with no check, a confidence check, and an agreement check."""
    faults = [f for f in FAULTS if f in stats]
    series = [("No check", lambda s: s["wrong"], None),
              ("Confidence below 0.9 held back", lambda s: s["conf0.9"][0], lambda s: s["conf0.9"][1]),
              ("Two graders must agree", lambda s: s["agree"][0], lambda s: s["agree"][1])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    fig.subplots_adjust(left=0.20, right=0.97, top=0.80, bottom=0.18, wspace=0.08)
    h = 0.26
    for k, (name, through, held) in enumerate(series):
        ys = [i + (k - 1) * (h + 0.02) for i in range(len(faults))]
        axes[0].barh(ys, [100 * through(stats[f]) for f in faults], height=h, color=SERIES[k], label=name)
        if held:
            axes[1].barh(ys, [100 * held(stats[f]) for f in faults], height=h, color=SERIES[k])
    axes[0].set_yticks(range(len(faults)), [FAULT_NAMES[f] for f in faults])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Wrong feedback reaching the student (%)")
    axes[1].set_xlabel("Feedback held back for a human (%)")
    for ax in axes:
        ax.set_xlim(0, 100)
        ax.grid(axis="y", visible=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.58, 0.0))
    titles(fig, f"Safety checks in the pipeline ({model})",
           "Lower on the left is better; the right panel is the cost (work sent to a human). 120 tasks per fault.")
    save(fig, f"06_pipeline_checks_{model.replace(':', '-')}.png")


def main():
    """Make every chart whose data exists."""
    os.makedirs(OUT_DIR, exist_ok=True)
    chart_accuracy()
    rows = chart_confidence_auroc()
    for m, a, lo, hi in sorted(rows):
        print(f"  AUROC {m}: {a:.3f} (95% CI {lo:.3f} to {hi:.3f})")
    chart_mass_capture()
    chart_confusion()
    all_stats = {}
    for model in ["qwen2.5:7b", "llama3.1:8b"]:
        stats = pipeline_stats(model.replace(":", "-"))
        if stats:
            all_stats[model] = stats
            chart_pipeline_checks(model, stats)
        else:
            print(f"(pipeline charts for {model} skipped - run not finished yet)")
    chart_pipeline_harm(all_stats)


if __name__ == "__main__":
    main()
