# Build the presentation deck (results/slides/presentation.pptx) from the charts and result files.
# Slides carry charts, tables and factual titles only; every slide's speaker notes say
# "[Your talking points]" so the team writes the explanations themselves.
#
# Needs: python -m scripts.make_charts first. Optional logo: results/slides/vit_logo.png
# Usage: python -m scripts.make_slides
import json
import os

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from scripts.compare_eval_sets import grader_file
from scripts.run_pipeline import load_jsonl
from src.faults import FAULTS

OUT = os.path.join("results", "slides", "presentation.pptx")
LOGO = os.path.join("results", "slides", "vit_logo.png")
CHARTS = os.path.join("results", "charts")
NAVY = RGBColor(0x1F, 0x38, 0x64)
INK = RGBColor(0x0B, 0x0B, 0x0B)
GREY = RGBColor(0x52, 0x51, 0x4E)
HEADER_FILL = RGBColor(0xD9, 0xE2, 0xF3)
MODELS = ["qwen2.5:7b", "llama3.1:8b", "mistral:7b", "gemma2:9b"]
NOTES = "[Your talking points]"
FOOTER = "Fault-tolerant multi-agent grading  |  Final presentation, 24 October 2026"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def text_box(slide, left, top, width, height, text, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT):
    """Add a text box; each line of text becomes its own paragraph."""
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size, run.font.bold, run.font.color.rgb = Pt(size), bold, color
    return box


def new_slide(title, number):
    """A blank slide with a title, a footer, a slide number and the talking-points note."""
    slide = prs.slides.add_slide(BLANK)
    text_box(slide, 0.5, 0.3, 12.3, 0.8, title, size=28, bold=True, color=NAVY)
    text_box(slide, 0.5, 7.0, 9, 0.35, FOOTER, size=10, color=GREY)
    text_box(slide, 12.0, 7.0, 0.8, 0.35, str(number), size=10, color=GREY, align=PP_ALIGN.RIGHT)
    slide.notes_slide.notes_text_frame.text = NOTES
    return slide


def picture(slide, path, left=0.6, top=1.25, max_w=12.1, max_h=5.65):
    """Place an image as large as fits in the content area, keeping its shape, centred."""
    w, h = Image.open(path).size
    scale = min(max_w / w, max_h / h)
    width, height = w * scale, h * scale
    slide.shapes.add_picture(path, Inches(left + (max_w - width) / 2), Inches(top + (max_h - height) / 2),
                             Inches(width), Inches(height))


def table(slide, headers, rows, left, top, widths, size=14, row_h=0.45):
    """A simple table with a shaded header row."""
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(left), Inches(top),
                                   Inches(sum(widths)), Inches(row_h * (len(rows) + 1)))
    t = shape.table
    for j, w in enumerate(widths):
        t.columns[j].width = Inches(w)
    for i, row in enumerate([headers] + rows):
        for j, value in enumerate(row):
            cell = t.cell(i, j)
            cell.text = str(value)
            para = cell.text_frame.paragraphs[0]
            para.font.size, para.font.bold = Pt(size), i == 0
            para.font.color.rgb = INK
            cell.fill.solid()
            cell.fill.fore_color.rgb = HEADER_FILL if i == 0 else RGBColor(0xFF, 0xFF, 0xFF)


def accuracy(recs, field="predicted_label"):
    """Share of records whose field matches the gold label."""
    return sum(r[field] == r["gold_label"] for r in recs) / len(recs)


def judge_numbers():
    """Judge accuracy and confidence AUROC on full test_ua (familiar questions), and judge
    accuracy on test_uq (new questions), computed from the saved grader runs."""
    from scripts.compare_confidence import auroc
    from src.judge import features, judge_predict, train_judge
    out = []
    for regime, set_name in [("seen", "test_ua"), ("unseen", "test_uq")]:
        model, _, _ = train_judge(regime)
        tasks = json.load(open(f"data/eval_{set_name}.json", encoding="utf-8"))
        X, _ = features(tasks, set_name, regime)
        labels, probs = judge_predict(model, X)
        right = [l == t["gold_label"] for l, t in zip(labels, tasks)]
        out.append((sum(right) / len(right), auroc(list(zip(probs.max(axis=1), right)))))
    return out[0][0], out[0][1], out[1][0]


def main():
    """Build every slide in order and save the deck."""
    n = 0

    # 1. Title
    slide = prs.slides.add_slide(BLANK)
    if os.path.exists(LOGO):
        slide.shapes.add_picture(LOGO, Inches(5.92), Inches(0.6), Inches(1.5), Inches(1.5))
    text_box(slide, 0.8, 2.5, 11.7, 1.0, "Fault-Tolerant Multi-Agent Grading", size=40, bold=True,
             color=NAVY, align=PP_ALIGN.CENTER)
    text_box(slide, 0.8, 3.5, 11.7, 0.6, "Injecting faults into a Planner-Tutor-Grader-Feedback pipeline (MAS-FIRE extension)",
             size=20, color=GREY, align=PP_ALIGN.CENTER)
    text_box(slide, 0.8, 4.6, 11.7, 1.2, "[Team member names and registration numbers]\n[Course / faculty]",
             size=16, color=GREY, align=PP_ALIGN.CENTER)
    text_box(slide, 0.8, 6.2, 11.7, 0.5, "24 October 2026", size=14, color=GREY, align=PP_ALIGN.CENTER)
    slide.notes_slide.notes_text_frame.text = NOTES

    # 2. Pipeline and setup
    n = 2
    slide = new_slide("The pipeline and experimental setup", n)
    names = ["Planner", "Tutor", "Grader", "Feedback"]
    jobs = ["Key points a full\nanswer needs", "Compares answer with\nkey points", "Picks label +\nconfidence",
            "Writes the message\nthe student sees"]
    for i, (name, job) in enumerate(zip(names, jobs)):
        x = 0.7 + i * 3.15
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.5), Inches(2.5), Inches(1.4))
        box.fill.solid()
        box.fill.fore_color.rgb = HEADER_FILL
        box.line.color.rgb = NAVY
        tf = box.text_frame
        tf.text = name
        tf.paragraphs[0].font.size, tf.paragraphs[0].font.bold = Pt(20), True
        tf.paragraphs[0].font.color.rgb = NAVY
        p = tf.add_paragraph()
        p.text = job
        p.font.size = Pt(12)
        p.font.color.rgb = INK
        if i < 3:
            arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 2.6), Inches(2.0), Inches(0.45), Inches(0.4))
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = GREY
            arrow.line.fill.background()
    tasks = json.load(open("data/tasks.json", encoding="utf-8"))
    setup = [
        f"Data: SciEntsBank (SemEval-2013 Task 7). Main set: {len(tasks)} answers, 30 per label "
        "(correct, contradictory, partially correct, irrelevant).",
        "Larger sets: 537 unseen answers (test_ua), 733 unseen-question answers (test_uq), "
        "600 unseen-domain answers (test_ud sample).",
        "Models, run locally with Ollama on one RTX 4060 (8 GB): " + ", ".join(MODELS) + ".",
        "All calls at temperature 0, seed 42. Grader confidence read from first-token probabilities.",
        "Non-LLM comparison: TF-IDF + logistic regression trained on the SciEntsBank train split.",
    ]
    text_box(slide, 0.7, 3.4, 12.0, 3.3, "\n".join("•  " + s for s in setup), size=16)

    # 3. Faults
    n += 1
    slide = new_slide("Seven faults injected one at a time", n)
    rows = [[f, desc] for f, desc in FAULTS.items() if f != "none"]
    table(slide, ["Fault", "What breaks"], rows, 0.7, 1.4, [3.2, 8.7], size=15, row_h=0.6)

    # 4-11. Charts
    charts = [
        ("Grading accuracy on the 120-task set", "01_grader_accuracy.png"),
        ("Single Grader across evaluation sets", "07_eval_sets.png"),
    ]
    for title, file in charts:
        n += 1
        picture(new_slide(title, n), os.path.join(CHARTS, file))

    n += 1
    slide = new_slide("Single Grader vs full pipeline (120 tasks, no faults)", n)
    rows = []
    for m in MODELS:

        gpath = grader_file(m, "120")
        ppath = os.path.join("results", "pipeline", m.replace(":", "-"), "none.jsonl")
        single = accuracy(json.load(open(gpath, encoding="utf-8"))) if os.path.exists(gpath) else None
        pipe = load_jsonl(ppath)
        rows.append([m, f"{single:.3f}" if single is not None else "-",
                     f"{accuracy(pipe, 'delivered'):.3f}" if len(pipe) == 120 else "-"])
    table(slide, ["Model", "Single Grader accuracy", "Full pipeline accuracy (verdict the student reads)"],
          rows, 1.2, 1.6, [3.0, 3.6, 4.6], size=16, row_h=0.6)

    for title, file in [("Harm done by each fault", "05_pipeline_harm.png"),
                        ("Larger test set: fault effects on 537 answers", "08_test_ua_fault_effects.png"),
                        ("Safety checks in the pipeline (qwen2.5:7b)", "06_pipeline_checks_qwen2.5-7b.png"),
                        ("Confidence methods (v2 + qwen2.5:7b)", "02_confidence_auroc.png"),
                        ("Measurement finding: prefilling the answer", "03_mass_capture.png"),
                        ("Main Grader confusion matrix", "04_confusion_main.png")]:
        if os.path.exists(os.path.join(CHARTS, file)):
            n += 1
            picture(new_slide(title, n), os.path.join(CHARTS, file))

    # Scope
    n += 1
    slide = new_slide("Scope of the current results", n)
    test_ua_done = os.path.exists(os.path.join(CHARTS, "08_test_ua_fault_effects.png"))
    scope = [
        ("Pipeline + fault results: 120 answers per model and fault, plus 537 answers (test_ua) per model."
         if test_ua_done else
         "Pipeline + fault results: 120 tasks per model and fault; larger test sets are still running."),
        "One dataset (SciEntsBank); deterministic decoding (temperature 0), so no run-to-run variance.",
        "Faults chosen by the team, not yet mapped to a published fault taxonomy.",
        "Feedback quality is scored only by the verdict it states; no human rating yet.",
        "Statistics: 95% Wilson / bootstrap intervals; paired McNemar tests with Holm correction.",
    ]
    text_box(slide, 0.7, 1.4, 12.0, 4.5, "\n".join("•  " + s for s in scope), size=18)

    # Redesigned pipeline (only once its charts exist)
    if os.path.exists(os.path.join(CHARTS, "09_v2_old_vs_new_120.png")):
        n += 1
        slide = new_slide("Redesigned pipeline (in progress, for the final report)", n)
        boxes = [("Planner", "retrieves teacher-graded\nexamples"), ("3 LLM graders", "+ trained classifier"),
                 ("Judge", "combines grades,\ngives confidence"), ("Feedback", "verdict fixed to\nthe judge's label")]
        for i, (name, job) in enumerate(boxes):
            x = 0.7 + i * 3.15
            box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.4), Inches(2.5), Inches(1.3))
            box.fill.solid()
            box.fill.fore_color.rgb = HEADER_FILL
            box.line.color.rgb = NAVY
            tf = box.text_frame
            tf.text = name
            tf.paragraphs[0].font.size, tf.paragraphs[0].font.bold = Pt(18), True
            tf.paragraphs[0].font.color.rgb = NAVY
            p = tf.add_paragraph()
            p.text = job
            p.font.size = Pt(12)
            p.font.color.rgb = INK
            if i < 3:
                arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 2.6), Inches(1.85), Inches(0.45), Inches(0.4))
                arrow.fill.solid()
                arrow.fill.fore_color.rgb = GREY
                arrow.line.fill.background()
        seen_acc, seen_auc, unseen_acc = judge_numbers()
        points = [
            "Safety layer: every message carries a task ID, question fingerprint and checksum (lost, misrouted "
            "or corrupted messages are re-requested); the reference answer is checked against the question bank; "
            "low-confidence grades go to a teacher.",
            f"Judge accuracy {seen_acc:.3f} on 537 familiar-question answers (old main setup 0.490), "
            f"{unseen_acc:.3f} on 733 new-question answers.",
            f"Judge confidence separates right from wrong grades: AUROC {seen_auc:.3f} (earlier confidence methods 0.55-0.61).",
            "Status: fault experiments done on the 120-task set; the 537 / 733 / 600-answer runs are in progress.",
        ]
        text_box(slide, 0.7, 3.1, 12.0, 3.6, "\n".join("•  " + s for s in points), size=16)
        for title, file in [("Redesigned pipeline: wrong feedback reaching the student", "09_v2_old_vs_new_120.png"),
                            ("Redesigned pipeline: faults with the safety layer off and on", "10_v2_faults_120.png")]:
            n += 1
            picture(new_slide(title, n), os.path.join(CHARTS, file))

    # Paper plan
    n += 1
    slide = new_slide("Plan to a paper-ready version", n)
    plan = [
        ["Final runs, tables and figures", "Code / experiments", "1 day"],
        ["Map the 7 faults to a published fault taxonomy (MAS-FIRE); add missing types and rerun",
         "Team reads, code implements", "2-3 days"],
        ["Improve agent prompts using the training split (pipeline currently below single Grader)",
         "Code / experiments", "1-2 days"],
        ["Stronger safety check (full verdict, not first word) with trade-off curves", "Code / experiments", "1-2 days"],
        ["Human check: 2 team members label about 150 feedback messages", "Team", "8-10 hours"],
        ["Related work (25-30 papers) and paper writing in the ACL template", "Team", "25-35 hours"],
    ]
    table(slide, ["Step", "Who", "Estimated effort"], plan, 0.7, 1.4, [7.6, 2.6, 1.7], size=14, row_h=0.65)
    text_box(slide, 0.7, 6.25, 12.0, 0.5, "Target: a workshop paper (e.g. BEA, educational NLP at ACL venues); "
             "total about 1.5-2 weeks of combined work.", size=14, color=GREY)

    # AI disclosure
    n += 1
    slide = new_slide("AI tool use disclosure", n)
    table(slide, ["Tool", "Used for", "Extent (fill in)"],
          [["Claude Code (Anthropic)", "Code, running experiments, analysis scripts, charts, this slide layout", "[state honestly]"],
           ["[other tool, if any]", "", ""]],
          0.7, 1.5, [3.3, 5.6, 3.0], size=15, row_h=0.7)

    # Close
    n += 1
    slide = new_slide("Questions", n)
    text_box(slide, 0.7, 3.0, 12.0, 1.0, "Code and results: github.com/Debojyoti17/edu-masfire", size=22,
             color=GREY, align=PP_ALIGN.CENTER)

    prs.save(OUT)
    print(f"Saved {OUT} with {len(prs.slides)} slides")


if __name__ == "__main__":
    main()
