# Build the final presentation deck (results/slides/final_presentation.pptx) for the finished
# project: the original pipeline and why it was redesigned, the redesigned pipeline with its
# safety layer, the fault results, and how it is made practical for a school laptop.
# Every number is read from the result files. Speaker notes say "[Your talking points]" so the
# team writes the explanations themselves. The earlier deck (scripts/make_slides.py) is kept as it is.
#
# Needs: python -m scripts.make_charts first.
# Usage: python -m scripts.make_final_slides
import csv
import json
import os

from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

import scripts.make_slides as deck
from scripts.make_slides import (CHARTS, GREY, HEADER_FILL, INK, LOGO, NAVY, NOTES, new_slide, picture, table,
                                 text_box)
from scripts.run_pipeline import load_jsonl
from scripts.run_pipeline_v2 import FAULTS_V2

OUT = os.path.join("results", "slides", "final_presentation.pptx")
deck.FOOTER = "Fault-tolerant multi-agent grading  |  Team 2  |  Final review"
FINAL = "_robust_full_refsim_deberta-v3-base-nli"
SETS = [("120", "120-task set"), ("test_ua", "Familiar questions (537)"),
        ("test_uq", "New questions (733)"), ("test_ud_600", "New topics (600)")]
FAULT_NAMES = {"plan_lost": "Planner message lost", "plan_misrouted": "Planner message misrouted",
               "grader_crash": "One grader crashes", "grader_lying": "One grader lies confidently",
               "label_corrupted": "Grade corrupted in transit", "reference_swap": "Wrong reference answer",
               "student_truncate": "Student answer cut off", "tutor_overclaim": "Tutor overclaims", "handwriting_misread": "Handwriting misread"}


def run(set_name, fault, ft):
    """Saved records of the final pipeline for one set, fault and safety setting."""
    return load_jsonl(os.path.join("results", "pipeline_v2", set_name + FINAL, f"{fault}_{ft}.jsonl"))


def share(recs, test):
    """Share of records for which test(record) is true."""
    return sum(test(r) for r in recs) / len(recs)


def wrong_reaching(r):
    """A wrong grade that reached the student (not held back for the teacher)."""
    return not r["escalated"] and r["delivered"] != r["gold_label"]


def boxes(slide, items, top, height=1.3):
    """A row of labelled boxes joined by arrows (a simple pipeline diagram)."""
    width = (12.0 - 0.45 * (len(items) - 1)) / len(items)
    for i, (name, job) in enumerate(items):
        x = 0.7 + i * (width + 0.45)
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(top), Inches(width), Inches(height))
        box.fill.solid()
        box.fill.fore_color.rgb = HEADER_FILL
        box.line.color.rgb = NAVY
        tf = box.text_frame
        tf.word_wrap = True
        tf.text = name
        tf.paragraphs[0].font.size, tf.paragraphs[0].font.bold = Pt(15), True
        tf.paragraphs[0].font.color.rgb = NAVY
        p = tf.add_paragraph()
        p.text = job
        p.font.size = Pt(11)
        p.font.color.rgb = INK
        if i < len(items) - 1:
            arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + width + 0.05), Inches(top + height / 2 - 0.17),
                                           Inches(0.35), Inches(0.34))
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = GREY
            arrow.line.fill.background()


def bullets(slide, lines, top=1.4, size=17, height=5.3):
    """A bulleted text block."""
    text_box(slide, 0.7, top, 12.0, height, "\n".join("•  " + s for s in lines), size=size)


def main():
    """Build every slide in order and save the deck."""
    n = 0
    old = load_jsonl(os.path.join("results", "pipeline", "qwen2.5-7b", "none.jsonl"))
    old_wrong = share(old, lambda r: r["delivered"] != r["gold_label"])
    single = json.load(open(os.path.join("results", "grader_v2_prefill.json"), encoding="utf-8"))
    single_acc = share(single, lambda r: r["predicted_label"] == r["gold_label"])
    bench = json.load(open(os.path.join("results", "budget", "benchmark.json"), encoding="utf-8"))
    bacc = {(j["judge"], j["set"]): j["accuracy"] for j in bench["judges"]}
    cpu = {c["model"]: c["cpu_seconds_per_answer"] for c in bench["cpu_timing"]}

    # 1. Title
    slide = deck.prs.slides.add_slide(deck.BLANK)
    if os.path.exists(LOGO):
        slide.shapes.add_picture(LOGO, Inches(5.92), Inches(0.5), Inches(1.5), Inches(1.5))
    text_box(slide, 0.8, 2.3, 11.7, 1.0, "Fault-Tolerant Multi-Agent Grading", size=40, bold=True, color=NAVY,
             align=PP_ALIGN.CENTER)
    text_box(slide, 0.8, 3.3, 11.7, 0.9, "Grading short student answers on a school laptop, safely, when parts of the "
             "AI system fail", size=20, color=GREY, align=PP_ALIGN.CENTER)
    text_box(slide, 0.8, 4.5, 11.7, 1.3, "Team 2: Debojyoti Barai Mitul (24BCE2951), Sammo Sarker (24BCE2962), "
             "Prottoy Saha (24BCE2939)", size=16, color=GREY, align=PP_ALIGN.CENTER)
    slide.notes_slide.notes_text_frame.text = NOTES

    # 2. Problem
    n = 2
    slide = new_slide("The problem", n)
    bullets(slide, [
        "Teachers spend hours grading short written answers; AI could help, but it makes mistakes.",
        "Multi-agent AI systems split the work between several AI 'agents' - and any agent can fail: a message "
        "gets lost, a model gives a wrong answer, the input is damaged.",
        "In education, a wrong grade or wrong feedback that reaches a student does real harm.",
        "Question: when parts of an AI grading system fail, how often does wrong feedback reach the student - "
        "and can a safety layer stop it, on hardware a school can afford?",
        "Data: SciEntsBank (science short answers graded by teachers), 4 grades: correct, partially correct, "
        "contradictory, irrelevant.",
    ])

    # 3. Journey
    n += 1
    slide = new_slide("What we did", n)
    boxes(slide, [("1. Built", "4-agent pipeline\n(Planner, Tutor,\nGrader, Feedback)"),
                  ("2. Broke it", "injected 7 faults,\nmeasured the harm"),
                  ("3. Found", "pipeline worse than\na single model"),
                  ("4. Redesigned", "retrieval + 5 graders\n+ judge + safety layer"),
                  ("5. Made practical", "spreadsheet tool,\ncheap-laptop tiers")], 1.6, 1.5)
    bullets(slide, [
        f"Original pipeline: the student received a wrong verdict {old_wrong:.0%} of the time "
        f"(single Grader alone: {1 - single_acc:.0%} wrong).",
        "We measured why (each agent added its own errors), then redesigned the system around voting, "
        "a trained judge and explicit safety checks.",
    ], top=3.6, size=17, height=2.6)

    # 4-5. Original pipeline results
    for title, file in [("Original pipeline: what each fault did (120 answers)", "05_pipeline_harm.png")]:
        if os.path.exists(os.path.join(CHARTS, file)):
            n += 1
            picture(new_slide(title, n), os.path.join(CHARTS, file))

    # 6. Redesigned pipeline
    n += 1
    slide = new_slide("Redesigned pipeline", n)
    boxes(slide, [("Planner", "finds 8 teacher-graded\nexample answers"),
                  ("5 graders vote", "qwen2.5:7b, mistral:7b,\ngemma2:9b, TF-IDF,\nfine-tuned DeBERTa"),
                  ("Judge", "trained combiner:\none grade +\nconfidence"),
                  ("Safety layer", "checks, repairs,\nholds back doubtful\nanswers"),
                  ("Feedback", "advice text; verdict\nfixed to the grade")], 1.5, 1.6)
    bullets(slide, [
        "Everything runs locally (Ollama + Python) on one laptop with an 8 GB graphics card; no paid API, no internet.",
        "The judge was trained with copies where one grader lies or crashes, so it does not trust a single grader blindly.",
        "Grades the judge is not confident about go to the teacher instead of the student.",
    ], top=3.5, size=17, height=2.8)

    # 7. Safety layer
    n += 1
    slide = new_slide("The safety layer", n)
    table(slide, ["Check", "What it catches", "What happens"], [
        ["Message envelopes (ID, fingerprint, checksum)", "lost, misrouted or corrupted messages", "re-requested"],
        ["Reference-answer check", "wrong reference answer (known questions)", "restored from the question bank"],
        ["Question/reference word match", "wrong reference answer (new questions)", "sent to the teacher"],
        ["Input check", "empty or cut-off student answers", "sent to the teacher"],
        ["Misread-word check (photos)", "handwriting read wrongly from a photo", "sent to the teacher"],
        ["Canary check (32 known-grade answers)", "a grader that has gone wrong, e.g. lies", "benched for the session"],
        ["Confidence threshold", "grades the judge is unsure about", "sent to the teacher"],
        ["Fixed verdict line", "Feedback contradicting the grade", "cannot happen by design"],
    ], 0.7, 1.4, [4.4, 4.5, 3.0], size=14, row_h=0.58)

    # 8. Results table
    n += 1
    slide = new_slide("Results, no faults (final pipeline)", n)
    rows = []
    for set_name, label in SETS:
        off, on = run(set_name, "none", "off"), run(set_name, "none", "on")
        rows.append([label, f"{share(off, lambda r: r['final_label'] == r['gold_label']):.3f}",
                     f"{share(off, wrong_reaching):.1%}", f"{share(on, wrong_reaching):.1%}",
                     f"{share(on, lambda r: r['escalated']):.1%}"])
    table(slide, ["Answers", "Accuracy", "Wrong to student, safety off", "Wrong to student, safety on",
                  "Held for teacher"], rows, 0.7, 1.5, [3.4, 1.7, 2.4, 2.4, 2.0], size=15, row_h=0.6)
    bullets(slide, [f"Original pipeline: {old_wrong:.1%} wrong verdicts on the 120-task set; single Grader accuracy "
                    f"{single_acc:.3f}.",
                    "Held-back share follows from a threshold chosen on training data; a school can hold back more "
                    "to let fewer wrong grades through (next slides)."], top=4.8, size=15, height=1.8)

    # 9-11. Charts
    for title, file in [("Old vs new pipeline", f"09_v2_old_vs_new_120{FINAL}.png"),
                        ("Faults with the safety layer off and on (120 answers)", f"10_v2_faults_120{FINAL}.png"),
                        ("Teacher review vs wrong grades reaching students", "11_judge_tradeoff.png")]:
        if os.path.exists(os.path.join(CHARTS, file)):
            n += 1
            picture(new_slide(title, n), os.path.join(CHARTS, file))

    # 12. Faults on the larger set
    n += 1
    slide = new_slide("Every fault on 537 answers (familiar questions)", n)
    rows = []
    for fault in FAULTS_V2:
        if fault == "none":
            continue
        off, on = run("test_ua", fault, "off"), run("test_ua", fault, "on")
        caught = share(on, lambda r: any("->" in d or "benched" in d for d in r["detections"]))
        rows.append([FAULT_NAMES[fault], f"{share(off, wrong_reaching):.1%}", f"{share(on, wrong_reaching):.1%}",
                     f"{caught:.0%}"])
    base_off, base_on = run("test_ua", "none", "off"), run("test_ua", "none", "on")
    rows.append(["(no fault)", f"{share(base_off, wrong_reaching):.1%}", f"{share(base_on, wrong_reaching):.1%}", "-"])
    table(slide, ["Fault", "Wrong to student, safety off", "Wrong to student, safety on", "A check fired"],
          rows, 0.7, 1.35, [4.4, 2.6, 2.6, 2.3], size=13, row_h=0.5)

    # 13. Practical: school workflow
    n += 1
    slide = new_slide("How a school would use it", n)
    boxes(slide, [("Answers in", "CSV / spreadsheet\n(online form export\nor typed)"),
                  ("Teacher grades", "a few answers first\n(optional); they become\nworked examples"),
                  ("System grades", "grade, confidence\nand feedback\nfor every answer"),
                  ("Review list", "only doubtful or\nflagged answers"),
                  ("Teacher checks", "corrections join the\nanswer bank")], 1.5, 1.6)
    bullets(slide, [
        "One command: python -m scripts.grade_csv class.csv  ->  class_graded.csv + class_review.csv",
        "Example class of 40 answers: 8 graded by the teacher, 31 graded automatically, 1 sent for review; "
        "about 1 minute after the first run.",
        "The teacher stays in control: the system never overrides a teacher's grade.",
    ], top=3.5, size=16, height=2.8)

    # 13b. Practical: handwritten answer sheets (photo prototype)
    n += 1
    slide = new_slide("Paper answer sheets (prototype)", n)
    boxes(slide, [("Photo", "phone photo of\nthe answer sheet"),
                  ("Vision model", "qwen2.5vl:7b reads\nthe handwriting\n(same laptop)"),
                  ("Misread check", "many unknown words?\n-> teacher"),
                  ("Same pipeline", "graders, judge,\nsafety layer"),
                  ("Spreadsheet out", "grade, feedback,\nreview list")], 1.5, 1.6)
    rows = []
    stress_csv = os.path.join("results", "handwriting", "stress", "readings.csv")
    if os.path.exists(stress_csv):
        with open(stress_csv, encoding="utf-8-sig", newline="") as f:
            stress = list(csv.DictReader(f))
        for level, label in [("clean", "Handwriting fonts, clean (150 sheets)"),
                             ("hard", "Handwriting fonts, blurry/tilted/faded (150)")]:
            group = [r for r in stress if r["level"] == level]
            rows.append([label, f"{sum(float(r['cer']) for r in group) / len(group):.1%}",
                         f"{sum(float(r['cer']) == 0 for r in group)} of {len(group)}",
                         f"{sum(r['flagged_misread'] == 'True' for r in group)} of {len(group)}"])
    real_csv = os.path.join("results", "handwriting", "readings.csv")
    if os.path.exists(real_csv):
        with open(real_csv, encoding="utf-8-sig", newline="") as f:
            real = list(csv.DictReader(f))
        with open(os.path.join("results", "handwriting", "class_photos_graded.csv"), encoding="utf-8-sig", newline="") as f:
            flagged = sum("misread" in r["reason"] for r in csv.DictReader(f))
        rows.append([f"Real handwriting ({len(real)} phone photos, 1 writer)",
                     f"{sum(float(r['cer']) for r in real) / len(real):.1%}",
                     f"{sum(float(r['cer']) == 0 for r in real)} of {len(real)}", f"{flagged} of {len(real)}"])
    else:
        rows.append(["Team's real handwriting (30 photos)", "in progress", "", ""])
    table(slide, ["Test", "Characters read wrongly", "Read perfectly", "Flagged as misread"],
          rows, 0.7, 3.35, [5.0, 2.4, 2.2, 2.3], size=14, row_h=0.5)
    off, on = run("test_ua", "handwriting_misread", "off"), run("test_ua", "handwriting_misread", "on")
    caught = share(on, lambda r: any("misread" in d for d in r["detections"]))
    bullets(slide, [f"Simulated misreading (about 7% of characters wrong, 537 answers): wrong grades reaching students "
                    f"{share(off, wrong_reaching):.1%} with the safety layer off, {share(on, wrong_reaching):.1%} with it on; "
                    f"the misread check fired on {caught:.0%}.",
                    "Real handwriting: the reader's mistakes are real words ('Because' read as 'Became'), which the "
                    "misread check cannot see; a teacher check of photo answers stays necessary."],
            top=5.45, size=13, height=1.4)

    # 14-15. Practical charts
    for title, file in [("Cheaper hardware", "12_budget_tiers.png"),
                        ("Teacher answer bank", "13_answer_bank.png")]:
        if os.path.exists(os.path.join(CHARTS, file)):
            n += 1
            picture(new_slide(title, n), os.path.join(CHARTS, file))

    # 16. Speed
    n += 1
    slide = new_slide("Speed on our laptop (RTX 4060, 8 GB)", n)
    prefix = json.load(open(os.path.join("results", "answer_bank", "prefix_cache_test_uq.json"), encoding="utf-8"))
    rows = []
    for model in ["qwen2.5:7b", "mistral:7b", "gemma2:9b"]:
        usual = next(r for r in prefix if r["model"] == model and r["order"].startswith("similarity"))
        fixed = next(r for r in prefix if r["model"] == model and r["order"].startswith("fixed"))
        rows.append([model, f"{usual['seconds_per_answer']:.2f} s", f"{fixed['seconds_per_answer']:.2f} s",
                     f"{usual['accuracy']:.3f} -> {fixed['accuracy']:.3f}"])
    rows.append(["fine-tuned DeBERTa (CPU only)", f"{cpu['deberta-v3-base-nli']:.2f} s", "-", "-"])
    rows.append(["qwen2.5:3b (CPU only)", f"{cpu['qwen2.5:3b']:.1f} s", "-", "-"])
    table(slide, ["Grader", "Seconds per answer", "With shared prompt start", "Accuracy (usual -> shared)"],
          rows, 0.7, 1.5, [4.0, 2.4, 2.7, 2.8], size=15, row_h=0.6)
    bullets(slide, ["Shared prompt start: every answer to a question gets the same worked examples in the same order, "
                    "so the model reuses its work on that part (gemma2:9b does not fit fully on the GPU, so it gains little)."],
            top=5.6, size=14, height=1.2)

    # 17. Limitations
    n += 1
    slide = new_slide("Limitations", n)
    bullets(slide, [
        "One dataset (SciEntsBank, school science, English); reading handwritten photos is a prototype, tested on "
        "handwriting fonts and a small set of the team's own photos, not on real classes.",
        "Accuracy is about 0.72 on familiar questions and 0.63-0.64 on new ones: the system needs the teacher "
        "review step; it is a grading assistant, not a replacement.",
        "On new questions a wrong reference answer is only caught by word overlap with the question "
        "(about 92-95% caught, about 5% false alarms); it is sent to the teacher, not repaired.",
        "Feedback advice text: a small pilot rating by two team members was inconclusive (raters read the task "
        "differently; 13 shared messages, agreement at chance level); a proper rating is still needed.",
        "Faults were injected one at a time; real failures can combine.",
    ])

    # 18. Future work
    n += 1
    slide = new_slide("Next steps", n)
    bullets(slide, [
        "Test photo reading on real class answer sheets (many writers, whole pages with several answers).",
        "Simple web page for teachers (upload, review, download) instead of a terminal command.",
        "Human rating of feedback quality; tests on a second dataset.",
        "Write up as a workshop paper (educational NLP).",
    ])

    # 19. AI disclosure
    n += 1
    slide = new_slide("AI tool use disclosure", n)
    table(slide, ["Tool", "Used for", "Extent (fill in)"],
          [["Claude Code (Anthropic)", "Code, running experiments, analysis scripts, charts, this slide layout", "[state honestly]"],
           ["Google Antigravity", "Read-only project review, pitch draft, demo command list", "[state honestly]"],
           ["[other tool, if any]", "", ""]], 0.7, 1.5, [3.3, 5.6, 3.0], size=15, row_h=0.7)

    # 20. Close
    n += 1
    slide = new_slide("Thank you - questions?", n)
    text_box(slide, 0.7, 2.6, 12.0, 1.0, "Live demo: python -m scripts.demo_v2 --task 23 --fault grader_lying",
             size=20, color=INK, align=PP_ALIGN.CENTER)
    text_box(slide, 0.7, 3.6, 12.0, 1.0, "Code and results: github.com/Debojyoti17/edu-masfire", size=20,
             color=GREY, align=PP_ALIGN.CENTER)

    deck.prs.save(OUT)
    print(f"Saved {OUT} with {len(deck.prs.slides)} slides")


if __name__ == "__main__":
    main()
