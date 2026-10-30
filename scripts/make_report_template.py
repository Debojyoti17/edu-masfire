# Word file for the team's final report: the VIT logo, the nine sections mam asked for, and the hard
# parts already in place - every result table and chart (numbers read from the result files), the
# architecture diagram, the code repository section and the AI-use facts. The report text itself is
# NOT here: mam's rule is that the report must be the students' own writing, so each place the team
# writes is marked by a yellow box that says what to cover.
#
# A .docx file is a zip of XML files, so this builds one with the standard library only.
# Writes results/report/Team2_Final_Report_DRAFT.docx.
# Usage: python -m scripts.make_report_template
import json
import os
import re
import zipfile
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

import scripts.make_report_pack as pack
from scripts.analyze_photo_checks import load
from scripts.check_feedback import RUN, praises
from scripts.run_pipeline import load_jsonl
from scripts.run_pipeline_v2 import FAULTS_V2

OUT = os.path.join("results", "report", "Team2_Final_Report_DRAFT.docx")
CHARTS = os.path.join("results", "charts")
LOGO = os.path.join("results", "slides", "vit_logo.png")
ARCHITECTURE = os.path.join(CHARTS, "00_architecture.png")
PAGE_WIDTH = 9638  # usable width of an A4 page in Word's units (twips)
FINAL = "_robust_full_refsim_deberta-v3-base-nli"
INK, NAVY, BLUE = "111111", "1F3864", "2F5496"

images = []  # (name inside the docx, source file) of every picture used
counters = {"table": 0, "figure": 0}


def run(text, size=22, colour=INK, bold=False, italic=False, yellow=False):
    """One piece of text with one look."""
    props = '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>'
    props += ("<w:b/>" if bold else "") + ("<w:i/>" if italic else "")
    props += f'<w:color w:val="{colour}"/><w:sz w:val="{size}"/>' + ('<w:highlight w:val="yellow"/>' if yellow else "")
    return f'<w:r><w:rPr>{props}</w:rPr><w:t xml:space="preserve">{escape(str(text))}</w:t></w:r>'


def para(content, before=40, after=80, centre=False, style=None, page_break=False, keep=False):
    """One paragraph holding ready-made runs."""
    props = f'<w:pStyle w:val="{style}"/>' if style else ""
    props += "<w:keepNext/>" if keep else ""
    props += "<w:pageBreakBefore/>" if page_break else ""
    props += f'<w:spacing w:before="{before}" w:after="{after}"/>' + ('<w:jc w:val="center"/>' if centre else "")
    return f"<w:p><w:pPr>{props}</w:pPr>{content}</w:p>"


def heading(text, level=1, page_break=False):
    """A numbered section heading that Word lists in its navigation pane."""
    return para(run(text, 30 if level == 1 else 25, NAVY if level == 1 else BLUE, bold=True),
                before=240 if level == 1 else 180, style=f"Heading{level}", page_break=page_break, keep=True)


def write_here(*points):
    """A yellow box: what the team should write here, in its own words."""
    out = para(run("[YOU WRITE THIS PART - delete this yellow text when done]", 20, bold=True, yellow=True), before=100, after=20)
    return out + "".join(para(run("- " + p, 20, yellow=True), before=0, after=10) for p in points) + para("", before=0, after=60)


def note(text):
    """A short plain note under a table or figure."""
    return para(run(text, 18, "555555", italic=True), before=20, after=120)


def caption(kind, text):
    """Number a table or figure in order of appearance. Returns its caption paragraph."""
    counters[kind] += 1
    return para(run(f"{kind.title()} {counters[kind]}. {text}", 20, NAVY, bold=True), before=160, after=60, keep=True)


def table(headers, rows, widths=None, size=17):
    """A table with a shaded header row; widths are shares of the page width."""
    n = len(headers)
    shares = widths or ([2] + [1] * (n - 1))
    cols = [int(PAGE_WIDTH * s / sum(shares)) for s in shares]
    borders = "".join(f'<w:{s} w:val="single" w:sz="4" w:color="999999"/>' for s in ["top", "left", "bottom", "right", "insideH", "insideV"])
    out = [f'<w:tbl><w:tblPr><w:tblW w:w="{sum(cols)}" w:type="dxa"/><w:tblBorders>{borders}</w:tblBorders><w:tblLayout w:type="fixed"/>'
           f'</w:tblPr><w:tblGrid>{"".join(f"<w:gridCol w:w=\"{c}\"/>" for c in cols)}</w:tblGrid>']
    for i, row in enumerate([headers] + [list(r) for r in rows]):
        shade = '<w:shd w:val="clear" w:color="auto" w:fill="D9E2F3"/>' if i == 0 else ""
        cells = "".join(f'<w:tc><w:tcPr><w:tcW w:w="{c}" w:type="dxa"/>{shade}</w:tcPr>'
                        f'{para(run(text, size, bold=i == 0), before=10, after=10)}</w:tc>' for c, text in zip(cols, row))
        header = "<w:trPr><w:tblHeader/></w:trPr>" if i == 0 else ""
        out.append(f"<w:tr>{header}{cells}</w:tr>")
    return "".join(out) + "</w:tbl>"


def picture(path, inches=6.5):
    """A picture placed in its own centred paragraph, scaled to the given width."""
    images.append((f"image{len(images) + 1}{os.path.splitext(path)[1]}", path))
    number = len(images)
    w, h = Image.open(path).size
    cx, cy = int(inches * 914400), int(inches * 914400 * h / w)
    drawing = (f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
               f'<wp:docPr id="{number}" name="Picture {number}"/>'
               '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
               '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
               '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
               f'<pic:nvPicPr><pic:cNvPr id="{number}" name="Picture {number}"/><pic:cNvPicPr/></pic:nvPicPr>'
               f'<pic:blipFill><a:blip r:embed="rIdImg{number}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
               f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
               '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
               '</a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')
    return para(drawing, centre=True)


def figure(file, text, inches=6.5):
    """A chart from results/charts with a numbered caption."""
    return caption("figure", text) + picture(os.path.join(CHARTS, file), inches)


def pack_table(start, widths=None, size=17):
    """The report-pack table whose caption starts with `start`, with a new number and its source."""
    title, headers, rows, source = next(t for t in pack.tables if t[0].startswith(start))
    return (caption("table", re.sub(r"^Table \w+\.\s*", "", title)) + table(headers, rows, widths, size) + note("Source: " + source))


def draw_architecture():
    """Draw the system's architecture diagram and save it with the charts."""
    fig, ax = plt.subplots(figsize=(14.5, 5.4))
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 44)
    ax.axis("off")
    boxes = [(1, 24, 15, "Student answer", "spreadsheet row, or a\nphoto read by a local\nvision model"),
             (19, 24, 15, "Input checks", "empty / cut-off answer,\nreference answer,\nmisread words"),
             (37, 24, 15, "Planner", "finds 8 teacher-graded\nanswers as worked\nexamples"),
             (55, 24, 16, "Five graders", "3 local LLMs, TF-IDF,\nfine-tuned DeBERTa"),
             (74, 24, 15, "Judge", "combines the votes into\none grade + confidence"),
             (92, 24, 17, "Feedback", "verdict line fixed to\nthe grade; advice\nwritten by an LLM")]
    for x, y, w, title, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, 16, boxstyle="round,pad=0.3,rounding_size=1.2", fc="#D9E2F3", ec="#1F3864"))
        ax.text(x + w / 2, y + 12.6, title, ha="center", va="center", fontsize=11.5, fontweight="bold", color="#1F3864")
        ax.text(x + w / 2, y + 5.8, text, ha="center", va="center", fontsize=9.2)
    for (x, _, w, *_), (nx, *_) in zip(boxes, boxes[1:]):
        ax.annotate("", xy=(nx - 0.5, 32), xytext=(x + w + 0.5, 32), arrowprops=dict(arrowstyle="-|>", color="#444444"))
    layer = ("Safety layer around every step: each message carries an ID, a question fingerprint and a checksum, so lost, "
             "misrouted or damaged messages are re-requested;\na canary check before the session benches any grader that fails answers with known grades")
    ax.add_patch(FancyBboxPatch((1, 1), 108, 6.5, boxstyle="round,pad=0.2,rounding_size=1", fc="#E2EFDA", ec="#548235"))
    ax.text(55, 4.25, layer, ha="center", va="center", fontsize=9.5)
    ax.add_patch(FancyBboxPatch((58, 11), 37, 7.5, boxstyle="round,pad=0.2,rounding_size=1", fc="#FFF2CC", ec="#BF9000"))
    ax.text(76.5, 14.75, "Low confidence or a failed check:\nanswer goes to the teacher, not the student", ha="center", va="center", fontsize=10)
    ax.annotate("", xy=(81, 18.9), xytext=(81, 23.6), arrowprops=dict(arrowstyle="-|>", color="#BF9000", lw=1.6))
    fig.savefig(ARCHITECTURE, dpi=170, bbox_inches="tight")
    plt.close(fig)


def headline_rows():
    """Original pipeline vs final system: the numbers for the measurable objectives."""
    tasks_old = load_jsonl(os.path.join("results", "pipeline", "qwen2.5-7b", "none.jsonl"))
    wrong_old = pack.share(tasks_old, lambda r: r["delivered"] != r["gold_label"])
    on = pack.v2("120", "none", "on")
    return [["Wrong verdict reaching the student (120 tasks, no fault)", f"{wrong_old:.1%}", f"{pack.share(on, pack.wrong_reaching):.1%}"],
            ["Same, with the least confident 25% sent to the teacher", "not available", f"{pack.wrong_after_holdback(on):.1%}"],
            ["Grade accuracy (120 tasks)", f"{1 - wrong_old:.3f} (one grader alone: 0.492)", f"{pack.share(on, lambda r: r['final_label'] == r['gold_label']):.3f}"],
            ["Runs on one laptop, no paid API or internet", "yes", "yes"]]


def praise_rows():
    """How often the feedback of the final run praises the answer, per verdict."""
    recs = [r for r in load_jsonl(RUN) if r.get("feedback")]
    names = {"contradictory": "contradictory", "irrelevant": "irrelevant",
             "partially_correct_incomplete": "partially correct", "correct": "correct"}
    rows = []
    for label, name in names.items():
        group = [r for r in recs if r["label_sent"] == label]
        hits = sum(praises(r["feedback"]) for r in group)
        rows.append([name, len(group), f"{hits} ({hits / len(group):.0%})"])
    return rows


def photo_rows():
    """Reading errors on the real handwriting photos, by writer and by how the photo was taken."""
    rows = load("readings.csv")
    out = []
    for name, keep in [("All photos", lambda r: True), ("Writer A", lambda r: r["writer"] == "Sammo"), ("Writer B", lambda r: r["writer"] == "Prottoy"),
                       ("Photo taken upright", lambda r: r["taken"] == "upright"), ("Photo taken sideways", lambda r: r["taken"] == "sideways")]:
        group = [r for r in rows if keep(r)]
        out.append([name, len(group), f"{sum(r['cer'] for r in group) / len(group):.1%}", sum(r["cer"] == 0 for r in group)])
    return out


def file_lines(path):
    """The lines of a text file, without blanks."""
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def body():
    """Everything in the report, in mam's order."""
    d = []
    # Cover page
    d += [para("", before=500), picture(LOGO, 2.6),
          para(run("Fault-Tolerant Multi-Agent Grading of Short Student Answers", 40, NAVY, bold=True), before=500, after=120, centre=True),
          para(run("Final Project Report", 28, BLUE), centre=True, after=480),
          table(["Name", "Registration number"], [["Debojyoti Barai Mitul", "24BCE2951"], ["Sammo Sarker", "24BCE2962"], ["Prottoy Saha", "24BCE2939"]], [1, 1], 22),
          para(run("Team 2", 24, bold=True), before=360, centre=True),
          para(run("[Course name and code]", 22, yellow=True), centre=True), para(run("[Faculty name]", 22, yellow=True), centre=True),
          para(run("[School, Vellore Institute of Technology]   October 2026", 22, yellow=True), centre=True)]

    # How to use
    d += [heading("How to use this file (delete this page)", page_break=True),
          para(run("Mam's rule: \"AI-generated content submitted as the team's original academic work is strictly prohibited. The report must "
                   "reflect the students' own understanding, analysis, implementation, and writing.\" So the text of this report is yours to write.")),
          para(run("What is already here: the nine headings from mam's list, all tables and charts, the architecture diagram, and the code repository section. "
                   "Every number was read from the project's result files. Where you see yellow, write in your own words what it asks for, then delete the yellow text.")),
          para(run("These tables, charts and the diagram were produced with AI assistance, and Section 9 says so. Keep that disclosure accurate. "
                   "Update the table and figure numbers you refer to if you remove any.")),
          para(run("Careful with one claim: see the note under the judge comparison in Section 7. The improvement from the fine-tuned grader is only clearly "
                   "beyond chance on one test set.", bold=True))]

    # 1
    d += [heading("1. Problem Statement", page_break=True),
          write_here("What problem are you solving? (grading short written answers takes teachers a long time; AI can help but makes mistakes.)",
                     "Why does it matter? What happens to a student who gets a wrong grade or wrong feedback?",
                     "Why is a multi-agent AI system a special risk here? (one agent's mistake can pass to the next.)",
                     "Why must it run on a cheap laptop without paid services or internet? (schools, cost, student privacy.)",
                     "State the research question in one sentence: when parts of the system fail, how often does wrong feedback reach the student, and can it be stopped?")]
    # 2
    d += [heading("2. Objectives"),
          write_here("List 4-6 specific, measurable objectives in your own words. Use the table below to show each was measured.",
                     "Ideas: build the pipeline; inject faults and measure the harm; redesign it; add a safety layer; keep it on one laptop; test it on handwriting."),
          caption("table", "Main measures, original pipeline and final system"),
          table(["Measure", "Original pipeline", "Final system"], headline_rows(), [3, 1.4, 1])]
    # 3
    d += [heading("3. Literature Review"),
          write_here("Critically review at least 25-30 research articles (mam's requirement). Reuse your earlier literature review if you have one.",
                     "Group them by theme: automatic short answer grading; LLMs for grading; multi-agent systems and their reliability; fault injection; confidence and uncertainty; handwriting recognition in education.",
                     "For each theme: what has been done, what works, what is missing. End with the research gap your project fills.",
                     "Fill in the table below as you read. Only cite papers you have actually read."),
          caption("table", "Literature review summary (add your papers)"),
          table(["No.", "Paper (author, year)", "Method", "Dataset", "Main finding", "Limitation / gap"],
                [[1, "Dzikovska et al. (2013): SemEval-2013 Task 7", "", "", "", ""], [2, "Jia et al. (2026): MAS-FIRE", "", "", "", ""], [3, "He et al. (2021): DeBERTaV3", "", "", "", ""]]
                + [[i, "", "", "", "", ""] for i in range(4, 31)], [0.5, 2.6, 1.3, 1.1, 1.8, 1.8])]
    # 4
    d += [heading("4. Limitations of Existing Systems"),
          write_here("Discuss the key problems of existing approaches, citing the papers from Section 3.",
                     "Points to consider: small models are weak alone (see Table below); a model's own confidence is a poor guide to whether it is right; pipelines pass errors along; most work tests accuracy, not what happens when parts fail; many systems need paid cloud models; handwriting is rarely handled."),
          caption("table", "What we measured about single models (our own early experiments)"),
          table(["Finding", "Measured result"],
                [["Best single LLM grader on the 120-task set", "0.492 accuracy (qwen2.5:7b, rubric prompt)"],
                 ["Four ways of getting a confidence number from one model", "AUROC 0.547 to 0.605 (0.5 = chance): not usable as a safety check"],
                 ["Original 4-agent pipeline vs one grader alone", "Pipeline was worse than the single grader (see Section 7)"]], [2, 3])]
    # 5
    faults = [[k, v] for k, v in FAULTS_V2.items() if k != "none"]
    d += [heading("5. Proposed Methodology", page_break=True),
          write_here("Explain the design in your own words: first the original pipeline and why it failed, then the redesign and why each part is there.",
                     "Walk through the diagram below from left to right, as you did in the demo."),
          heading("5.1 System architecture", 2), caption("figure", "Architecture of the final system"), picture(ARCHITECTURE),
          heading("5.2 Components", 2), caption("table", "Components of the final system"),
          table(["Component", "What it does", "Details"],
                [["Planner", "Finds worked examples", "8 teacher-graded answers to the same question; for new questions, similar answers plus a no-example grader"],
                 ["LLM graders (3)", "Each gives a probability for every grade", "qwen2.5:7b, mistral:7b, gemma2:9b through Ollama; temperature 0, seed 42, 4096-token context"],
                 ["TF-IDF classifier", "Non-LLM grader", "Logistic regression on word features of the train split"],
                 ["Fine-tuned DeBERTa", "Small trained grader", "DeBERTa-v3-base NLI model, 3 epochs, learning rate 2e-5, 4,946 train answers"],
                 ["Judge", "One grade and a confidence", "Logistic regression on all graders' probabilities; separate judges for seen and new questions; trained with fault-injected copies"],
                 ["Safety layer", "Detects and contains faults", "See the safety checks table"],
                 ["Feedback", "Message to the student", "Verdict line fixed to the judge's grade; advice written by qwen2.5:7b"],
                 ["Photo reader", "Handwriting to text", "qwen2.5vl:7b vision model on the same laptop (prototype)"]], [1.3, 1.6, 3.4]),
          heading("5.3 Fault injection", 2), caption("table", "Faults injected, one at a time"), table(["Fault", "What happens"], faults, [1, 3]),
          heading("5.4 Safety layer", 2), caption("table", "Safety checks"),
          table(["Check", "What it catches", "What happens"],
                [["Message envelopes (ID, fingerprint, checksum)", "Lost, misrouted or corrupted messages", "Re-requested"],
                 ["Reference-answer check", "Wrong reference answer (known questions)", "Restored from the question bank"],
                 ["Question/reference word match", "Wrong reference answer (new questions)", "Sent to the teacher"],
                 ["Input check", "Empty or cut-off answers", "Sent to the teacher"],
                 ["Misread-word check (photos)", "Handwriting read wrongly", "Sent to the teacher"],
                 ["Canary check (32 known-grade answers)", "A grader that has gone wrong", "Benched for the session"],
                 ["Confidence threshold", "Grades the judge is unsure of", "Sent to the teacher"],
                 ["Fixed verdict line", "Feedback contradicting the grade", "Cannot happen by design"]], [2, 2, 1.4]),
          write_here("Explain how the judge was trained and how the confidence threshold was chosen (the report pack's Methods summary has the facts).",
                     "Explain the metrics in your own words: accuracy; wrong feedback reaching the student; share held back for a teacher.")]
    # 6
    d += [heading("6. Code Repository", page_break=True),
          para(run("Source code: https://github.com/Debojyoti17/edu-masfire", bold=True)),
          para(run("Full source code with execution notes (HOW_TO_RUN.md) was also uploaded to the course Google Drive folder, in the folder for registration numbers 24BCE2951, 24BCE2962 and 24BCE2939.")),
          caption("table", "Repository contents"),
          table(["Path", "Contents"],
                [["src/", "The system: Planner, graders, judge, safety layer, faults, photo reader"],
                 ["scripts/", "Experiment runners, analysis, charts, demo (demo_v2.py), teacher tools (grade_csv.py, grade_photo.py)"],
                 ["tests/", "Automatic tests of the safety layer and fault injection"],
                 ["data/", "120-task evaluation set and larger test sets"],
                 ["results/", "Result files, summaries, charts"],
                 ["README.md, HOW_TO_RUN.md, DEMO_GUIDE.md", "Setup, execution notes and demo steps"]], [1.3, 3])]
    # 7
    d += [heading("7. Experimental Analysis and Current Progress", page_break=True),
          write_here("Introduce the experiments: the data (SciEntsBank), the four test sets, and the hardware (one laptop, RTX 4060 8 GB).",
                     "For each table and figure below, write 2-4 sentences: what it shows, and why. Do not just repeat the numbers."),
          heading("7.1 Original pipeline", 2), pack_table("Table 1."), figure("05_pipeline_harm.png", "Harm done by each fault in the original pipeline"),
          write_here("Why was the 4-agent pipeline worse than one grader alone? What did that teach you?"),
          heading("7.2 Final pipeline with no faults", 2), pack_table("Table 2.", [2.2, 0.7, 1.4, 1.5, 1.5, 0.9]),
          figure(f"09_v2_old_vs_new_120{FINAL}.png", "Original vs redesigned pipeline"),
          write_here("Compare the sets. Why does the safety layer hold back far more answers on new questions than on familiar ones?"),
          heading("7.3 Faults with the safety layer off and on", 2), pack_table("Table 3a."), pack_table("Table 3b."),
          figure(f"10_v2_faults_120{FINAL}.png", "Faults with the safety layer off and on (120 answers)"),
          write_here("Which faults does the safety layer repair, and which does it only hold back for the teacher? Which fault is hardest?"),
          heading("7.4 Teacher review trade-off and the fine-tuned grader", 2), pack_table("Table 4.", [1.8, 1.3, 1.3, 1.8]),
          note("Care with this claim: the new judge lowered wrong grades by 1 to 3 points on every set, but the paired difference is only clear on test_ud. "
               "On the other sets it is within the noise, so do not state it as proven there."),
          figure("11_judge_tradeoff.png", "Teacher review vs wrong grades reaching students"),
          write_here("Explain the trade-off: more teacher review means fewer wrong grades. How should a school choose?"),
          heading("7.5 Cheaper hardware", 2), pack_table("Table 5."), figure("12_budget_tiers.png", "Accuracy by hardware tier"), pack_table("Table 7.", [3, 1.4]),
          write_here("What does a school lose by using a cheaper laptop, or no graphics card at all?"),
          heading("7.6 Teacher answer bank", 2), pack_table("Table 8."), figure("13_answer_bank.png", "Teacher answer bank simulation"),
          write_here("Accuracy rises, but teacher workload at the same safety level does not fall. Why?"),
          heading("7.7 Handwritten answer photos (prototype)", 2), pack_table("Table 9."),
          caption("table", "Reading errors on real handwriting, by writer and photo direction"),
          table(["Photos", "Number", "Characters read wrongly", "Read perfectly"], photo_rows(), [2, 0.8, 1.4, 1]),
          note("From the 28 phone photos: " + "; ".join(l.lstrip("- ") for l in file_lines(os.path.join("results", "handwriting", "summary.md"))[2:3]) + "."),
          write_here("How did you test handwriting? Say honestly that fonts are much easier than real writing, and that this is only 28 photos by 2 writers.",
                     "Why does the misread-word check miss real mistakes? (they are real words, e.g. 'Because' read as 'Became')."),
          heading("7.8 Feedback quality", 2),
          caption("table", "Feedback containing praise, by verdict (120 tasks, final run)"),
          table(["Verdict", "Messages", "Messages containing praise"], praise_rows(), [2, 1, 2]),
          note("Praise is detected with a fixed list of praising phrases, a simple check. Praise for contradictory or irrelevant answers is the problem."),
          caption("table", "Pilot human rating of feedback"),
          table(["Measure", "Result"], [[l.lstrip("- ").split(":")[0], ":".join(l.split(":")[1:]).strip()] for l in file_lines(os.path.join("results", "human_rating", "summary.md"))[1:6] if ":" in l], [1.4, 3]),
          write_here("What do these two tables say about the feedback? Why was the rating only a pilot, and what would you change next time?"),
          heading("7.9 Current status and limitations", 2),
          write_here("Summarise what is finished and what is not.",
                     "Limitations to include: one dataset; accuracy needs teacher review; photo reading is a small prototype; feedback advice text is the weakest part; faults tested one at a time.",
                     "Work after these results: per-verdict feedback prompt, reading photos in 4 rotations, photo-specific checks. Only describe these as measured if the results exist when you submit.",
                     "Future work, in your own words.")]
    # 8
    d += [heading("8. References", page_break=True),
          write_here("Mam requires APA style and all your 25-30 sources. Add yours here in alphabetical order. The ones below are the dataset and methods this project directly used; check that each is still cited in your text."),
          para(run("Dzikovska, M., Nielsen, R., Brew, C., Leacock, C., Giampiccolo, D., Bentivogli, L., Clark, P., Dagan, I., & Dang, H. T. (2013). SemEval-2013 Task 7: The Joint Student Response Analysis and 8th Recognizing Textual Entailment Challenge. In Second Joint Conference on Lexical and Computational Semantics (*SEM), Volume 2 (pp. 263-274). Association for Computational Linguistics.")),
          para(run("He, P., Gao, J., & Chen, W. (2021). DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. arXiv. https://arxiv.org/abs/2111.09543")),
          para(run("Jia, J., Deng, Z., Chen, Z., Wang, Y., & Zheng, Z. (2026). MAS-FIRE: Fault injection and reliability evaluation for LLM-based multi-agent systems. arXiv. https://arxiv.org/abs/2602.19843")),
          para(run("Laurer, M., van Atteveldt, W., Casas, A., & Welbers, K. Less annotating, more classifying: Addressing the data scarcity issue of supervised machine learning with deep transfer learning and BERT-NLI. ", yellow=True) + run("[Check the journal, year, volume and pages before submitting]", 20, bold=True, yellow=True))]
    # 9
    d += [heading("9. Contribution Statement"),
          caption("table", "Individual contributions (fill in honestly)"),
          table(["Role", "Debojyoti Barai Mitul", "Sammo Sarker", "Prottoy Saha"],
                [[r, "", "", ""] for r in ["Literature review", "Methodology", "Implementation", "Experimentation", "Analysis", "Report preparation"]]
                + [["Handwriting test sheets", "", "Wrote and photographed H01-H15", "Wrote and photographed H16-H30"],
                   ["Feedback rating", "", "Rater A", "Rater B"]], [1.3, 1.3, 1.3, 1.3]),
          caption("table", "Use of AI tools (required by the course; correct anything that is not accurate)"),
          table(["Tool", "Purpose", "Extent"],
                [["Claude Code (Anthropic)", "Writing code, running the experiments, analysis scripts, charts, slides, tables and the architecture diagram in this report, demo guides", "Wrote most of the code and ran the experiments under the team's direction; produced this report's tables and figures. [State how much of this was yours.]"],
                 ["Google Antigravity", "Read-only review of the project, pitch draft, list of demo commands", "Review and drafts only; no project files changed"],
                 ["[Any other tool]", "", ""]], [1.2, 2.4, 2.4]),
          write_here("Add a sentence stating that the report text is the team's own writing, only if that is true.")]
    return "".join(d)


def main():
    """Build the Word file."""
    draw_architecture()
    pack.main(os.devnull)  # fills pack.tables; writes nothing that matters
    content = body()
    ns = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
          'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"')
    head = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    document = (f'{head}<w:document {ns}><w:body>{content}<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter"/>'
                '<w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="567" w:footer="567" w:gutter="0"/>'
                '</w:sectPr></w:body></w:document>')
    footer = (f'{head}<w:ftr {ns}><w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:fldChar w:fldCharType="begin"/></w:r>'
              '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')
    styles = (f'{head}<w:styles {ns}><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
              + "".join(f'<w:style w:type="paragraph" w:styleId="Heading{n}"><w:name w:val="heading {n}"/><w:basedOn w:val="Normal"/>'
                        f'<w:next w:val="Normal"/><w:qFormat/><w:pPr><w:outlineLvl w:val="{n - 1}"/></w:pPr></w:style>' for n in [1, 2])
              + '</w:styles>')
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    rels = (f'{head}<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rIdStyles" Type="{rel}/styles" Target="styles.xml"/>'
            f'<Relationship Id="rIdFooter" Type="{rel}/footer" Target="footer1.xml"/>'
            + "".join(f'<Relationship Id="rIdImg{i}" Type="{rel}/image" Target="media/{name}"/>' for i, (name, _) in enumerate(images, start=1))
            + '</Relationships>')
    root = (f'{head}<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{rel}/officeDocument" Target="word/document.xml"/></Relationships>')
    wml = "application/vnd.openxmlformats-officedocument.wordprocessingml"
    types = (f'{head}<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
             '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
             '<Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'
             f'<Override PartName="/word/document.xml" ContentType="{wml}.document.main+xml"/>'
             f'<Override PartName="/word/styles.xml" ContentType="{wml}.styles+xml"/>'
             f'<Override PartName="/word/footer1.xml" ContentType="{wml}.footer+xml"/></Types>')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", types)
        z.writestr("_rels/.rels", root)
        z.writestr("word/document.xml", document)
        z.writestr("word/styles.xml", styles)
        z.writestr("word/footer1.xml", footer)
        z.writestr("word/_rels/document.xml.rels", rels)
        for name, path in images:
            z.write(path, "word/media/" + name)
    print(f"Saved {OUT}: {counters['table']} tables, {counters['figure']} figures")


if __name__ == "__main__":
    main()
