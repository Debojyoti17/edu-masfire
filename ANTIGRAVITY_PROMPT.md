You are reviewing a finished university AI project in this folder before its FINAL REVIEW (a live
demo to my teacher, "mam", on Tuesday 3 Nov 2026). I am a student on the team and I am still learning
Python, so explain everything in simple words.

## Strict rules
1. Do NOT change, move, rename or delete any existing file. Never touch data/, results/, models/,
   src/, scripts/, tests/ or any file that already exists.
2. Do NOT install or update any library, and do NOT pull Ollama models.
3. Do NOT run any command without asking me first. Never run long experiments (anything that grades
   hundreds of answers or trains a model). The GPU is shared with another project.
4. You may only CREATE new files, and only inside a new folder named `antigravity_review/final/`.
5. Every number you write must come from a file in this project. Name the file it came from. If you
   cannot find a number, say so; never guess.
6. If code and documents disagree, believe the code and the result files, and tell me about the difference.

## Where to start reading
README.md, CLAUDE.md (start with "Current state" under Decisions), HANDOFF.md (latest status),
DEMO_GUIDE.md, DEMO_COMMANDS_v2.docx (the demo command sheet), then src/ and scripts/.
Results: results/report_pack.html (all tables), results/pipeline_v2_summary.md,
results/handwriting/summary.md and photo_checks.md, results/human_rating/summary.md,
results/feedback_check/summary.md (if present). Tests: `python -m unittest discover -s tests -t .`
(you may ask me to run it; it takes about 1 second and needs no GPU).

## What I need (write to antigravity_review/final/review.md)
1. **Correctness review of the code**: read src/pipeline_v2.py, src/health.py, src/judge.py,
   src/ocr.py, scripts/run_pipeline_v2.py, scripts/grade_csv.py, scripts/grade_photo.py and
   scripts/demo_v2.py. List real bugs or risky spots with file and line, and why. Do not list style
   preferences.
2. **Claims check**: for each main claim in the report pack and the demo sheet, say whether the
   result files support it, including whether the confidence intervals overlap.
3. **Demo risk check**: go through DEMO_COMMANDS_v2.docx step by step. For each command, check its
   options against the code and say what could go wrong live and how to recover.
4. **Hard questions mam might ask** (at least 12), each with a short honest answer based on the files.
5. **Top 5 things to fix or prepare before Tuesday**, most important first.

When you finish, give me a short summary in the chat.
