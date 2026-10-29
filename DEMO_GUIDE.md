# Demo guide (Team 2)

How the live demo works, how we built it, how the system would be used in a real school, and the
exact commands to run in front of mam. Numbers come from `results/report_pack.html`
(`python -m scripts.make_report_pack`).

---

## 1. What happens in a demo session

The demo runs the **real pipeline**, not a recording or a mock-up. `scripts/demo_v2.py` calls the
same functions the experiments used (`prepare` and `finish_task` in `scripts/run_pipeline_v2.py`)
on one student answer from our 120-task set, and prints what every stage did.

For one answer the screen shows:

1. **The question, the teacher's reference answer and the student's answer**, plus the teacher's
   real grade (the system never sees it; we show it so the audience can judge the result).
2. **Planner:** whether the system has teacher-graded examples for this question (then it shows the
   graders 8 graded answers to the same question) or not (then it uses similar questions).
3. **The graders vote:** qwen2.5:7b, mistral:7b and gemma2:9b (local language models), a TF-IDF
   classifier and our fine-tuned DeBERTa model. Each shows its grade and how sure it is.
4. **The judge** combines the five votes into one grade and a confidence. If the confidence is
   below the threshold, the answer is held back for the teacher.
5. **The safety layer** reports any problem it found and what it did about it.
6. **The feedback** the student would receive (2-3 sentences; its verdict line is fixed to the grade).

With `--fault <name>`, the same answer is then run twice more with that fault injected: once with
the safety layer **off** (what would happen without our protection) and once **on** (what our
system actually does). This is the core of the project in one screen: *a part breaks -> without
protection the student gets a wrong grade -> with protection the problem is caught.*

## 2. How we made the demo

- **Deterministic:** every model call uses temperature 0 and seed 42, so the same answer gives the
  same output every time. What you rehearse is what mam sees.
- **Fast:** every model answer is saved in a cache (`results/pipeline_v2/cache/`) keyed by its exact
  prompt, so a demo run takes about 15 seconds instead of minutes. Anything new (for example new
  feedback text) is generated live by the local models.
- **Good examples chosen from data, not by hand:** `--suggest` reads the saved experiment results
  and lists answers where a fault turned a right grade into a wrong one, and what the safety layer
  did with it (fixed / held for the teacher).
- **Faults are injected the same way as in the experiments** (`FAULTS_V2` in `run_pipeline_v2.py`):
  a lost or misrouted message, a crashed or lying grader, a corrupted grade, a wrong reference answer,
  a cut-off student answer, an overclaiming Tutor.

## 3. How it would really work in a school

```
 answers (CSV)  ->  [teacher grades a few, optional]  ->  python -m scripts.grade_csv class.csv
                                                              |
              class_graded.csv  <------------------------------+------------------>  class_review.csv
   (grade, confidence, feedback for every answer)                     (only doubtful / flagged answers,
                                                                        with the two most likely grades)
                                                                                     |
                               teacher fills teacher_label, copies it into class.csv, runs again
                               -> those answers become worked examples (answer bank)
```

1. **Collect answers** in a spreadsheet: one row per answer with student, question, reference
   answer, student answer. Google Forms, Microsoft Forms and Moodle all export this as CSV. **Paper answer sheets
   (prototype):** photograph each answer, put the photo's file name in a `photo` column and leave
   `student_answer` empty. A local vision model (qwen2.5vl:7b) reads the handwriting first, and
   answers whose reading contains many unknown words go to the teacher instead of being graded
   automatically.
2. **Optional:** the teacher grades the first few answers to each new question in the
   `teacher_label` column. Those are kept exactly as given and shown to the graders as examples
   (in our simulation this raised accuracy on new questions, e.g. 0.533 -> 0.660 with 2 per question
   on new topics).
3. **Run one command.** At the start of every session the canary check grades 32 answers with known
   grades and benches any grader that has gone wrong. Then every answer is graded, checked and either
   given feedback or held back.
4. **The teacher opens the review file**, which lists only the answers the system was unsure about
   or flagged (cut-off answers, a reference answer that does not match its question), and grades them.
5. **Corrections go back in** as `teacher_label` values; the next run uses them as examples.
6. The teacher stays in control: a teacher's grade is never changed, and doubtful grades are never
   sent automatically.

**Hardware:** the full system needs a laptop with an 8 GB graphics card (ours: RTX 4060). With small
models it needs about 4 GB. With no graphics card at all, the TF-IDF + fine-tuned DeBERTa version
still reaches 0.708 accuracy on familiar questions at about 0.12 s per answer (Table 5 and 7 of the
report pack), but it cannot write feedback text.

**Time:** the 40-answer example class takes about 10-60 seconds once the system has run once (the
first run trains the judge and grades the canaries, which takes several minutes).

---

## 4. Demo-day command sheet

Before mam arrives (10 minutes):

1. Start the **Ollama** app (Start menu -> Ollama). Wait until its icon is in the taskbar tray.
2. Open a terminal in the project folder:
   ```
   cd C:\Users\Gigabyte\edu-masfire
   .venv\Scripts\activate
   ```
3. Warm-up run (loads the models into the graphics card, so the live runs are fast):
   ```
   python -m scripts.demo_v2 --task 23 --fault grader_lying
   ```
4. Open `results\slides\final_presentation.pptx` and the folder `results\charts`.
5. Close other heavy programs (games, other AI tools) so the GPU is free.

Live (about 5-7 minutes):

| Step | Command | What to say | What to point at |
|---|---|---|---|
| 1 | `python -m scripts.demo_v2 --task 23` | "One student answer goes through the whole system. Five graders vote, a judge combines them, the safety layer checks, and the student gets feedback." | the five votes, the judge's grade and confidence, RIGHT, the feedback |
| 2 | `python -m scripts.demo_v2 --task 23 --fault grader_lying` | "Now we break it: qwen confidently gives a wrong grade. Without protection, one lying grader fools the judge. With our safety layer, the canary check catches it before the session and benches it." | Safety layer OFF -> WRONG; ON -> "benched by canary check" -> RIGHT |
| 3 | `python -m scripts.demo_v2 --task 1 --fault student_truncate` | "The student's answer arrives cut in half. The graders would mark it correct, which is wrong. Our input check notices it does not end like a sentence and sends it to the teacher." | OFF -> WRONG; ON -> HELD BACK |
| 4 | `python -m scripts.demo_v2 --task 0 --fault label_corrupted` | "The grade is damaged on its way to the student. Every message carries a checksum, so the damage is detected and the grade is sent again." | OFF -> "correct" (wrong); ON -> "corrupted -> re-sent" -> RIGHT |
| 5 | `python -m scripts.grade_csv examples\class_quiz.csv` | "This is how a teacher would use it: one spreadsheet in, two spreadsheets out." | then open `examples\class_quiz_graded.csv` and `examples\class_quiz_review.csv` in Excel |
| 5b | `python -m scripts.grade_csv results\handwriting\class_photos.csv --no-feedback` (about 1-2 minutes) | "Schools still use paper. These are 28 phone photos of our own handwriting. A vision model on the same laptop reads them, then the same grading and safety checks run. Same grade as the typed answer in 23 of 28. Reading depends a lot on the writer (3% vs 26% of characters wrong) and on holding the phone upright (sideways photos: 28% wrong). Its mistakes are mostly real words, like 'Because' read as 'Became', so photo answers still need a teacher's look." (Takes about 2-3 minutes for 28 photos.) | open `results\handwriting\photos\H12.jpg`, then `class_photos_graded.csv`: the `student_answer` column is what the model read. Faster backup: `python -m scripts.grade_csv examples\photo_quiz\photo_quiz.csv --no-feedback` (6 font sheets, about 45 s) |
| 6 | (slides) | results: old vs new pipeline, faults, cheap-laptop tiers | slides 8-15 |

Other good examples: `python -m scripts.demo_v2 --suggest` lists them for every fault.
The original 4-agent pipeline can be shown with `python -m scripts.demo --task 9 --fault grader_label_flip`.

## 5. If something goes wrong live

- **"Failed to connect to Ollama"**: Ollama is not running. Start the Ollama app, wait 15 seconds, run again.
- **A run is slow (over a minute)**: a model is being loaded or new feedback is being written. Talk
  through the slide while it finishes; the next run is fast.
- **Anything else**: show the saved transcripts in `results\demo\` (made with `--save`, see the
  checklist) or the slides. Keep a screen recording of a full rehearsal as the last backup.

## 6. Rehearsal checklist (day before)

- [ ] Run every command in the table above once, in order, with `--save` added to the demo_v2
      commands, so `results\demo\` holds backup transcripts.
- [ ] Time the pitch (5 minutes) and the demo (5-7 minutes) together, twice.
- [ ] Each team member can answer: what is a fault, what does the judge do, what is the canary check,
      why is the old pipeline worse than one model, what happens to doubtful grades, what does it need
      to run.
- [ ] Fill in the AI-use disclosure slide honestly.
- [ ] Record the screen during one full rehearsal (Windows: Win + G, or Win + Alt + R).
