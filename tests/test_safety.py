# Automatic tests for the safety layer and the fault injection: each check must catch the problem
# it is built for and must let normal answers through. No GPU or Ollama needed (about 10 seconds;
# the word and reference checks read the train split once).
#
# Usage (from the project folder): python -m unittest discover -s tests -t . -v
import unittest

from scripts.clean_ratings import clean_a, clean_b
from src.faults import corrupt_task, donor_index, flip_label
from src.health import (benched, chance_of_score_this_low, check_input, fit_reference_check, looks_misread,
                        reference_looks_wrong)
from src.ocr import character_error_rate, simulate_misreading
from src.pipeline_v2 import check_message, fingerprint, make_message

TASK = {"task_id": "T1", "question": "How can you use a magnet to find out if the key is iron or aluminum?",
        "reference_answer": "If the magnet attracts the key, it is iron. If not, it is aluminum.",
        "student_answer": "If it attracts, it is iron. If it does not, it is aluminum."}
OTHER = {"task_id": "T2", "question": "Why does the tree shown in the picture have a shadow?",
         "reference_answer": "The tree blocks the light from the Sun.",
         "student_answer": "Because the tree blocks the sunlight."}


class MessageChecks(unittest.TestCase):
    """Message envelopes: lost, misrouted and corrupted messages must be noticed."""

    def test_intact_message_is_ok(self):
        self.assertEqual(check_message(make_message(TASK, "judge", {"label": "correct"}), TASK), "ok")

    def test_lost_message(self):
        self.assertEqual(check_message(None, TASK), "lost")

    def test_message_for_another_task_is_misrouted(self):
        self.assertEqual(check_message(make_message(OTHER, "judge", {"label": "correct"}), TASK), "misrouted")

    def test_changed_content_is_corrupted(self):
        message = make_message(TASK, "judge", {"label": "correct"})
        message["payload"] = {"label": "irrelevant"}
        self.assertEqual(check_message(message, TASK), "corrupted")

    def test_fingerprint_changes_with_any_change(self):
        self.assertNotEqual(fingerprint("The tree blocks the light."), fingerprint("The tree blocks the light!"))


class InputCheck(unittest.TestCase):
    """Empty or cut-off answers go to the teacher; normal answers pass."""

    def test_normal_answer_passes(self):
        self.assertEqual(check_input(TASK), [])

    def test_empty_answer(self):
        self.assertTrue(check_input(dict(TASK, student_answer="   ")))

    def test_cut_off_answer(self):
        self.assertTrue(check_input(dict(TASK, student_answer="If it attracts, it is")))


class CanaryCheck(unittest.TestCase):
    """A grader is benched only when its canary score is far below its normal accuracy."""

    def test_far_below_normal_is_benched(self):
        gold = ["correct"] * 32
        self.assertTrue(benched(["correct"] * 8 + ["irrelevant"] * 24, gold, expected=0.7))

    def test_normal_score_is_not_benched(self):
        gold = ["correct"] * 32
        self.assertFalse(benched(["correct"] * 22 + ["irrelevant"] * 10, gold, expected=0.7))

    def test_chance_of_any_score_is_one(self):
        self.assertAlmostEqual(chance_of_score_this_low(32, 32, 0.7), 1.0)


class ReferenceAndPhotoChecks(unittest.TestCase):
    """A reference answer from another question is noticed; misread text is noticed."""

    def test_swapped_reference_is_flagged(self):
        vectorizer = fit_reference_check([TASK, OTHER])
        self.assertFalse(reference_looks_wrong(TASK, vectorizer))
        self.assertTrue(reference_looks_wrong(dict(TASK, reference_answer=OTHER["reference_answer"]), vectorizer))

    def test_clean_reading_is_not_flagged_as_misread(self):
        self.assertFalse(looks_misread(TASK))

    def test_garbled_reading_is_flagged_as_misread(self):
        self.assertTrue(looks_misread(dict(TASK, student_answer="lf it attrocts, lt ls lrom. lf lt dnes nof, lt ls alumlnvm.")))

    def test_simulated_misreading_is_repeatable_and_changes_text(self):
        a, b = simulate_misreading(TASK["student_answer"], seed=1), simulate_misreading(TASK["student_answer"], seed=1)
        self.assertEqual(a, b)
        self.assertNotEqual(a, TASK["student_answer"])

    def test_character_error_rate(self):
        self.assertEqual(character_error_rate("salt", "salt"), 0)
        self.assertAlmostEqual(character_error_rate("salt", "sait"), 0.25)


class FaultInjection(unittest.TestCase):
    """The injected faults really change what the agents see."""

    def test_truncate_keeps_first_half(self):
        seen = corrupt_task(dict(TASK, student_answer="one two three four"), "student_truncate", OTHER)
        self.assertEqual(seen["student_answer"], "one two")

    def test_reference_swap_uses_another_question(self):
        self.assertEqual(corrupt_task(TASK, "reference_swap", OTHER)["reference_answer"], OTHER["reference_answer"])

    def test_donor_has_a_different_question(self):
        tasks = [TASK, dict(TASK, task_id="T1b"), OTHER]
        self.assertEqual(tasks[donor_index(tasks, 0)]["question"], OTHER["question"])

    def test_flipped_label_is_always_different(self):
        for label in ["correct", "contradictory", "partially_correct_incomplete", "irrelevant"]:
            self.assertNotEqual(flip_label(label), label)


class RatingCleanup(unittest.TestCase):
    """The team's rules for cleaning the feedback ratings."""

    def test_rater_a_rules(self):
        self.assertEqual(clean_a("understand"), "helpful")
        self.assertEqual(clean_a("don’t understand"), "not helpful")
        self.assertEqual(clean_a("harmful"), "harmful")
        self.assertEqual(clean_a("yes"), "")

    def test_rater_b_rules(self):
        self.assertEqual(clean_b("hamrful"), "harmful")
        self.assertEqual(clean_b("helpful maybe"), "helpful")
        self.assertEqual(clean_b("no idea"), "")


if __name__ == "__main__":
    unittest.main()
