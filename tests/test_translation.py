# -*- coding: utf-8 -*-
"""
SignSpeak AI - Unit Tests for ISL to English Translation Module
==============================================================
Tests cover:
  - All 25 individual sign classes translated to valid English sentences.
  - Digit prefix normalization (e.g. '1_One' -> 'One.').
  - Known sentence rules (SVO, SOV, questions, imperatives).
  - Safe fallback for unknown combinations (e.g. ['What', 'Water'] -> 'What water.').
  - Unknown / unseen labels preserved literally.
  - Empty, None, and invalid token handling.
  - Repeated signs with both natural formatting and collapse options.
  - Punctuation and sentence capitalization integrity.
  - Detailed TranslationResult metadata (rule names, fallback flags).
"""

import unittest
from src.translation.isl_to_english import (
    ALL_25_CLASSES,
    ISLToEnglishTranslator,
    TranslationResult,
    translate_gloss_sequence,
)


class TestISLToEnglishTranslation(unittest.TestCase):

    def setUp(self):
        self.translator = ISLToEnglishTranslator()

    # ── 1. Empty & Invalid Input Handling ─────────────────────────────────────

    def test_empty_and_invalid_inputs(self):
        self.assertEqual(self.translator.translate([]), "")
        self.assertEqual(self.translator.translate([""]), "")
        self.assertEqual(self.translator.translate(["   "]), "")
        self.assertEqual(self.translator.translate([None]), "")
        self.assertEqual(self.translator.translate(["-"]), "")
        self.assertEqual(self.translator.translate(["unknown"]), "")
        self.assertEqual(self.translator.translate(["PAUSED"]), "")
        self.assertEqual(self.translator.translate(["-", "unknown", ""]), "")

    # ── 2. All 25 Individual Sign Classes ─────────────────────────────────────

    def test_all_25_individual_labels(self):
        """Verifies that every enrolled class produces a valid English sentence."""
        self.assertEqual(len(ALL_25_CLASSES), 25, "Expected exactly 25 enrolled classes.")

        for label in ALL_25_CLASSES:
            sentence = self.translator.translate([label])
            self.assertTrue(len(sentence) > 0, f"Translation for '{label}' should not be empty.")
            self.assertTrue(sentence[0].isupper(), f"Sentence '{sentence}' should start with uppercase.")
            self.assertTrue(
                sentence.endswith((".", "?")),
                f"Sentence '{sentence}' should end with '.' or '?'."
            )

    def test_specific_single_signs(self):
        self.assertEqual(self.translator.translate(["Water"]), "Water.")
        self.assertEqual(self.translator.translate(["Food"]), "Food.")
        self.assertEqual(self.translator.translate(["Drink"]), "Drink.")
        self.assertEqual(self.translator.translate(["Eat"]), "Eat.")
        self.assertEqual(self.translator.translate(["Help"]), "Help.")
        self.assertEqual(self.translator.translate(["I_Me"]), "I.")
        self.assertEqual(self.translator.translate(["You"]), "You.")
        self.assertEqual(self.translator.translate(["We"]), "We.")
        self.assertEqual(self.translator.translate(["What"]), "What?")
        self.assertEqual(self.translator.translate(["Where"]), "Where?")

    # ── 3. Digit Prefix Normalization ─────────────────────────────────────────

    def test_digit_prefix_normalization(self):
        self.assertEqual(self.translator.translate(["1_One"]), "One.")
        self.assertEqual(self.translator.translate(["2_Two"]), "Two.")
        self.assertEqual(self.translator.translate(["10_Ten"]), "Ten.")
        self.assertEqual(
            self.translator.translate(["1_One", "Water"]),
            "One water."
        )

    # ── 4. Known Combination Rules ────────────────────────────────────────────

    def test_want_combinations(self):
        # SVO order
        self.assertEqual(
            self.translator.translate(["I_Me", "Want_Desire_Wish_For", "Water"]),
            "I want water."
        )
        self.assertEqual(
            self.translator.translate(["You", "Want_Desire_Wish_For", "Water"]),
            "You want water."
        )
        self.assertEqual(
            self.translator.translate(["I_Me", "Want_Desire_Wish_For", "Food"]),
            "I want food."
        )
        self.assertEqual(
            self.translator.translate(["I_Me", "Want_Desire_Wish_For", "Milk"]),
            "I want milk."
        )
        self.assertEqual(
            self.translator.translate(["I_Me", "Want_Desire_Wish_For", "Drink"]),
            "I want to drink."
        )
        self.assertEqual(
            self.translator.translate(["I_Me", "Want_Desire_Wish_For", "Eat"]),
            "I want to eat."
        )
        self.assertEqual(
            self.translator.translate(["I_Me", "Want_Desire_Wish_For", "Go"]),
            "I want to go."
        )

        # SOV order (natural ISL syntax)
        self.assertEqual(
            self.translator.translate(["I_Me", "Water", "Want_Desire_Wish_For"]),
            "I want water."
        )

    def test_question_rules(self):
        self.assertEqual(self.translator.translate(["Where", "Milk"]), "Where is the milk?")
        self.assertEqual(self.translator.translate(["Where", "Water"]), "Where is the water?")
        self.assertEqual(self.translator.translate(["Where", "Food"]), "Where is the food?")
        self.assertEqual(self.translator.translate(["Where", "You"]), "Where are you?")
        self.assertEqual(self.translator.translate(["Where", "Go"]), "Where are you going?")

        # ISL Topic-Comment order: Object + Where
        self.assertEqual(self.translator.translate(["Milk", "Where"]), "Where is the milk?")

        # What questions
        self.assertEqual(
            self.translator.translate(["What", "You", "Want_Desire_Wish_For"]),
            "What do you want?"
        )

    def test_give_and_help_rules(self):
        self.assertEqual(self.translator.translate(["Give", "I_Me", "Water"]), "Give me water.")
        self.assertEqual(self.translator.translate(["Give", "I_Me", "Food"]), "Give me food.")
        self.assertEqual(self.translator.translate(["Give", "Water"]), "Give water.")
        self.assertEqual(self.translator.translate(["Help", "I_Me"]), "Help me.")
        self.assertEqual(self.translator.translate(["You", "Help", "I_Me"]), "Please help me.")
        self.assertEqual(self.translator.translate(["I_Me", "Help", "You"]), "I will help you.")

    def test_possessive_and_action_rules(self):
        self.assertEqual(self.translator.translate(["My", "Food"]), "My food.")
        self.assertEqual(self.translator.translate(["My", "Water"]), "My water.")
        self.assertEqual(self.translator.translate(["I_Me", "Eat", "Food"]), "I eat food.")
        self.assertEqual(self.translator.translate(["I_Me", "Drink", "Water"]), "I drink water.")
        self.assertEqual(self.translator.translate(["I_Me", "Go"]), "I am going.")
        self.assertEqual(self.translator.translate(["You", "Go"]), "You go.")

    # ── 5. Safe Fallback Behavior ─────────────────────────────────────────────

    def test_safe_fallback_without_hallucinating_meaning(self):
        """
        'What' + 'Water': Never silently invent words like 'about' unless linguistically
        justified; safe fallback preserves original glosses.
        """
        res = self.translator.translate_detailed(["What", "Water"])
        self.assertEqual(res.text, "What water.")
        self.assertTrue(res.is_fallback)

    def test_safe_fallback_unknown_combination(self):
        res = self.translator.translate_detailed(["Food", "Water"])
        self.assertEqual(res.text, "Food water.")
        self.assertTrue(res.is_fallback)

        res2 = self.translator.translate_detailed(["Go", "Eat"])
        self.assertEqual(res2.text, "Go eat.")
        self.assertTrue(res2.is_fallback)

    # ── 6. Unknown / Unseen Labels ────────────────────────────────────────────

    def test_unknown_labels_preserved_literally(self):
        res = self.translator.translate_detailed(["UnseenSign"])
        self.assertEqual(res.text, "UnseenSign.")
        self.assertTrue(res.is_fallback)

        res2 = self.translator.translate_detailed(["CustomGesture", "Water"])
        self.assertEqual(res2.text, "CustomGesture water.")
        self.assertTrue(res2.is_fallback)

    # ── 7. Repeated Signs ─────────────────────────────────────────────────────

    def test_repeated_signs_natural_formatting(self):
        # Default: collapse_repetitions=False -> formats as repetition
        self.assertEqual(self.translator.translate(["Help", "Help"]), "Help, help.")
        self.assertEqual(self.translator.translate(["Water", "Water"]), "Water, water.")
        self.assertEqual(self.translator.translate(["One", "One"]), "One, one.")
        self.assertEqual(self.translator.translate(["Help", "Help", "Help"]), "Help, help, help.")

    def test_repeated_signs_collapse_option(self):
        collapsing_translator = ISLToEnglishTranslator(collapse_repetitions=True)
        self.assertEqual(collapsing_translator.translate(["Help", "Help"]), "Help.")
        self.assertEqual(collapsing_translator.translate(["Water", "Water"]), "Water.")
        # Repeated subject in sentence
        self.assertEqual(
            collapsing_translator.translate(["I_Me", "I_Me", "Want_Desire_Wish_For", "Water"]),
            "I want water."
        )

    # ── 8. Punctuation & Capitalization ───────────────────────────────────────

    def test_punctuation_and_capitalization(self):
        test_cases = [
            (["Help"], "Help."),
            (["Where"], "Where?"),
            (["What"], "What?"),
            (["I_Me", "Go"], "I am going."),
            (["Where", "Milk"], "Where is the milk?"),
            (["one", "two"], "One two."),
        ]
        for glosses, expected in test_cases:
            result = self.translator.translate(glosses)
            self.assertEqual(result, expected)
            self.assertTrue(result[0].isupper())
            self.assertTrue(result.endswith((".", "?")))
            # No double punctuation
            self.assertFalse(result.endswith(".."))
            self.assertFalse(result.endswith("??"))

    # ── 9. Standalone Convenience Function ────────────────────────────────────

    def test_convenience_function(self):
        text = translate_gloss_sequence(["I_Me", "Want_Desire_Wish_For", "Water"])
        self.assertEqual(text, "I want water.")

        text_rep = translate_gloss_sequence(["Help", "Help"], collapse_repetitions=True)
        self.assertEqual(text_rep, "Help.")


if __name__ == "__main__":
    unittest.main()
