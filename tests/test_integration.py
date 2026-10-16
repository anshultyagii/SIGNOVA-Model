# -*- coding: utf-8 -*-
"""
SignSpeak AI - Integration Tests: Gloss Sequence Buffer & ISL-to-English Translation
===================================================================================
Tests end-to-end integration between:
  - GlossSequenceBuffer: accumulates ordered gloss tokens, suppresses duplicates,
    detects idle periods, and finalizes sequences.
  - ISLToEnglishTranslator: receives finalized sequences and produces readable English.

Test coverage:
  1. Known sequence translating correctly (SVO, SOV, questions).
  2. Unknown glosses using documented safe fallback without hallucination.
  3. Empty and invalid sequences handled safely without exceptions.
  4. Multiple finalized sequences translating completely independently.
  5. Translation triggered strictly once per finalized sequence across multiple frames.
"""

import unittest
from src.recognition.gloss_buffer import GlossSequenceBuffer
from src.translation import ISLToEnglishTranslator, TranslationResult


class TestEndToEndIntegration(unittest.TestCase):

    def setUp(self):
        self.buffer = GlossSequenceBuffer(idle_timeout_seconds=1.5, min_confidence=0.50)
        self.translator = ISLToEnglishTranslator()

    # ── 1. Known Sequence Translating Correctly ───────────────────────────────

    def test_known_sequence_translates_correctly(self):
        """
        Simulates: Signer signs 'I_Me', then 'Want_Desire_Wish_For', then 'Water'.
        After 1.5s idle pause, sequence finalizes and translates to 'I want water.'.
        """
        # Add signs with timestamps
        self.assertTrue(self.buffer.add_gloss("I_Me", confidence=0.85, timestamp=1.0))
        self.assertTrue(self.buffer.add_gloss("Want_Desire_Wish_For", confidence=0.80, timestamp=2.0))
        self.assertTrue(self.buffer.add_gloss("Water", confidence=0.90, timestamp=2.8))

        # Check before idle timeout (elapsed = 0.5s < 1.5s)
        emitted_early = self.buffer.check_idle_timeout(timestamp=3.3)
        self.assertIsNone(emitted_early)

        # Trigger idle timeout (elapsed = 1.6s >= 1.5s)
        emitted = self.buffer.check_idle_timeout(timestamp=4.4)
        self.assertIsNotNone(emitted)
        self.assertEqual(emitted, ["I_Me", "Want_Desire_Wish_For", "Water"])

        # Translate finalized sequence
        english = self.translator.translate(emitted)
        self.assertEqual(english, "I want water.")

    def test_known_question_and_sov_sequences(self):
        """Tests natural ISL questions and SOV orders."""
        buf = GlossSequenceBuffer(idle_timeout_seconds=1.5)

        # Where question: 'Where' -> 'Milk'
        buf.add_gloss("Where", confidence=0.88, timestamp=10.0)
        buf.add_gloss("Milk", confidence=0.84, timestamp=11.0)
        emitted_q = buf.check_idle_timeout(timestamp=12.6)
        self.assertEqual(self.translator.translate(emitted_q), "Where is the milk?")

        # SOV word order: 'I_Me' -> 'Food' -> 'Want_Desire_Wish_For'
        buf.add_gloss("I_Me", confidence=0.82, timestamp=20.0)
        buf.add_gloss("Food", confidence=0.85, timestamp=21.0)
        buf.add_gloss("Want_Desire_Wish_For", confidence=0.79, timestamp=22.0)
        emitted_sov = buf.check_idle_timeout(timestamp=23.6)
        self.assertEqual(self.translator.translate(emitted_sov), "I want food.")

    # ── 2. Unknown Glosses & Safe Fallback ─────────────────────────────────────

    def test_unknown_glosses_use_documented_safe_fallback(self):
        """
        Unrecognized glosses or unmodeled combinations must use the documented
        safe fallback that preserves original glosses without inventing unsaid words.
        """
        # Unknown gesture + known noun
        self.buffer.add_gloss("CustomGesture", confidence=0.75, timestamp=1.0)
        self.buffer.add_gloss("Water", confidence=0.85, timestamp=2.0)

        emitted = self.buffer.check_idle_timeout(timestamp=3.6)
        self.assertEqual(emitted, ["CustomGesture", "Water"])

        res: TranslationResult = self.translator.translate_detailed(emitted)
        self.assertEqual(res.text, "CustomGesture water.")
        self.assertTrue(res.is_fallback)
        self.assertEqual(res.matched_rule, "SAFE_FALLBACK")

    def test_unmodeled_combination_safe_fallback(self):
        """
        'What' + 'Water': Safe fallback preserves original tokens ('What water.')
        without hallucinating prepositions like 'about'.
        """
        self.buffer.add_gloss("What", confidence=0.85, timestamp=1.0)
        self.buffer.add_gloss("Water", confidence=0.88, timestamp=2.0)

        emitted = self.buffer.check_idle_timeout(timestamp=3.6)
        res = self.translator.translate_detailed(emitted)
        self.assertEqual(res.text, "What water.")
        self.assertTrue(res.is_fallback)

    # ── 3. Empty & Invalid Sequences Handled Safely ───────────────────────────

    def test_empty_sequence_handling(self):
        """Idle checks on empty buffer must return None and translate to empty string."""
        emitted = self.buffer.check_idle_timeout(timestamp=10.0)
        self.assertIsNone(emitted)

        # Direct empty sequence translation
        self.assertEqual(self.translator.translate([]), "")
        self.assertEqual(self.translator.translate(None), "")

    def test_invalid_tokens_rejected_without_triggering_translation(self):
        """Invalid tokens must not enter the buffer or trigger translation."""
        self.assertFalse(self.buffer.add_gloss(None, timestamp=1.0))
        self.assertFalse(self.buffer.add_gloss("", timestamp=1.1))
        self.assertFalse(self.buffer.add_gloss("-", timestamp=1.2))
        self.assertFalse(self.buffer.add_gloss("unknown", timestamp=1.3))
        self.assertFalse(self.buffer.add_gloss("PAUSED", timestamp=1.4))

        self.assertTrue(self.buffer.is_empty)
        emitted = self.buffer.check_idle_timeout(timestamp=5.0)
        self.assertIsNone(emitted)

    # ── 4. Multiple Finalized Sequences Translate Independently ───────────────

    def test_multiple_finalized_sequences_translate_independently(self):
        """
        Multiple sequences finalized sequentially over a session must translate
        completely independently without residue from previous sequences.
        """
        translations = []

        # Sequence 1: 'I_Me' -> 'Want_Desire_Wish_For' -> 'Food'
        self.buffer.add_gloss("I_Me", confidence=0.85, timestamp=1.0)
        self.buffer.add_gloss("Want_Desire_Wish_For", confidence=0.80, timestamp=1.8)
        self.buffer.add_gloss("Food", confidence=0.92, timestamp=2.5)

        seq1 = self.buffer.check_idle_timeout(timestamp=4.1)
        self.assertIsNotNone(seq1)
        translations.append(self.translator.translate(seq1))

        # Check subsequent frame immediately returns None
        self.assertIsNone(self.buffer.check_idle_timeout(timestamp=4.2))

        # Sequence 2: 'You' -> 'Go'
        self.buffer.add_gloss("You", confidence=0.88, timestamp=5.0)
        self.buffer.add_gloss("Go", confidence=0.83, timestamp=6.0)

        seq2 = self.buffer.check_idle_timeout(timestamp=7.6)
        self.assertIsNotNone(seq2)
        translations.append(self.translator.translate(seq2))

        # Sequence 3: 'Where' -> 'Food'
        self.buffer.add_gloss("Where", confidence=0.91, timestamp=9.0)
        self.buffer.add_gloss("Food", confidence=0.89, timestamp=10.0)

        seq3 = self.buffer.check_idle_timeout(timestamp=11.6)
        self.assertIsNotNone(seq3)
        translations.append(self.translator.translate(seq3))

        # Verify all 3 translations match expected outputs independently
        self.assertEqual(translations, [
            "I want food.",
            "You go.",
            "Where is the food?"
        ])
        self.assertEqual(len(self.buffer.completed_sequences), 3)

    # ── 5. Translation Triggered Exactly Once Per Sequence ────────────────────

    def test_translation_triggered_strictly_once_across_frames(self):
        """
        Simulates a continuous 30-frame webcam capture loop while idle.
        Translation must execute exactly ONCE when the idle timeout threshold
        is crossed, and zero times on all surrounding frames.
        """
        self.buffer.add_gloss("Water", confidence=0.90, timestamp=1.0)

        translation_events = []

        # Simulate 30 frames at ~100ms intervals from t=1.1 to t=4.0
        for step in range(30):
            current_time = 1.1 + step * 0.1
            emitted = self.buffer.check_idle_timeout(timestamp=current_time)
            if emitted is not None:
                trans = self.translator.translate(emitted)
                translation_events.append((current_time, trans))

        # Exactly 1 event must have fired
        self.assertEqual(len(translation_events), 1)
        event_time, event_text = translation_events[0]
        self.assertAlmostEqual(event_time, 2.5, places=1)
        self.assertEqual(event_text, "Water.")


if __name__ == "__main__":
    unittest.main()
