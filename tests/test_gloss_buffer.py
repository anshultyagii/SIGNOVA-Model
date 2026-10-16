# -*- coding: utf-8 -*-
"""
SignSpeak AI - Unit Tests for GlossSequenceBuffer
=================================================
Tests:
  - Ordered sequence accumulation.
  - Consecutive duplicate suppression with cooldown.
  - Idle finalization after configurable pause.
  - Ongoing activity preventing premature finalization.
  - Rejection of empty, invalid, and low-confidence predictions.
  - Label canonicalization (e.g. '1_One' -> 'One').
  - Manual finalize and reset behavior.
"""

import unittest
from src.recognition.gloss_buffer import GlossSequenceBuffer, canonicalize_gloss, GlossItem


class TestGlossSequenceBuffer(unittest.TestCase):

    def test_canonicalize_gloss(self):
        self.assertEqual(canonicalize_gloss("1_One"), "One")
        self.assertEqual(canonicalize_gloss("10_Ten"), "Ten")
        self.assertEqual(canonicalize_gloss("Drink"), "Drink")
        self.assertEqual(canonicalize_gloss("  I_Me  "), "I_Me")
        self.assertEqual(canonicalize_gloss(""), "")
        self.assertEqual(canonicalize_gloss(None), "")

    def test_ordered_sequence_accumulation(self):
        buf = GlossSequenceBuffer()
        added1 = buf.add_gloss("I_Me", confidence=0.85, timestamp=1.0)
        added2 = buf.add_gloss("Want_Desire_Wish_For", confidence=0.78, timestamp=1.8)
        added3 = buf.add_gloss("Water", confidence=0.92, timestamp=2.5)

        self.assertTrue(added1)
        self.assertTrue(added2)
        self.assertTrue(added3)
        self.assertEqual(len(buf), 3)
        self.assertEqual(buf.current_sequence, ["I_Me", "Want_Desire_Wish_For", "Water"])
        self.assertEqual(buf.last_gloss, "Water")

        # Verify item details
        items = buf.current_items
        self.assertEqual(items[0].gloss, "I_Me")
        self.assertAlmostEqual(items[0].confidence, 0.85)
        self.assertAlmostEqual(items[0].timestamp, 1.0)

    def test_duplicate_suppression_within_cooldown(self):
        buf = GlossSequenceBuffer(duplicate_cooldown_seconds=1.2)

        # First addition: accepted
        self.assertTrue(buf.add_gloss("Drink", confidence=0.80, timestamp=10.0))
        self.assertEqual(buf.current_sequence, ["Drink"])

        # Rapid repeated prediction of same sign (e.g. at t=10.4): rejected
        self.assertFalse(buf.add_gloss("Drink", confidence=0.82, timestamp=10.4))
        self.assertEqual(buf.current_sequence, ["Drink"])

        # Another repeat at t=11.0 (< 1.2s from original t=10.0): rejected
        self.assertFalse(buf.add_gloss("Drink", confidence=0.79, timestamp=11.0))
        self.assertEqual(buf.current_sequence, ["Drink"])

        # Same sign added after cooldown expires (t=11.5 >= 10.0 + 1.2): accepted
        self.assertTrue(buf.add_gloss("Drink", confidence=0.85, timestamp=11.5))
        self.assertEqual(buf.current_sequence, ["Drink", "Drink"])

    def test_different_sign_not_suppressed(self):
        buf = GlossSequenceBuffer(duplicate_cooldown_seconds=2.0)
        self.assertTrue(buf.add_gloss("One", confidence=0.8, timestamp=1.0))
        # Different sign arriving quickly is accepted
        self.assertTrue(buf.add_gloss("Two", confidence=0.8, timestamp=1.2))
        # "One" again after "Two" is accepted (not consecutive repeat)
        self.assertTrue(buf.add_gloss("One", confidence=0.8, timestamp=1.4))
        self.assertEqual(buf.current_sequence, ["One", "Two", "One"])

    def test_idle_finalization(self):
        buf = GlossSequenceBuffer(idle_timeout_seconds=1.5)
        buf.add_gloss("Drink", confidence=0.8, timestamp=10.0)
        buf.add_gloss("Water", confidence=0.8, timestamp=10.6)

        # Check before timeout (elapsed = 0.5s < 1.5s)
        res_early = buf.check_idle_timeout(timestamp=11.1)
        self.assertIsNone(res_early)
        self.assertFalse(buf.is_empty)

        # Check right at timeout edge (elapsed = 1.4s < 1.5s)
        res_edge = buf.check_idle_timeout(timestamp=12.0)
        self.assertIsNone(res_edge)

        # Check after timeout (elapsed = 1.6s >= 1.5s)
        res_timeout = buf.check_idle_timeout(timestamp=12.2)
        self.assertEqual(res_timeout, ["Drink", "Water"])
        self.assertTrue(buf.is_empty)
        self.assertEqual(buf.completed_sequences, [["Drink", "Water"]])
        self.assertEqual(buf.last_completed_sequence, ["Drink", "Water"])

        # Subsequent check on now-empty buffer returns None
        self.assertIsNone(buf.check_idle_timeout(timestamp=13.0))

    def test_activity_prevents_idle_finalization(self):
        buf = GlossSequenceBuffer(idle_timeout_seconds=1.5)
        buf.add_gloss("Eat", confidence=0.8, timestamp=0.0)

        # Signer is still moving/active at t=1.0 (before 1.5s timeout)
        buf.record_activity(timestamp=1.0)

        # At t=1.8 (1.8s from add_gloss, but only 0.8s from activity): should NOT finalize
        self.assertIsNone(buf.check_idle_timeout(timestamp=1.8))

        # At t=2.6 (1.6s from activity >= 1.5s): should finalize
        emitted = buf.check_idle_timeout(timestamp=2.6)
        self.assertEqual(emitted, ["Eat"])

    def test_empty_and_invalid_inputs_ignored(self):
        buf = GlossSequenceBuffer()
        self.assertFalse(buf.add_gloss(None, timestamp=1.0))
        self.assertFalse(buf.add_gloss("", timestamp=1.0))
        self.assertFalse(buf.add_gloss("   ", timestamp=1.0))
        self.assertFalse(buf.add_gloss("-", timestamp=1.0))
        self.assertFalse(buf.add_gloss("unknown", timestamp=1.0))
        self.assertFalse(buf.add_gloss("PAUSED", timestamp=1.0))
        self.assertTrue(buf.is_empty)
        self.assertIsNone(buf.check_idle_timeout(timestamp=5.0))

    def test_low_confidence_rejected(self):
        buf = GlossSequenceBuffer(min_confidence=0.60)
        # 0.45 < 0.60: rejected
        self.assertFalse(buf.add_gloss("Drink", confidence=0.45, timestamp=1.0))
        self.assertTrue(buf.is_empty)

        # 0.65 >= 0.60: accepted
        self.assertTrue(buf.add_gloss("Drink", confidence=0.65, timestamp=1.5))
        self.assertEqual(buf.current_sequence, ["Drink"])

    def test_label_canonicalization(self):
        buf = GlossSequenceBuffer(canonicalize=True)
        buf.add_gloss("1_One", confidence=0.8, timestamp=1.0)
        buf.add_gloss("10_Ten", confidence=0.8, timestamp=2.0)
        self.assertEqual(buf.current_sequence, ["One", "Ten"])

    def test_manual_finalize_and_reset(self):
        buf = GlossSequenceBuffer()
        # Empty finalize returns empty list
        self.assertEqual(buf.finalize(), [])

        buf.add_gloss("Help", confidence=0.8, timestamp=1.0)
        buf.add_gloss("Food", confidence=0.8, timestamp=2.0)
        self.assertEqual(len(buf), 2)

        # Manual finalize
        emitted = buf.finalize()
        self.assertEqual(emitted, ["Help", "Food"])
        self.assertTrue(buf.is_empty)
        self.assertEqual(buf.completed_sequences, [["Help", "Food"]])

        # Reset clears active sequence
        buf.add_gloss("Go", confidence=0.8, timestamp=3.0)
        self.assertEqual(len(buf), 1)
        buf.reset()
        self.assertTrue(buf.is_empty)
        # History preserved across reset
        self.assertEqual(buf.completed_sequences, [["Help", "Food"]])

        # clear_history wipes both
        buf.clear_history()
        self.assertEqual(buf.completed_sequences, [])

    def test_consecutive_same_sign_with_gesture_boundary(self):
        """
        Legitimate consecutive repetitions of the same sign across distinct gestures
        (separated by a gesture boundary) must both be recorded.
        Frame-level duplicate predictions without a boundary must still be rejected.
        """
        buf = GlossSequenceBuffer(duplicate_cooldown_seconds=1.2)

        # 1. First gesture "One"
        self.assertTrue(buf.add_gloss("One", confidence=0.85, timestamp=1.0))
        self.assertEqual(buf.current_sequence, ["One"])

        # 2. Spurious frame duplicate of same gesture (no boundary, <1.2s): rejected
        self.assertFalse(buf.add_gloss("One", confidence=0.88, timestamp=1.3))
        self.assertEqual(buf.current_sequence, ["One"])

        # 3. Signer drops hands and makes a NEW gesture for "One" quickly (t=1.6s, elapsed 0.6s < 1.2s)
        buf.mark_gesture_boundary()
        self.assertTrue(buf.add_gloss("One", confidence=0.82, timestamp=1.6))
        self.assertEqual(buf.current_sequence, ["One", "One"])

        # 4. Spurious repeat of the 2nd gesture without a new boundary: rejected
        self.assertFalse(buf.add_gloss("One", confidence=0.80, timestamp=1.8))
        self.assertEqual(buf.current_sequence, ["One", "One"])

    def test_idle_finalization_occurs_only_once_and_restarts_cleanly(self):
        """
        Idle finalization must emit the sequence exactly once.
        Subsequent checks must return None until new signs form a fresh sequence.
        """
        buf = GlossSequenceBuffer(idle_timeout_seconds=1.5)

        # Sequence 1: "I_Me" -> "Want_Desire_Wish_For"
        buf.add_gloss("I_Me", confidence=0.9, timestamp=1.0)
        buf.add_gloss("Want_Desire_Wish_For", confidence=0.85, timestamp=1.8)

        # Before timeout: None
        self.assertIsNone(buf.check_idle_timeout(timestamp=2.5))

        # At timeout: emits exactly once
        seq1 = buf.check_idle_timeout(timestamp=3.4)
        self.assertEqual(seq1, ["I_Me", "Want_Desire_Wish_For"])
        self.assertTrue(buf.is_empty)

        # Subsequent idle checks on empty buffer return None (no repeated emissions)
        self.assertIsNone(buf.check_idle_timeout(timestamp=3.5))
        self.assertIsNone(buf.check_idle_timeout(timestamp=4.0))
        self.assertIsNone(buf.check_idle_timeout(timestamp=10.0))

        # Sequence 2: starts fresh
        buf.add_gloss("Water", confidence=0.95, timestamp=12.0)
        self.assertEqual(buf.current_sequence, ["Water"])

        # Before sequence 2 timeout: None
        self.assertIsNone(buf.check_idle_timeout(timestamp=12.8))

        # At sequence 2 timeout: emits only sequence 2
        seq2 = buf.check_idle_timeout(timestamp=13.6)
        self.assertEqual(seq2, ["Water"])
        self.assertTrue(buf.is_empty)

        # Check total history
        self.assertEqual(buf.completed_sequences, [
            ["I_Me", "Want_Desire_Wish_For"],
            ["Water"]
        ])

    def test_reset_clears_gesture_boundary(self):
        buf = GlossSequenceBuffer(duplicate_cooldown_seconds=1.2)
        buf.mark_gesture_boundary()
        buf.reset()
        self.assertFalse(buf._gesture_boundary_active)

    def test_remove_last_gloss_undo(self):
        buf = GlossSequenceBuffer()
        # Pop on empty buffer
        self.assertIsNone(buf.remove_last_gloss())
        self.assertEqual(len(buf), 0)

        # Add 3 glosses
        buf.add_gloss("I_Me", confidence=0.9, timestamp=1.0)
        buf.add_gloss("Want_Desire_Wish_For", confidence=0.85, timestamp=2.0)
        buf.add_gloss("Water", confidence=0.92, timestamp=3.0)
        self.assertEqual(buf.current_sequence, ["I_Me", "Want_Desire_Wish_For", "Water"])

        # Undo last gloss ("Water")
        popped = buf.remove_last_gloss()
        self.assertIsNotNone(popped)
        self.assertEqual(popped.gloss, "Water")
        self.assertEqual(buf.current_sequence, ["I_Me", "Want_Desire_Wish_For"])
        self.assertEqual(buf.last_gloss, "Want_Desire_Wish_For")

        # Undo second gloss ("Want_Desire_Wish_For")
        popped2 = buf.remove_last_gloss()
        self.assertEqual(popped2.gloss, "Want_Desire_Wish_For")
        self.assertEqual(buf.current_sequence, ["I_Me"])
        self.assertEqual(buf.last_gloss, "I_Me")

        # Undo first gloss ("I_Me")
        popped3 = buf.remove_last_gloss()
        self.assertEqual(popped3.gloss, "I_Me")
        self.assertEqual(buf.current_sequence, [])
        self.assertIsNone(buf.last_gloss)

        # Another pop on now-empty buffer
        self.assertIsNone(buf.remove_last_gloss())


if __name__ == "__main__":
    unittest.main()
