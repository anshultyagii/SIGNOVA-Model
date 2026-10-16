# -*- coding: utf-8 -*-
"""
SignSpeak AI - NLP Model Translation Unit & Integration Tests
=============================================================
Tests:
  1. ISL dataset loading and zero-leakage deterministic splitting.
  2. Evaluation metrics (BLEU, ROUGE-L, Exact Match).
  3. T5ISLTranslator fallback behavior when weights are missing.
  4. ConfigurableISLTranslator mode switching (rules vs neural model).
  5. Empty, single, unknown, and full multi-gloss sequence translations.
  6. Neural inference execution and CPU compatibility.
"""

import os
import unittest
from typing import List

from src.translation.dataset import (
    DEFAULT_DATASET_PATH,
    load_isl_gloss_dataset,
    split_dataset,
    PROMPT_PREFIX,
)
from src.translation.metrics import (
    compute_exact_match,
    compute_bleu,
    compute_rouge_l,
    evaluate_predictions,
    generate_qualitative_review,
)
from src.translation.model_translator import (
    ConfigurableISLTranslator,
    T5ISLTranslator,
    DEFAULT_MODEL_DIR,
)
from src.translation.isl_to_english import ISLToEnglishTranslator


class TestDatasetAndSplitting(unittest.TestCase):
    """Verifies dataset loading, format, and zero-leakage splitting."""

    def test_load_dataset_structure(self):
        dataset = load_isl_gloss_dataset()
        self.assertGreaterEqual(len(dataset), 50, "Should load at least 50 genuine pairs")
        first = dataset[0]
        self.assertIn("id", first)
        self.assertIn("glosses", first)
        self.assertIn("gloss_text", first)
        self.assertIn("english_text", first)
        self.assertIsInstance(first["glosses"], list)
        self.assertTrue(len(first["glosses"]) > 0)
        self.assertTrue(len(first["english_text"]) > 0)

    def test_deterministic_split_zero_leakage(self):
        dataset = load_isl_gloss_dataset()
        train_a, test_a = split_dataset(dataset, test_ratio=0.2, seed=42)
        train_b, test_b = split_dataset(dataset, test_ratio=0.2, seed=42)

        # Deterministic
        self.assertEqual([x["id"] for x in train_a], [x["id"] for x in train_b])
        self.assertEqual([x["id"] for x in test_a], [x["id"] for x in test_b])

        # Zero leakage
        train_glosses = {x["gloss_text"] for x in train_a}
        test_glosses = {x["gloss_text"] for x in test_a}
        overlap = train_glosses.intersection(test_glosses)
        self.assertEqual(len(overlap), 0, f"Found data leakage across train/test: {overlap}")


class TestMetrics(unittest.TestCase):
    """Verifies BLEU, ROUGE-L, and Exact Match calculations."""

    def test_exact_match(self):
        preds = ["I want water.", "Where is hospital?"]
        refs = ["I want water.", "Where is the hospital?"]
        em = compute_exact_match(preds, refs)
        self.assertEqual(em, 0.5)

    def test_bleu_and_rouge(self):
        preds = ["I want water.", "Good morning."]
        refs = ["I want water.", "Good morning."]
        bleu = compute_bleu(preds, refs)
        rouge = compute_rouge_l(preds, refs)
        self.assertGreater(bleu, 80.0)
        self.assertGreater(rouge, 80.0)

    def test_qualitative_review(self):
        glosses = ["I_Me Water Want_Desire_Wish_For"]
        preds = ["I want water."]
        refs = ["I want water."]
        review = generate_qualitative_review(glosses, preds, refs)
        self.assertEqual(len(review), 1)
        self.assertEqual(review[0]["gloss"], glosses[0])
        self.assertEqual(review[0]["prediction"], preds[0])
        self.assertEqual(review[0]["reference"], refs[0])


class TestConfigurableTranslator(unittest.TestCase):
    """Verifies hybrid switcher between rule-based and neural translation."""

    def setUp(self):
        self.translator = ConfigurableISLTranslator(mode="rules")

    def test_rules_mode_translation(self):
        self.assertEqual(self.translator.mode, "rules")
        res = self.translator.translate_detailed(["I_Me", "Want_Desire_Wish_For", "Water"])
        self.assertIn("water", res.text.lower())
        self.assertFalse(res.is_fallback)

    def test_mode_switching(self):
        self.translator.set_mode("rules")
        self.assertEqual(self.translator.mode, "rules")
        with self.assertRaises(ValueError):
            self.translator.set_mode("invalid_mode")

    def test_empty_sequence_handling(self):
        res = self.translator.translate_detailed([])
        self.assertEqual(res.text, "")
        self.assertEqual(res.normalized_glosses, [])

    def test_unknown_gloss_fallback(self):
        res = self.translator.translate_detailed(["NONEXISTENT_GLOSS_XYZ"])
        self.assertIn("NONEXISTENT_GLOSS_XYZ", res.text)


class TestT5TranslatorFallbacks(unittest.TestCase):
    """Verifies T5 translator behavior when checkpoint is unavailable."""

    def test_missing_model_fallback(self):
        t5_missing = T5ISLTranslator(model_dir="F:\\SignSpeakAI\\models\\non_existent_dir_999")
        self.assertFalse(t5_missing.is_loaded)
        res = t5_missing.translate_detailed(["Water", "Want_Desire_Wish_For"])
        self.assertTrue(res.is_fallback)
        self.assertIn("water", res.text.lower())

    def test_empty_tokens_t5(self):
        t5_missing = T5ISLTranslator(model_dir="F:\\SignSpeakAI\\models\\non_existent_dir_999")
        res = t5_missing.translate_detailed([])
        self.assertEqual(res.text, "")


class TestT5LoadedModel(unittest.TestCase):
    """Verifies actual neural inference with the fine-tuned T5-small checkpoint."""

    @classmethod
    def setUpClass(cls):
        cls.translator = ConfigurableISLTranslator(mode="model")

    def test_model_loaded(self):
        self.assertTrue(
            self.translator.is_model_loaded,
            "Fine-tuned T5 model should be successfully loaded from models/isl_t5_translator"
        )

    def test_neural_translation_known_patterns(self):
        # SOV sequence: Subject + Object + Verb
        res = self.translator.translate_detailed(["I_Me", "Want_Desire_Wish_For", "Water"])
        self.assertFalse(res.is_fallback)
        self.assertEqual(res.matched_rule, "T5_SEQ2SEQ_INFERENCE")
        self.assertIn("water", res.text.lower())

        # Test single sign
        res_eat = self.translator.translate_detailed(["Eat"])
        self.assertIn("eat", res_eat.text.lower())

    def test_neural_translation_question(self):
        # Question pattern
        res = self.translator.translate_detailed(["What", "Food"])
        self.assertIn("food", res.text.lower())
        self.assertTrue(res.text.endswith("?") or res.text.endswith("."))

    def test_hybrid_mode_switch(self):
        self.translator.set_mode("rules")
        res_rules = self.translator.translate_detailed(["I_Me", "Water"])
        self.translator.set_mode("model")
        res_model = self.translator.translate_detailed(["I_Me", "Water"])
        self.assertIsInstance(res_rules.text, str)
        self.assertIsInstance(res_model.text, str)

    def test_missing_sign_contextual_inference_and_inferred_words(self):
        # 1. Rule mode contextual inference
        self.translator.set_mode("rules")
        res_rules = self.translator.translate_detailed(
            ["I_Me", "Water"],
            confidences=[0.85, 0.90]
        )
        self.assertEqual(res_rules.text, "I want water.")
        self.assertFalse(res_rules.uncertainty_flag)
        self.assertIn("want", res_rules.inferred_words)

        # Low confidence triggers uncertainty flag
        res_low_conf = self.translator.translate_detailed(
            ["I_Me", "Water"],
            confidences=[0.45, 0.90]
        )
        self.assertTrue(res_low_conf.uncertainty_flag)

        # 2. Model mode contextual inference
        self.translator.set_mode("model")
        res_model = self.translator.translate_detailed(
            ["I_Me", "Water"],
            confidences=[0.92, 0.88]
        )
        self.assertIsInstance(res_model.inferred_words, list)
        self.assertFalse(res_model.uncertainty_flag)

        res_model_low = self.translator.translate_detailed(
            ["I_Me", "Water"],
            confidences=[0.35, 0.88]
        )
        self.assertTrue(res_model_low.uncertainty_flag)


if __name__ == "__main__":
    unittest.main()
