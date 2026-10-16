# -*- coding: utf-8 -*-
"""
SignSpeak AI - Unit Tests for Evaluation Script
===============================================
Tests label canonicalization, metric computations, confusion matrix generation,
and skipped sample handling.
"""

import unittest
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_recorded_samples import (
    canonicalize_label,
    compute_evaluation_metrics,
    format_text_confusion_matrix,
    EvaluationSampleResult,
    SkippedSample
)


class TestLabelCanonicalization(unittest.TestCase):
    """Tests label stripping of digit prefixes and formatting."""

    def test_digit_prefixes(self):
        self.assertEqual(canonicalize_label("1_One"), "One")
        self.assertEqual(canonicalize_label("2_Two"), "Two")
        self.assertEqual(canonicalize_label("10_Ten"), "Ten")
        self.assertEqual(canonicalize_label("5_Five"), "Five")
        self.assertEqual(canonicalize_label("9_Nine"), "Nine")

    def test_regular_sign_labels(self):
        self.assertEqual(canonicalize_label("Drink"), "Drink")
        self.assertEqual(canonicalize_label("Water"), "Water")
        self.assertEqual(canonicalize_label("I_Me"), "I_Me")
        self.assertEqual(canonicalize_label("Want_Desire_Wish_For"), "Want_Desire_Wish_For")

    def test_already_canonical_numbers(self):
        self.assertEqual(canonicalize_label("One"), "One")
        self.assertEqual(canonicalize_label("Ten"), "Ten")

    def test_whitespace_and_empty(self):
        self.assertEqual(canonicalize_label("  Food  "), "Food")
        self.assertEqual(canonicalize_label(""), "")
        self.assertEqual(canonicalize_label(None), "")


class TestEvaluationMetricCalculations(unittest.TestCase):
    """Tests accuracy calculation, per-class stats, confusion matrix, and misclassifications."""

    def setUp(self):
        self.classes = ["Drink", "Help", "One", "Water"]

        # Synthetic sample 1: One -> 1_One (Top-1 correct, Top-3 correct)
        s1 = EvaluationSampleResult(
            file_path="dummy/One_1.npz",
            file_name="One_1.npz",
            ground_truth_raw="One",
            ground_truth_canonical="One",
            top1_prediction_raw="1_One",
            top1_prediction_canonical="One",
            top1_similarity=0.85,
            top1_distance=0.176,
            is_top1_correct=True,
            is_top3_correct=True,
            ground_truth_rank=1,
            query_frames=30,
            active_window=(5, 25),
            candidates=[
                {"rank": 1, "label": "1_One", "similarity": 0.85, "distance": 0.176},
                {"rank": 2, "label": "2_Two", "similarity": 0.60, "distance": 0.667},
                {"rank": 3, "label": "Drink", "similarity": 0.50, "distance": 1.000}
            ]
        )

        # Synthetic sample 2: Drink -> Drink (Top-1 correct, Top-3 correct)
        s2 = EvaluationSampleResult(
            file_path="dummy/Drink_1.npz",
            file_name="Drink_1.npz",
            ground_truth_raw="Drink",
            ground_truth_canonical="Drink",
            top1_prediction_raw="Drink",
            top1_prediction_canonical="Drink",
            top1_similarity=0.90,
            top1_distance=0.111,
            is_top1_correct=True,
            is_top3_correct=True,
            ground_truth_rank=1,
            query_frames=28,
            active_window=(4, 24),
            candidates=[
                {"rank": 1, "label": "Drink", "similarity": 0.90, "distance": 0.111},
                {"rank": 2, "label": "Water", "similarity": 0.70, "distance": 0.428},
                {"rank": 3, "label": "Help", "similarity": 0.40, "distance": 1.500}
            ]
        )

        # Synthetic sample 3: Water -> Drink (Top-1 error, Top-3 correct, GT rank 2)
        s3 = EvaluationSampleResult(
            file_path="dummy/Water_1.npz",
            file_name="Water_1.npz",
            ground_truth_raw="Water",
            ground_truth_canonical="Water",
            top1_prediction_raw="Drink",
            top1_prediction_canonical="Drink",
            top1_similarity=0.75,
            top1_distance=0.333,
            is_top1_correct=False,
            is_top3_correct=True,
            ground_truth_rank=2,
            query_frames=40,
            active_window=(5, 35),
            candidates=[
                {"rank": 1, "label": "Drink", "similarity": 0.75, "distance": 0.333},
                {"rank": 2, "label": "Water", "similarity": 0.72, "distance": 0.388},
                {"rank": 3, "label": "One", "similarity": 0.30, "distance": 2.333}
            ]
        )

        # Synthetic sample 4: Help -> One (Top-1 error, Top-3 error, GT rank 4)
        s4 = EvaluationSampleResult(
            file_path="dummy/Help_1.npz",
            file_name="Help_1.npz",
            ground_truth_raw="Help",
            ground_truth_canonical="Help",
            top1_prediction_raw="1_One",
            top1_prediction_canonical="One",
            top1_similarity=0.60,
            top1_distance=0.667,
            is_top1_correct=False,
            is_top3_correct=False,
            ground_truth_rank=4,
            query_frames=35,
            active_window=(3, 30),
            candidates=[
                {"rank": 1, "label": "1_One", "similarity": 0.60, "distance": 0.667},
                {"rank": 2, "label": "Drink", "similarity": 0.55, "distance": 0.818},
                {"rank": 3, "label": "Water", "similarity": 0.45, "distance": 1.222},
                {"rank": 4, "label": "Help", "similarity": 0.40, "distance": 1.500}
            ]
        )

        self.results = [s1, s2, s3, s4]
        self.skipped = [
            SkippedSample(file_path="dummy/corrupt.npz", file_name="corrupt.npz", reason="Zero frames")
        ]

    def test_overall_accuracies(self):
        summary = compute_evaluation_metrics(
            results=self.results,
            skipped=self.skipped,
            all_classes=self.classes
        )
        self.assertEqual(summary.evaluated_count, 4)
        self.assertEqual(summary.skipped_count, 1)
        self.assertEqual(summary.total_files_found, 5)

        # Top-1: 2 correct out of 4 (50%)
        self.assertEqual(summary.top1_correct_count, 2)
        self.assertAlmostEqual(summary.top1_accuracy, 0.50, places=4)

        # Top-3: 3 correct out of 4 (75%)
        self.assertEqual(summary.top3_correct_count, 3)
        self.assertAlmostEqual(summary.top3_accuracy, 0.75, places=4)

    def test_per_class_metrics(self):
        summary = compute_evaluation_metrics(
            results=self.results,
            skipped=self.skipped,
            all_classes=self.classes
        )
        pc = summary.per_class_metrics

        self.assertEqual(pc["One"]["total"], 1)
        self.assertEqual(pc["One"]["top1_correct"], 1)
        self.assertAlmostEqual(pc["One"]["top1_accuracy"], 1.0)

        self.assertEqual(pc["Water"]["total"], 1)
        self.assertEqual(pc["Water"]["top1_correct"], 0)
        self.assertEqual(pc["Water"]["top3_correct"], 1)
        self.assertAlmostEqual(pc["Water"]["top3_accuracy"], 1.0)

        self.assertEqual(pc["Help"]["total"], 1)
        self.assertEqual(pc["Help"]["top1_correct"], 0)
        self.assertEqual(pc["Help"]["top3_correct"], 0)
        self.assertAlmostEqual(pc["Help"]["top1_accuracy"], 0.0)

    def test_confusion_matrix(self):
        summary = compute_evaluation_metrics(
            results=self.results,
            skipped=self.skipped,
            all_classes=self.classes
        )
        cm = summary.confusion_matrix
        # Classes: ["Drink", "Help", "One", "Water"]
        # Drink (idx 0) -> predicted Drink (idx 0): 1
        self.assertEqual(cm[0][0], 1)
        # Help (idx 1) -> predicted One (idx 2): 1
        self.assertEqual(cm[1][2], 1)
        # One (idx 2) -> predicted One (idx 2): 1
        self.assertEqual(cm[2][2], 1)
        # Water (idx 3) -> predicted Drink (idx 0): 1
        self.assertEqual(cm[3][0], 1)

    def test_misclassifications_list(self):
        summary = compute_evaluation_metrics(
            results=self.results,
            skipped=self.skipped,
            all_classes=self.classes
        )
        misc_files = {m["file_name"] for m in summary.misclassifications}
        self.assertIn("Water_1.npz", misc_files)
        self.assertIn("Help_1.npz", misc_files)
        self.assertNotIn("Drink_1.npz", misc_files)
        self.assertNotIn("One_1.npz", misc_files)

    def test_formatting_confusion_matrix(self):
        summary = compute_evaluation_metrics(
            results=self.results,
            skipped=self.skipped,
            all_classes=self.classes
        )
        text_cm = format_text_confusion_matrix(summary.classes, summary.confusion_matrix)
        self.assertIn("True \\ Pred", text_cm)
        self.assertIn("Drink", text_cm)
        self.assertIn("Water", text_cm)


if __name__ == "__main__":
    unittest.main()
