# -*- coding: utf-8 -*-
"""
SignSpeak AI - Translation Package
===================================
Provides deterministic rule-based translation, neural T5 Seq2Seq translation,
and configurable hybrid translation from ISL gloss sequences to English sentences.
"""

from .isl_to_english import (
    ISLToEnglishTranslator,
    TranslationResult,
    translate_gloss_sequence,
)
from .model_translator import (
    T5ISLTranslator,
    ConfigurableISLTranslator,
)
from .dataset import (
    ISLGlossDataset,
    load_isl_gloss_dataset,
    split_dataset,
)
from .metrics import (
    compute_bleu,
    compute_exact_match,
    compute_rouge_l,
    evaluate_predictions,
)

__all__ = [
    "ISLToEnglishTranslator",
    "TranslationResult",
    "translate_gloss_sequence",
    "T5ISLTranslator",
    "ConfigurableISLTranslator",
    "ISLGlossDataset",
    "load_isl_gloss_dataset",
    "split_dataset",
    "compute_bleu",
    "compute_exact_match",
    "compute_rouge_l",
    "evaluate_predictions",
]
