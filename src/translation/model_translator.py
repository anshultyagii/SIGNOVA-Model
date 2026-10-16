# -*- coding: utf-8 -*-
"""
SignSpeak AI - NLP Model-Based Translator & Configurable Hybrid Translator
==========================================================================
Provides:
  1. T5ISLTranslator: Seq2Seq neural translator using fine-tuned T5-small.
  2. ConfigurableISLTranslator: Unified interface offering selection between
     the fine-tuned neural model and the deterministic rule-based translator.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.recognition.gloss_buffer import canonicalize_gloss
from src.translation.dataset import PROMPT_PREFIX
from src.translation.isl_to_english import (
    GLOSS_LEXICON,
    ISLToEnglishTranslator,
    TranslationResult,
    extract_inferred_words,
)

DEFAULT_MODEL_DIR_V2 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "isl_t5_translator_v2"
)
DEFAULT_MODEL_DIR_V1 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "isl_t5_translator"
)
DEFAULT_MODEL_DIR = DEFAULT_MODEL_DIR_V2 if os.path.isdir(DEFAULT_MODEL_DIR_V2) else DEFAULT_MODEL_DIR_V1


class T5ISLTranslator:
    """
    Inference engine using a fine-tuned T5 Seq2Seq model for ISL gloss-to-English translation.
    Loaded once at startup, thread-safe, and optimized for low-latency CPU execution.
    """
    _cached_model: Optional[AutoModelForSeq2SeqLM] = None
    _cached_tokenizer: Optional[AutoTokenizer] = None
    _cached_dir: Optional[str] = None
    _cached_device: Optional[torch.device] = None

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 64,
        num_beams: int = 2
    ):
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.max_length = max_length
        self.num_beams = num_beams
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSeq2SeqLM] = None
        self.is_loaded = False
        self.load_error: Optional[str] = None

        self._load_model()

    def _load_model(self):
        """Loads tokenizer and weights from model directory once, reusing cached instance."""
        if not os.path.isdir(self.model_dir):
            self.load_error = f"Model checkpoint directory does not exist: {self.model_dir}"
            self.is_loaded = False
            return

        # Check in-memory class cache
        if (
            T5ISLTranslator._cached_model is not None
            and T5ISLTranslator._cached_tokenizer is not None
            and T5ISLTranslator._cached_dir == self.model_dir
            and T5ISLTranslator._cached_device == self.device
        ):
            self.model = T5ISLTranslator._cached_model
            self.tokenizer = T5ISLTranslator._cached_tokenizer
            self.is_loaded = True
            self.load_error = None
            return

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)
            model.to(self.device)
            model.eval()

            self.tokenizer = tokenizer
            self.model = model
            self.is_loaded = True
            self.load_error = None

            T5ISLTranslator._cached_model = model
            T5ISLTranslator._cached_tokenizer = tokenizer
            T5ISLTranslator._cached_dir = self.model_dir
            T5ISLTranslator._cached_device = self.device
        except Exception as exc:
            self.load_error = f"Failed to load model from {self.model_dir}: {exc}"
            self.is_loaded = False

    def _normalize_tokens(self, raw_tokens: Sequence[str]) -> List[str]:
        """Cleans and canonicalizes gloss tokens."""
        cleaned = []
        for token in raw_tokens:
            if token is None:
                continue
            s = str(token).strip()
            if not s or s in ("-", "unknown", "None", "PAUSED"):
                continue
            canonical = canonicalize_gloss(s)
            if canonical:
                cleaned.append(canonical)
        return cleaned

    def translate_detailed(
        self,
        glosses: Sequence[str],
        confidences: Optional[Sequence[float]] = None
    ) -> TranslationResult:
        """
        Translates a complete ordered gloss sequence using the T5 model.
        Returns TranslationResult with translation text, fallback status, and metadata,
        including contextually inferred words and uncertainty flagging.
        """
        raw_list = list(glosses) if glosses is not None else []
        norm_tokens = self._normalize_tokens(raw_list)

        # Safe handling for empty inputs
        if not norm_tokens:
            return TranslationResult(
                text="",
                is_fallback=False,
                matched_rule="EMPTY_INPUT",
                input_glosses=raw_list,
                normalized_glosses=[],
                inferred_words=[],
                uncertainty_flag=False,
            )

        # Fallback if model is not loaded
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            fallback_words = [GLOSS_LEXICON.get(tok, tok) for tok in norm_tokens]
            fallback_sentence = " ".join(fallback_words).strip().capitalize() + "."
            inferred = extract_inferred_words(fallback_sentence, norm_tokens)
            has_low_conf = any(c < 0.60 for c in confidences if isinstance(c, (int, float))) if confidences else False
            return TranslationResult(
                text=fallback_sentence,
                is_fallback=True,
                matched_rule=f"MODEL_UNAVAILABLE_FALLBACK ({self.load_error})",
                input_glosses=raw_list,
                normalized_glosses=norm_tokens,
                inferred_words=inferred,
                uncertainty_flag=has_low_conf or len(norm_tokens) > 1,
            )

        # Build prompt covering the entire sequence
        gloss_str = " ".join(norm_tokens)
        prompt = PROMPT_PREFIX + gloss_str

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    early_stopping=True
                )

            decoded = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

            # Clean and format decoded sentence
            if decoded:
                decoded = decoded[0].upper() + decoded[1:]
                if not decoded.endswith((".", "?", "!")):
                    decoded += "."
            else:
                decoded = " ".join(norm_tokens).capitalize() + "."

            inferred = extract_inferred_words(decoded, norm_tokens)
            has_low_conf = any(c < 0.60 for c in confidences if isinstance(c, (int, float))) if confidences else False

            return TranslationResult(
                text=decoded,
                is_fallback=False,
                matched_rule="T5_SEQ2SEQ_INFERENCE",
                input_glosses=raw_list,
                normalized_glosses=norm_tokens,
                inferred_words=inferred,
                uncertainty_flag=has_low_conf,
            )
        except Exception as exc:
            # Safe recovery from inference error
            fallback_words = [GLOSS_LEXICON.get(tok, tok) for tok in norm_tokens]
            fallback_sentence = " ".join(fallback_words).strip().capitalize() + "."
            inferred = extract_inferred_words(fallback_sentence, norm_tokens)
            has_low_conf = any(c < 0.60 for c in confidences if isinstance(c, (int, float))) if confidences else False
            return TranslationResult(
                text=fallback_sentence,
                is_fallback=True,
                matched_rule=f"INFERENCE_ERROR_FALLBACK ({exc})",
                input_glosses=raw_list,
                normalized_glosses=norm_tokens,
                inferred_words=inferred,
                uncertainty_flag=has_low_conf or len(norm_tokens) > 1,
            )

    def translate(self, glosses: Sequence[str]) -> str:
        """Convenience string-only translation."""
        return self.translate_detailed(glosses).text


class ConfigurableISLTranslator:
    """
    Configurable hybrid translator providing selectable neural model translation
    or deterministic rule-based translation with transparent fallback.
    """

    def __init__(
        self,
        mode: Optional[str] = None,
        model_dir: Optional[str] = None,
        collapse_repetitions: bool = False
    ):
        """
        Args:
            mode: "model" (default if checkpoint exists) or "rules".
                  Can also be configured via TRANSLATOR_MODE env var.
            model_dir: Custom model directory path.
            collapse_repetitions: Parameter forwarded to rule-based fallback.
        """
        env_mode = os.environ.get("TRANSLATOR_MODE", "").lower()
        if mode:
            self.mode = mode.lower()
        elif env_mode in ("model", "rules"):
            self.mode = env_mode
        else:
            # Default to "model" if model checkpoint directory exists, else "rules"
            chk_dir = model_dir or DEFAULT_MODEL_DIR
            self.mode = "model" if os.path.isdir(chk_dir) else "rules"

        self.rule_translator = ISLToEnglishTranslator(collapse_repetitions=collapse_repetitions)
        self.neural_translator: Optional[T5ISLTranslator] = None

        if self.mode == "model":
            self.neural_translator = T5ISLTranslator(model_dir=model_dir)

    @property
    def is_model_loaded(self) -> bool:
        return bool(self.neural_translator and self.neural_translator.is_loaded)

    def set_mode(self, mode: str):
        """Dynamically switches between 'model' and 'rules' mode."""
        clean_mode = mode.lower().strip()
        if clean_mode not in ("model", "rules"):
            raise ValueError(f"Invalid mode '{mode}'. Expected 'model' or 'rules'.")
        self.mode = clean_mode
        if self.mode == "model" and self.neural_translator is None:
            self.neural_translator = T5ISLTranslator()

    def translate_detailed(
        self,
        glosses: Sequence[str],
        confidences: Optional[Sequence[float]] = None
    ) -> TranslationResult:
        """Translates gloss sequence using selected engine."""
        if self.mode == "model" and self.neural_translator is not None:
            result = self.neural_translator.translate_detailed(glosses, confidences=confidences)
            # If model was unavailable or had an inference error, use rule-based result
            if result.is_fallback:
                rule_res = self.rule_translator.translate_detailed(glosses, confidences=confidences)
                return TranslationResult(
                    text=rule_res.text,
                    is_fallback=True,
                    matched_rule=f"MODEL_FALLBACK -> {rule_res.matched_rule}",
                    input_glosses=rule_res.input_glosses,
                    normalized_glosses=rule_res.normalized_glosses,
                    inferred_words=rule_res.inferred_words,
                    uncertainty_flag=rule_res.uncertainty_flag,
                )
            return result
        return self.rule_translator.translate_detailed(glosses, confidences=confidences)

    def translate(self, glosses: Sequence[str]) -> str:
        """Translates gloss sequence to English sentence string."""
        return self.translate_detailed(glosses).text
