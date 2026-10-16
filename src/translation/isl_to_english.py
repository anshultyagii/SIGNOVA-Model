# -*- coding: utf-8 -*-
"""
SignSpeak AI - ISL to English Translation Prototype
===================================================

IMPORTANT LINGUISTIC DISCLAIMER:
--------------------------------
This is a basic rule-based prototype for ISL-to-English translation,
NOT a linguistically validated Indian Sign Language translator.

Indian Sign Language (ISL) has its own distinct linguistic structure,
including:
  - Topic-Comment sentence structure
  - Subject-Object-Verb (SOV) typical word order
  - Non-Manual Markers (facial expressions, head tilts, shoulder shifts)
    which convey essential grammatical elements like negation and interrogation
  - Spatial indexing and directional verbs

Linear gloss-to-English translation without non-manual markers is inherently
approximate. This module provides transparent, deterministic mapping and
documented template rules covering the current 25-sign vocabulary, with safe
literal fallbacks when no grammatical template is matched.
"""

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Sequence, Tuple

from src.recognition.gloss_buffer import canonicalize_gloss


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary Definitions & Lexicon (25 Enrolled Classes)
# ─────────────────────────────────────────────────────────────────────────────

DIGIT_CLASSES = {
    "One", "Two", "Three", "Four", "Five",
    "Six", "Seven", "Eight", "Nine", "Ten",
}

NOUN_CLASSES = {
    "Food", "Milk", "Water",
}

VERB_CLASSES = {
    "Drink", "Eat", "Give", "Go", "Help", "Want_Desire_Wish_For",
}

PRONOUN_CLASSES = {
    "I_Me", "You", "We", "My",
}

QUESTION_CLASSES = {
    "What", "Where",
}

ALL_25_CLASSES = (
    DIGIT_CLASSES | NOUN_CLASSES | VERB_CLASSES | PRONOUN_CLASSES | QUESTION_CLASSES
)

# Basic dictionary mapping for fallback and word-level lookup
GLOSS_LEXICON: Dict[str, str] = {
    # Digits
    "One": "one",
    "Two": "two",
    "Three": "three",
    "Four": "four",
    "Five": "five",
    "Six": "six",
    "Seven": "seven",
    "Eight": "eight",
    "Nine": "nine",
    "Ten": "ten",
    # Verbs / Actions
    "Drink": "drink",
    "Eat": "eat",
    "Give": "give",
    "Go": "go",
    "Help": "help",
    "Want_Desire_Wish_For": "want",
    # Nouns / Concepts
    "Food": "food",
    "Milk": "milk",
    "Water": "water",
    # Pronouns / Determiners
    "I_Me": "I",
    "You": "you",
    "We": "we",
    "My": "my",
    # Questions
    "What": "what",
    "Where": "where",
}

# Single-token sentences for all 25 classes
SINGLE_GLOSS_TEMPLATES: Dict[str, str] = {
    # Questions
    "What": "What?",
    "Where": "Where?",
    # Pronouns
    "I_Me": "I.",
    "My": "My.",
    "You": "You.",
    "We": "We.",
    # Verbs
    "Drink": "Drink.",
    "Eat": "Eat.",
    "Give": "Give.",
    "Go": "Go.",
    "Help": "Help.",
    "Want_Desire_Wish_For": "Want.",
    # Nouns
    "Food": "Food.",
    "Milk": "Milk.",
    "Water": "Water.",
    # Digits
    "One": "One.",
    "Two": "Two.",
    "Three": "Three.",
    "Four": "Four.",
    "Five": "Five.",
    "Six": "Six.",
    "Seven": "Seven.",
    "Eight": "Eight.",
    "Nine": "Nine.",
    "Ten": "Ten.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Sentence Templates & Rules
# ─────────────────────────────────────────────────────────────────────────────

# Exact sequence rules (tuple of canonical glosses -> English sentence)
SENTENCE_RULES: Dict[Tuple[str, ...], str] = {
    # ── Three-token combinations (SVO and SOV) ────────────────────────
    # I_Me + Want + Object/Action
    ("I_Me", "Want_Desire_Wish_For", "Water"): "I want water.",
    ("I_Me", "Want_Desire_Wish_For", "Food"): "I want food.",
    ("I_Me", "Want_Desire_Wish_For", "Milk"): "I want milk.",
    ("I_Me", "Want_Desire_Wish_For", "Drink"): "I want to drink.",
    ("I_Me", "Want_Desire_Wish_For", "Eat"): "I want to eat.",
    ("I_Me", "Want_Desire_Wish_For", "Go"): "I want to go.",
    ("I_Me", "Want_Desire_Wish_For", "Help"): "I want help.",

    # SOV word order: I_Me + Object + Want (natural ISL syntax)
    ("I_Me", "Water", "Want_Desire_Wish_For"): "I want water.",
    ("I_Me", "Food", "Want_Desire_Wish_For"): "I want food.",
    ("I_Me", "Milk", "Want_Desire_Wish_For"): "I want milk.",

    # You + Want + Object/Action
    ("You", "Want_Desire_Wish_For", "Water"): "You want water.",
    ("You", "Want_Desire_Wish_For", "Food"): "You want food.",
    ("You", "Want_Desire_Wish_For", "Milk"): "You want milk.",
    ("You", "Want_Desire_Wish_For", "Drink"): "You want to drink.",
    ("You", "Want_Desire_Wish_For", "Eat"): "You want to eat.",
    ("You", "Want_Desire_Wish_For", "Go"): "You want to go.",
    ("You", "Want_Desire_Wish_For", "Help"): "You want help.",

    # SOV: You + Object + Want
    ("You", "Water", "Want_Desire_Wish_For"): "You want water.",
    ("You", "Food", "Want_Desire_Wish_For"): "You want food.",
    ("You", "Milk", "Want_Desire_Wish_For"): "You want milk.",

    # We + Want + Object/Action
    ("We", "Want_Desire_Wish_For", "Water"): "We want water.",
    ("We", "Want_Desire_Wish_For", "Food"): "We want food.",
    ("We", "Want_Desire_Wish_For", "Milk"): "We want milk.",
    ("We", "Want_Desire_Wish_For", "Drink"): "We want to drink.",
    ("We", "Want_Desire_Wish_For", "Eat"): "We want to eat.",
    ("We", "Want_Desire_Wish_For", "Go"): "We want to go.",
    ("We", "Want_Desire_Wish_For", "Help"): "We want help.",

    # Give combinations
    ("Give", "I_Me", "Water"): "Give me water.",
    ("Give", "I_Me", "Food"): "Give me food.",
    ("Give", "I_Me", "Milk"): "Give me milk.",
    ("Give", "You", "Water"): "Give you water.",
    ("Give", "You", "Food"): "Give you food.",
    ("Give", "You", "Milk"): "Give you milk.",

    # Questions (3-token)
    ("What", "You", "Want_Desire_Wish_For"): "What do you want?",
    ("What", "You", "Eat"): "What do you eat?",
    ("What", "You", "Drink"): "What do you drink?",

    # Subject + Verb + Object
    ("I_Me", "Drink", "Water"): "I drink water.",
    ("I_Me", "Drink", "Milk"): "I drink milk.",
    ("I_Me", "Eat", "Food"): "I eat food.",
    ("You", "Drink", "Water"): "You drink water.",
    ("You", "Drink", "Milk"): "You drink milk.",
    ("You", "Eat", "Food"): "You eat food.",
    ("We", "Drink", "Water"): "We drink water.",
    ("We", "Drink", "Milk"): "We drink milk.",
    ("We", "Eat", "Food"): "We eat food.",

    # Help interactions
    ("You", "Help", "I_Me"): "Please help me.",
    ("I_Me", "Help", "You"): "I will help you.",

    # ── Two-token combinations ────────────────────────────────────────
    # Where questions
    ("Where", "Milk"): "Where is the milk?",
    ("Where", "Water"): "Where is the water?",
    ("Where", "Food"): "Where is the food?",
    ("Where", "You"): "Where are you?",
    ("Where", "I_Me"): "Where am I?",
    ("Where", "We"): "Where are we?",
    ("Where", "Go"): "Where are you going?",

    # Topic + Question (ISL order: Object + Where)
    ("Milk", "Where"): "Where is the milk?",
    ("Water", "Where"): "Where is the water?",
    ("Food", "Where"): "Where is the food?",
    ("You", "Where"): "Where are you?",

    # Give + Noun
    ("Give", "Water"): "Give water.",
    ("Give", "Food"): "Give food.",
    ("Give", "Milk"): "Give milk.",

    # Help requests
    ("Help", "I_Me"): "Help me.",
    ("Help", "You"): "Help you.",

    # Subject + Want
    ("I_Me", "Want_Desire_Wish_For"): "I want.",
    ("You", "Want_Desire_Wish_For"): "You want.",
    ("We", "Want_Desire_Wish_For"): "We want.",

    # Want + Object / Verb
    ("Want_Desire_Wish_For", "Water"): "Want water.",
    ("Want_Desire_Wish_For", "Food"): "Want food.",
    ("Want_Desire_Wish_For", "Milk"): "Want milk.",
    ("Want_Desire_Wish_For", "Drink"): "Want to drink.",
    ("Want_Desire_Wish_For", "Eat"): "Want to eat.",
    ("Want_Desire_Wish_For", "Go"): "Want to go.",
    ("Want_Desire_Wish_For", "Help"): "Want help.",

    # Subject + Action
    ("I_Me", "Go"): "I am going.",
    ("You", "Go"): "You go.",
    ("We", "Go"): "We are going.",
    ("I_Me", "Eat"): "I eat.",
    ("You", "Eat"): "You eat.",
    ("We", "Eat"): "We eat.",
    ("I_Me", "Drink"): "I drink.",
    ("You", "Drink"): "You drink.",
    ("We", "Drink"): "We drink.",
    ("I_Me", "Help"): "I help.",

    # Possessives
    ("My", "Food"): "My food.",
    ("My", "Water"): "My water.",
    ("My", "Milk"): "My milk.",

    # Verb + Object
    ("Drink", "Water"): "Drink water.",
    ("Drink", "Milk"): "Drink milk.",
    ("Eat", "Food"): "Eat food.",

    # Missing-sign contextual inference (elliptical signing)
    ("I_Me", "Water"): "I want water.",
    ("I_Me", "Food"): "I want food.",
    ("I_Me", "Milk"): "I want milk.",
    ("You", "Water"): "You want water.",
    ("You", "Food"): "You want food.",
    ("You", "Milk"): "You want milk.",
    ("We", "Water"): "We want water.",
    ("We", "Food"): "We want food.",
    ("We", "Milk"): "We want milk.",
}

# ─────────────────────────────────────────────────────────────────────────────
# Inferred Words & Contextual Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

GLOSS_WORD_STEMS: Dict[str, set] = {
    "I_Me": {"i", "me", "my", "myself"},
    "You": {"you", "your", "yours", "yourself"},
    "We": {"we", "us", "our", "ours"},
    "My": {"my", "mine"},
    "Want_Desire_Wish_For": {"want", "wants", "wanted", "desire", "desires", "wish", "wishes"},
    "Water": {"water"},
    "Food": {"food"},
    "Milk": {"milk"},
    "Drink": {"drink", "drinks", "drinking", "drank"},
    "Eat": {"eat", "eats", "eating", "ate"},
    "Give": {"give", "gives", "giving", "gave"},
    "Go": {"go", "goes", "going", "went"},
    "Help": {"help", "helps", "helping", "helped"},
    "What": {"what"},
    "Where": {"where"},
    "One": {"one", "1"},
    "Two": {"two", "2"},
    "Three": {"three", "3"},
    "Four": {"four", "4"},
    "Five": {"five", "5"},
    "Six": {"six", "6"},
    "Seven": {"seven", "7"},
    "Eight": {"eight", "8"},
    "Nine": {"nine", "9"},
    "Ten": {"ten", "10"},
}


def extract_inferred_words(text: str, normalized_glosses: Sequence[str]) -> List[str]:
    """
    Identifies words present in the translation that were contextually inferred
    or grammatically inserted rather than directly recognized from the input glosses.
    """
    if not text or not normalized_glosses:
        return []

    known_stems = set()
    for gloss in normalized_glosses:
        stems = GLOSS_WORD_STEMS.get(gloss)
        if stems:
            known_stems.update(stems)
        else:
            known_stems.add(gloss.lower())

    tokens = re.findall(r"\b[A-Za-z0-9']+\b", text)
    inferred = []
    for tok in tokens:
        clean = tok.lower()
        if clean not in known_stems and clean not in inferred:
            inferred.append(clean)
    return inferred


@dataclass
class TranslationResult:
    """Detailed output from the translation engine."""
    text: str
    is_fallback: bool
    matched_rule: Optional[str] = None
    input_glosses: Optional[List[str]] = None
    normalized_glosses: Optional[List[str]] = None
    inferred_words: Optional[List[str]] = None
    uncertainty_flag: bool = False


class ISLToEnglishTranslator:
    """
    Translates ordered ISL gloss sequences into readable English sentences
    using deterministic rules and templates.
    """

    def __init__(self, collapse_repetitions: bool = False):
        """
        Args:
            collapse_repetitions: If True, collapses consecutive identical signs
                                 (e.g. ['Help', 'Help'] -> ['Help']) before translation.
                                 If False, formats repetitions naturally (e.g. 'Help, help.').
        """
        self.collapse_repetitions = collapse_repetitions

    def _normalize_tokens(self, raw_tokens: Sequence[str]) -> List[str]:
        """Cleans, canonicalizes, and filters empty/invalid tokens."""
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

    def _collapse_duplicates(self, tokens: List[str]) -> List[str]:
        """Collapses adjacent identical tokens."""
        if not tokens:
            return []
        collapsed = [tokens[0]]
        for tok in tokens[1:]:
            if tok != collapsed[-1]:
                collapsed.append(tok)
        return collapsed

    def _format_sentence(self, words: List[str], is_question: bool = False) -> str:
        """Formats words into a capitalized, punctuated sentence."""
        if not words:
            return ""
        sentence = " ".join(words).strip()
        if not sentence:
            return ""
        # Capitalize first character
        sentence = sentence[0].upper() + sentence[1:]
        # Ensure proper punctuation
        punct = "?" if is_question else "."
        if not sentence.endswith((".", "?", "!")):
            sentence += punct
        return sentence

    def _format_repeated_tokens(self, tokens: List[str]) -> str:
        """Formats repeated identical signs (e.g. ['Help', 'Help'] -> 'Help, help.')."""
        mapped_words = [GLOSS_LEXICON.get(tok, tok) for tok in tokens]
        if not mapped_words:
            return ""
        first = mapped_words[0].capitalize()
        rest = [w.lower() for w in mapped_words[1:]]
        sentence = f"{first}, {', '.join(rest)}."
        return sentence

    def translate_detailed(
        self,
        glosses: Sequence[str],
        confidences: Optional[Sequence[float]] = None
    ) -> TranslationResult:
        """
        Translates a gloss sequence and returns a TranslationResult with metadata,
        including contextually inferred words and uncertainty flagging.
        """
        raw_list = list(glosses) if glosses is not None else []
        norm_tokens = self._normalize_tokens(raw_list)

        if not norm_tokens:
            return TranslationResult(
                text="",
                is_fallback=False,
                matched_rule=None,
                input_glosses=raw_list,
                normalized_glosses=[],
                inferred_words=[],
                uncertainty_flag=False,
            )

        active_tokens = (
            self._collapse_duplicates(norm_tokens)
            if self.collapse_repetitions
            else list(norm_tokens)
        )

        tuple_key = tuple(active_tokens)
        text = ""
        is_fallback = False
        matched_rule = None

        # 1. Check exact multi-token sentence rules
        if tuple_key in SENTENCE_RULES:
            text = SENTENCE_RULES[tuple_key]
            is_fallback = False
            matched_rule = f"SENTENCE_RULES[{' + '.join(tuple_key)}]"

        # 2. Check (Number + Noun) pattern (e.g. 'One Water' -> 'One water.')
        elif (
            len(active_tokens) == 2
            and active_tokens[0] in DIGIT_CLASSES
            and active_tokens[1] in NOUN_CLASSES
        ):
            num = active_tokens[0]
            noun = GLOSS_LEXICON.get(active_tokens[1], active_tokens[1].lower())
            text = f"{num} {noun}."
            is_fallback = False
            matched_rule = f"NUMBER_NOUN[{num} + {active_tokens[1]}]"

        # 3. Check single gloss
        elif len(active_tokens) == 1:
            token = active_tokens[0]
            if token in SINGLE_GLOSS_TEMPLATES:
                text = SINGLE_GLOSS_TEMPLATES[token]
                is_fallback = False
                matched_rule = f"SINGLE_GLOSS[{token}]"
            else:
                # Single unknown token fallback
                raw_fallback = token[0].upper() + token[1:]
                text = f"{raw_fallback}."
                is_fallback = True
                matched_rule = "SINGLE_UNKNOWN_FALLBACK"

        # 4. Check repeated identical tokens (e.g. ['Help', 'Help'] -> 'Help, help.')
        elif len(active_tokens) > 1 and len(set(active_tokens)) == 1:
            text = self._format_repeated_tokens(active_tokens)
            is_fallback = False
            matched_rule = f"REPEATED_TOKENS[{active_tokens[0]}x{len(active_tokens)}]"

        # 5. Documented Safe Fallback:
        # Never silently invent meaning (e.g. do not inject 'about' into 'What Water').
        # Map known glosses to basic English equivalents and preserve unknown glosses literally.
        else:
            fallback_words = []
            for tok in active_tokens:
                if tok in GLOSS_LEXICON:
                    fallback_words.append(GLOSS_LEXICON[tok])
                else:
                    fallback_words.append(tok)

            text = self._format_sentence(fallback_words, is_question=False)
            is_fallback = True
            matched_rule = "SAFE_FALLBACK"

        inferred = extract_inferred_words(text, norm_tokens)
        has_low_conf = False
        if confidences:
            has_low_conf = any(c < 0.60 for c in confidences if isinstance(c, (int, float)))
        uncertainty = has_low_conf or (is_fallback and len(norm_tokens) > 1)

        return TranslationResult(
            text=text,
            is_fallback=is_fallback,
            matched_rule=matched_rule,
            input_glosses=raw_list,
            normalized_glosses=norm_tokens,
            inferred_words=inferred,
            uncertainty_flag=uncertainty,
        )

    def translate(self, glosses: Sequence[str]) -> str:
        """
        Translates an ordered gloss sequence to a clean English sentence.
        """
        return self.translate_detailed(glosses).text


def translate_gloss_sequence(
    glosses: Sequence[str],
    collapse_repetitions: bool = False
) -> str:
    """
    Convenience function to translate an ordered ISL gloss sequence to English.
    """
    translator = ISLToEnglishTranslator(collapse_repetitions=collapse_repetitions)
    return translator.translate(glosses)
