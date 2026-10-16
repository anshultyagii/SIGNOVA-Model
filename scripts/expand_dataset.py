# -*- coding: utf-8 -*-
"""
SignSpeak AI - Expand Synthetic Translation Dataset
===================================================
Expands dataset/isl_gloss_to_english.json with valid ISL grammatical structures
(SOV canonical orders, questions, and numeral-noun phrases) strictly drawn from the
25 enrolled vocabulary classes.

Each synthetic entry is explicitly tagged:
  "synthetic": true
  "provenance": "synthetic_rule_expansion"

Guarantees 100% unique gloss_text keys (zero leakage across train/test splits).
"""

import json
import os
from typing import Dict, List
from src.translation.isl_to_english import ALL_25_CLASSES

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset",
    "isl_gloss_to_english.json"
)

NEW_SYNTHETIC_CANDIDATES = [
    # ── Canonical Subject-Object-Verb (SOV) Constructions ──────────
    {
        "glosses": ["I_Me", "Water", "Want_Desire_Wish_For"],
        "english_text": "I want water.",
        "category": "sov_canonical"
    },
    {
        "glosses": ["I_Me", "Food", "Want_Desire_Wish_For"],
        "english_text": "I want food.",
        "category": "sov_canonical"
    },
    {
        "glosses": ["I_Me", "Milk", "Want_Desire_Wish_For"],
        "english_text": "I want milk.",
        "category": "sov_canonical"
    },
    {
        "glosses": ["You", "Water", "Want_Desire_Wish_For"],
        "english_text": "Do you want water?",
        "category": "sov_question"
    },
    {
        "glosses": ["You", "Food", "Want_Desire_Wish_For"],
        "english_text": "Do you want food?",
        "category": "sov_question"
    },
    {
        "glosses": ["You", "Milk", "Want_Desire_Wish_For"],
        "english_text": "Do you want milk?",
        "category": "sov_question"
    },
    {
        "glosses": ["We", "Water", "Want_Desire_Wish_For"],
        "english_text": "We want water.",
        "category": "sov_canonical"
    },
    {
        "glosses": ["We", "Food", "Want_Desire_Wish_For"],
        "english_text": "We want food.",
        "category": "sov_canonical"
    },
    {
        "glosses": ["We", "Milk", "Want_Desire_Wish_For"],
        "english_text": "We want milk.",
        "category": "sov_canonical"
    },
    {
        "glosses": ["I_Me", "Food", "Eat"],
        "english_text": "I eat food.",
        "category": "sov_canonical"
    },
    {
        "glosses": ["You", "Food", "Eat"],
        "english_text": "You eat food.",
        "category": "sov_canonical"
    },
    {
        "glosses": ["We", "Food", "Eat"],
        "english_text": "We eat food.",
        "category": "sov_canonical"
    },
    {
        "glosses": ["I_Me", "Water", "Drink"],
        "english_text": "I drink water.",
        "category": "sov_canonical"
    },
    {
        "glosses": ["You", "Water", "Drink"],
        "english_text": "You drink water.",
        "category": "sov_canonical"
    },
    {
        "glosses": ["We", "Water", "Drink"],
        "english_text": "We drink water.",
        "category": "sov_canonical"
    },
    {
        "glosses": ["I_Me", "Milk", "Drink"],
        "english_text": "I drink milk.",
        "category": "sov_canonical"
    },
    {
        "glosses": ["You", "Milk", "Drink"],
        "english_text": "You drink milk.",
        "category": "sov_canonical"
    },
    {
        "glosses": ["We", "Milk", "Drink"],
        "english_text": "We drink milk.",
        "category": "sov_canonical"
    },
    {
        "glosses": ["You", "Food", "Give"],
        "english_text": "Please give food.",
        "category": "sov_imperative"
    },
    {
        "glosses": ["You", "Water", "Give"],
        "english_text": "Please give water.",
        "category": "sov_imperative"
    },
    {
        "glosses": ["You", "Milk", "Give"],
        "english_text": "Please give milk.",
        "category": "sov_imperative"
    },
    {
        "glosses": ["I_Me", "Food", "Give"],
        "english_text": "I will give food.",
        "category": "sov_canonical"
    },
    {
        "glosses": ["I_Me", "Water", "Give"],
        "english_text": "I will give water.",
        "category": "sov_canonical"
    },
    {
        "glosses": ["I_Me", "Milk", "Give"],
        "english_text": "I will give milk.",
        "category": "sov_canonical"
    },
    {
        "glosses": ["We", "Food", "Give"],
        "english_text": "We will give food.",
        "category": "sov_canonical"
    },
    {
        "glosses": ["We", "Water", "Give"],
        "english_text": "We will give water.",
        "category": "sov_canonical"
    },
    {
        "glosses": ["We", "Milk", "Give"],
        "english_text": "We will give milk.",
        "category": "sov_canonical"
    },

    # ── Topic-Wh Question Constructions (Typical ISL Syntax) ──────
    {
        "glosses": ["Water", "Where"],
        "english_text": "Where is the water?",
        "category": "question_topic_wh"
    },
    {
        "glosses": ["Food", "Where"],
        "english_text": "Where is the food?",
        "category": "question_topic_wh"
    },
    {
        "glosses": ["Milk", "Where"],
        "english_text": "Where is the milk?",
        "category": "question_topic_wh"
    },
    {
        "glosses": ["You", "Where", "Go"],
        "english_text": "Where are you going?",
        "category": "question_wh"
    },
    {
        "glosses": ["I_Me", "Where", "Go"],
        "english_text": "Where should I go?",
        "category": "question_wh"
    },
    {
        "glosses": ["We", "Where", "Go"],
        "english_text": "Where should we go?",
        "category": "question_wh"
    },
    {
        "glosses": ["You", "What", "Want_Desire_Wish_For"],
        "english_text": "What do you want?",
        "category": "question_wh"
    },
    {
        "glosses": ["You", "What", "Eat"],
        "english_text": "What do you eat?",
        "category": "question_wh"
    },
    {
        "glosses": ["You", "What", "Drink"],
        "english_text": "What do you drink?",
        "category": "question_wh"
    },
    {
        "glosses": ["Water", "What"],
        "english_text": "What is that water?",
        "category": "question_topic_wh"
    },
    {
        "glosses": ["Food", "What"],
        "english_text": "What is that food?",
        "category": "question_topic_wh"
    },
    {
        "glosses": ["Milk", "What"],
        "english_text": "What is that milk?",
        "category": "question_topic_wh"
    },
    {
        "glosses": ["Where", "My", "Food"],
        "english_text": "Where is my food?",
        "category": "possessive_question"
    },
    {
        "glosses": ["Where", "My", "Water"],
        "english_text": "Where is my water?",
        "category": "possessive_question"
    },
    {
        "glosses": ["Where", "My", "Milk"],
        "english_text": "Where is my milk?",
        "category": "possessive_question"
    },
    {
        "glosses": ["My", "Food", "Where"],
        "english_text": "Where is my food?",
        "category": "possessive_question"
    },
    {
        "glosses": ["My", "Water", "Where"],
        "english_text": "Where is my water?",
        "category": "possessive_question"
    },
    {
        "glosses": ["My", "Milk", "Where"],
        "english_text": "Where is my milk?",
        "category": "possessive_question"
    },
    {
        "glosses": ["I_Me", "My", "Food", "Eat"],
        "english_text": "I eat my food.",
        "category": "possessive_action"
    },
    {
        "glosses": ["I_Me", "My", "Water", "Drink"],
        "english_text": "I drink my water.",
        "category": "possessive_action"
    },
    {
        "glosses": ["I_Me", "My", "Milk", "Drink"],
        "english_text": "I drink my milk.",
        "category": "possessive_action"
    },

    # ── Complete Numeral + Noun Combinations (Digits 1 to 10) ─────
    {
        "glosses": ["One", "Food"],
        "english_text": "One food.",
        "category": "digit_noun"
    },
    {
        "glosses": ["One", "Milk"],
        "english_text": "One milk.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Two", "Water"],
        "english_text": "Two waters.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Two", "Milk"],
        "english_text": "Two milks.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Three", "Water"],
        "english_text": "Three waters.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Three", "Food"],
        "english_text": "Three foods.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Four", "Food"],
        "english_text": "Four foods.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Four", "Milk"],
        "english_text": "Four milks.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Five", "Water"],
        "english_text": "Five waters.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Five", "Milk"],
        "english_text": "Five milks.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Six", "Water"],
        "english_text": "Six waters.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Six", "Food"],
        "english_text": "Six foods.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Six", "Milk"],
        "english_text": "Six milks.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Seven", "Water"],
        "english_text": "Seven waters.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Seven", "Food"],
        "english_text": "Seven foods.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Seven", "Milk"],
        "english_text": "Seven milks.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Eight", "Water"],
        "english_text": "Eight waters.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Eight", "Food"],
        "english_text": "Eight foods.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Eight", "Milk"],
        "english_text": "Eight milks.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Nine", "Water"],
        "english_text": "Nine waters.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Nine", "Food"],
        "english_text": "Nine foods.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Nine", "Milk"],
        "english_text": "Nine milks.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Ten", "Water"],
        "english_text": "Ten waters.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Ten", "Food"],
        "english_text": "Ten foods.",
        "category": "digit_noun"
    },
    {
        "glosses": ["Ten", "Milk"],
        "english_text": "Ten milks.",
        "category": "digit_noun"
    },

    # ── Numeral SOV and Imperatives ────────────────────────────────
    {
        "glosses": ["I_Me", "One", "Water", "Want_Desire_Wish_For"],
        "english_text": "I want one water.",
        "category": "numeral_sov"
    },
    {
        "glosses": ["I_Me", "Two", "Food", "Want_Desire_Wish_For"],
        "english_text": "I want two foods.",
        "category": "numeral_sov"
    },
    {
        "glosses": ["I_Me", "Three", "Milk", "Want_Desire_Wish_For"],
        "english_text": "I want three milks.",
        "category": "numeral_sov"
    },
    {
        "glosses": ["You", "One", "Water", "Want_Desire_Wish_For"],
        "english_text": "Do you want one water?",
        "category": "numeral_sov"
    },
    {
        "glosses": ["You", "Two", "Food", "Want_Desire_Wish_For"],
        "english_text": "Do you want two foods?",
        "category": "numeral_sov"
    },
    {
        "glosses": ["Give", "I_Me", "One", "Water"],
        "english_text": "Give me one water.",
        "category": "numeral_imperative"
    },
    {
        "glosses": ["Give", "I_Me", "Two", "Food"],
        "english_text": "Give me two foods.",
        "category": "numeral_imperative"
    },
    {
        "glosses": ["Give", "I_Me", "Three", "Milk"],
        "english_text": "Give me three milks.",
        "category": "numeral_imperative"
    },
    {
        "glosses": ["I_Me", "One", "Food", "Eat"],
        "english_text": "I eat one food.",
        "category": "numeral_action"
    },
    {
        "glosses": ["I_Me", "Two", "Water", "Drink"],
        "english_text": "I drink two waters.",
        "category": "numeral_action"
    },
]


def expand_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        existing_data = json.load(f)

    # Standardize existing items
    existing_gloss_map = {}
    normalized_existing = []
    for item in existing_data:
        gloss_text = " ".join(item.get("glosses", item.get("gloss_text", "").split()))
        item["gloss_text"] = gloss_text
        if "synthetic" not in item:
            item["synthetic"] = False
        if "provenance" not in item:
            item["provenance"] = "authentic"
        existing_gloss_map[gloss_text] = item
        normalized_existing.append(item)

    added_count = 0
    next_id = len(normalized_existing) + 1

    for cand in NEW_SYNTHETIC_CANDIDATES:
        gloss_text = " ".join(cand["glosses"])
        # Validate that all glosses are enrolled in 25 classes
        for g in cand["glosses"]:
            if g not in ALL_25_CLASSES:
                raise ValueError(f"Gloss '{g}' is not in enrolled 25 classes!")

        # Skip duplicates to maintain 100% unique sequence keys
        if gloss_text in existing_gloss_map:
            print(f"Skipping duplicate: {gloss_text}")
            continue

        item = {
            "id": f"pair_synth_{next_id:03d}",
            "glosses": cand["glosses"],
            "gloss_text": gloss_text,
            "english_text": cand["english_text"],
            "category": cand["category"],
            "synthetic": True,
            "provenance": "synthetic_rule_expansion",
        }
        existing_gloss_map[gloss_text] = item
        normalized_existing.append(item)
        next_id += 1
        added_count += 1

    print(f"Total dataset after expansion: {len(normalized_existing)} pairs (+{added_count} synthetic).")

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(normalized_existing, f, indent=2)
    print(f"Saved expanded dataset to {DATASET_PATH}")


if __name__ == "__main__":
    expand_dataset()
