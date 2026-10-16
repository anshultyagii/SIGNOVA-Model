# -*- coding: utf-8 -*-
"""
SignSpeak AI - ISL Gloss to English Dataset Loader & Preprocessing
==================================================================
Provides structured dataset loading, deterministic train/val splitting with zero leakage,
and PyTorch Dataset wrapping for T5 Seq2Seq fine-tuning.
"""

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "dataset",
    "isl_gloss_to_english.json"
)

PROMPT_PREFIX = "translate ISL gloss to English: "


def load_isl_gloss_dataset(file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads authentic paired ISL gloss-to-English examples from JSON.
    Each item contains:
      - 'id': str
      - 'glosses': List[str]
      - 'gloss_text': str
      - 'english_text': str
      - 'category': str
    """
    path = file_path or DEFAULT_DATASET_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of examples in {path}, got {type(data)}")

    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue
        gloss_text = str(item.get("gloss_text", "")).strip()
        english_text = str(item.get("english_text", "")).strip()
        if gloss_text and english_text:
            cleaned.append({
                "id": str(item.get("id", f"pair_{len(cleaned)+1:03d}")),
                "glosses": list(item.get("glosses", gloss_text.split())),
                "gloss_text": gloss_text,
                "english_text": english_text,
                "category": str(item.get("category", "general")),
                "synthetic": bool(item.get("synthetic", False)),
                "provenance": str(item.get("provenance", "authentic" if not item.get("synthetic") else "synthetic_rule_expansion")),
            })
    return cleaned


def split_dataset(
    data: List[Dict[str, Any]],
    test_ratio: float = 0.20,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits dataset into train and test/validation sets deterministically.
    Guarantees:
      - Fixed random seed (seed=42) for reproducibility.
      - Disjoint gloss sequences: no gloss sequence in test set appears in train set (zero leakage).
    """
    if not data:
        return [], []

    # Copy and shuffle with deterministic seed
    rng = random.Random(seed)
    shuffled = list(data)
    rng.shuffle(shuffled)

    n_test = max(1, int(len(shuffled) * test_ratio))
    test_set = shuffled[:n_test]
    train_set = shuffled[n_test:]

    # Assert zero data leakage
    train_glosses = {item["gloss_text"] for item in train_set}
    test_glosses = {item["gloss_text"] for item in test_set}
    leakage = train_glosses.intersection(test_glosses)
    if leakage:
        raise ValueError(f"Data leakage detected! Overlapping glosses: {leakage}")

    return train_set, test_set


class ISLGlossDataset(Dataset):
    """
    PyTorch Dataset wrapping tokenized ISL gloss inputs and English target sequences.
    Appends the standard T5 task prefix 'translate ISL gloss to English: '.
    """

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_source_length: int = 64,
        max_target_length: int = 64,
        prefix: str = PROMPT_PREFIX
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.prefix = prefix

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]
        source_text = self.prefix + item["gloss_text"]
        target_text = item["english_text"]

        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = target_encoding["input_ids"].squeeze(0)
        # Replace padding token id with -100 so cross-entropy loss ignores padding
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_encoding["input_ids"].squeeze(0),
            "attention_mask": source_encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }
