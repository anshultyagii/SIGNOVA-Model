# -*- coding: utf-8 -*-
"""
SignSpeak AI - T5 Translation Fine-Tuning Pipeline
==================================================
Fine-tunes a pretrained Seq2Seq T5 model on authentic ISL gloss-to-English pairs.
Saves model checkpoints, tokenizer, training curves, and held-out test evaluations.
"""

import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.translation.dataset import (
    DEFAULT_DATASET_PATH,
    ISLGlossDataset,
    PROMPT_PREFIX,
    load_isl_gloss_dataset,
    split_dataset,
)
from src.translation.metrics import evaluate_predictions, generate_qualitative_review

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "isl_t5_translator"
)


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_translator(
    dataset_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    base_model_name: str = "t5-small",
    epochs: int = 15,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    test_ratio: float = 0.20,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes the complete fine-tuning pipeline:
      1. Loads genuine ISL gloss-to-English dataset.
      2. Deterministically splits into train and validation/test (zero leakage).
      3. Fine-tunes T5 model with AdamW optimizer.
      4. Evaluates held-out test set with BLEU, ROUGE-L, and Exact Match.
      5. Saves the fine-tuned model checkpoint and tokenizer.
    """
    set_seed(seed)

    target_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(target_dir, exist_ok=True)

    # Set huggingface cache inside project directory
    hf_cache = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".cache",
        "huggingface"
    )
    os.environ["HF_HOME"] = hf_cache

    # Select compute device (CPU compatible)
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    print(f"[Train] Initializing ISL-to-English T5 fine-tuning pipeline...")
    print(f"[Train] Base model: {base_model_name}")
    print(f"[Train] Target device: {target_device}")
    print(f"[Train] Output directory: {target_dir}")

    # 1. Load and split dataset
    all_data = load_isl_gloss_dataset(dataset_path)
    train_data, test_data = split_dataset(all_data, test_ratio=test_ratio, seed=seed)
    print(f"[Train] Dataset loaded: {len(all_data)} genuine pairs ({len(train_data)} train, {len(test_data)} test)")

    # 2. Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    model.to(target_device)

    # 3. Create PyTorch datasets and dataloaders
    train_dataset = ISLGlossDataset(train_data, tokenizer, prefix=PROMPT_PREFIX)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 4. Configure optimizer and training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    epoch_losses = []
    training_start = time.time()

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        step_count = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(target_device)
            attention_mask = batch["attention_mask"].to(target_device)
            labels = batch["labels"].to(target_device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item())
            step_count += 1

        avg_loss = running_loss / max(1, step_count)
        epoch_losses.append(round(avg_loss, 4))
        print(f"[Train] Epoch {epoch:02d}/{epochs:02d} - Loss: {avg_loss:.4f}", flush=True)

    training_time = time.time() - training_start
    print(f"[Train] Fine-tuning completed in {training_time:.2f}s", flush=True)

    # 5. Held-Out Evaluation on Test Set
    print("[Train] Evaluating on held-out test set...", flush=True)
    model.eval()

    test_glosses = [item["gloss_text"] for item in test_data]
    test_references = [item["english_text"] for item in test_data]
    test_predictions = []

    inference_times = []
    with torch.no_grad():
        for gloss_str in test_glosses:
            prompt = PROMPT_PREFIX + gloss_str
            inputs = tokenizer(prompt, return_tensors="pt", max_length=64, truncation=True)
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

            t0 = time.time()
            gen_ids = model.generate(
                **inputs,
                max_length=64,
                num_beams=2,
                early_stopping=True
            )
            inference_times.append((time.time() - t0) * 1000)

            pred_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
            test_predictions.append(pred_text)

    # Compute BLEU, ROUGE-L, and Exact Match
    eval_metrics = evaluate_predictions(test_predictions, test_references)
    avg_inference_ms = float(np.mean(inference_times)) if inference_times else 0.0
    eval_metrics["avg_inference_ms"] = round(avg_inference_ms, 2)
    eval_metrics["training_time_seconds"] = round(training_time, 2)
    eval_metrics["epochs"] = epochs
    eval_metrics["train_sample_count"] = len(train_data)
    eval_metrics["test_sample_count"] = len(test_data)
    eval_metrics["final_loss"] = epoch_losses[-1] if epoch_losses else 0.0

    print(f"[Train] Held-Out Test Evaluation Results:")
    print(f"        Exact Match : {eval_metrics['exact_match'] * 100:.1f}%")
    print(f"        BLEU Score  : {eval_metrics['bleu']:.2f}")
    print(f"        ROUGE-L     : {eval_metrics['rouge_l']:.2f}")
    print(f"        Avg Latency : {eval_metrics['avg_inference_ms']:.1f}ms/sentence")

    # Qualitative review
    qualitative_review = generate_qualitative_review(
        glosses=test_glosses,
        predictions=test_predictions,
        references=test_references
    )

    # 6. Save model checkpoint and tokenizer
    print(f"[Train] Saving fine-tuned checkpoint to: {target_dir}")
    model.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)

    # Save training report metadata
    metadata = {
        "base_model": base_model_name,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "device": str(target_device),
        "metrics": eval_metrics,
        "loss_history": epoch_losses,
        "qualitative_review": qualitative_review,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "linguistic_disclaimer": (
            "These metrics evaluate lexical and string overlap on a curated ISL grammar set. "
            "They do not by themselves establish conversational fluency in Indian Sign Language."
        ),
    }

    metadata_path = os.path.join(target_dir, "training_metrics.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Train] Saved training metadata to {metadata_path}")
    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune T5 for ISL Gloss to English Translation")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to JSON dataset")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for fine-tuned checkpoint")
    parser.add_argument("--base_model", type=str, default="t5-small", help="Pretrained base model name")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_translator(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        base_model_name=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )
