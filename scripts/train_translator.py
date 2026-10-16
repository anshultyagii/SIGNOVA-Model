# -*- coding: utf-8 -*-
"""
SignSpeak AI - CLI Script to Train / Fine-Tune ISL T5 Translation Model
======================================================================
Usage:
    python scripts/train_translator.py [--epochs 15] [--lr 0.0005] [--output-dir models/isl_t5_translator]
"""

import argparse
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.translation.train import train_translator, DEFAULT_OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description="Fine-tune T5-small for ISL Gloss to English Translation")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 0.0005)")
    parser.add_argument("--model-name", type=str, default="t5-small", help="Base model name (default: t5-small)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Checkpoint output directory")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to ISL gloss dataset JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    metadata = train_translator(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        base_model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed
    )

    metrics = metadata.get("metrics", {})
    print("\n" + "=" * 60)
    print("TRAINING & EVALUATION SUMMARY:")
    print(f"  Final Loss   : {metrics.get('final_loss')}")
    print(f"  Exact Match  : {metrics.get('exact_match', 0) * 100:.1f}%")
    print(f"  BLEU Score   : {metrics.get('bleu')}")
    print(f"  ROUGE-L      : {metrics.get('rouge_l')}")
    print(f"  Avg Latency  : {metrics.get('avg_inference_ms')} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
