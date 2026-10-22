"""
Train Specialized ISL T5-Small Neural Seq2Seq Gloss-to-Text Model
Optimized for Indian Sign Language (ISLR) 25-Sign Vocabulary.
Includes Denoising Objective for Spurious Transit Invariance.
"""

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from scripts.generate_isl_nlp_dataset import generate_isl_dataset

MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "isl_t5_translator")

class ISLGlossDataset(Dataset):
    def __init__(self, data, tokenizer, max_in_len=32, max_tgt_len=32):
        self.tokenizer = tokenizer
        self.inputs = [f"translate ISL to English: {inp.upper()}" for inp, _ in data]
        self.targets = [tgt for _, tgt in data]
        self.max_in_len = max_in_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inp_text = self.inputs[idx]
        tgt_text = self.targets[idx]

        in_enc = self.tokenizer(
            inp_text,
            max_length=self.max_in_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        tgt_enc = self.tokenizer(
            tgt_text,
            max_length=self.max_tgt_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = tgt_enc.input_ids.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": in_enc.input_ids.squeeze(0),
            "attention_mask": in_enc.attention_mask.squeeze(0),
            "labels": labels
        }

def train_isl_t5(epochs=6, batch_size=8, lr=4e-4):
    print("=" * 65)
    print("      TRAINING SOTA ISL T5-SMALL NEURAL TRANSLATION MODEL")
    print("=" * 65)

    device = torch.device("cpu")
    base_model_name = "t5-small"
    print(f"Loading pre-trained base model: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to(device)

    # Generate training data with data augmentation
    raw_data = generate_isl_dataset()
    random.seed(42)
    random.shuffle(raw_data)

    split = int(0.9 * len(raw_data))
    train_data = raw_data[:split]
    val_data = raw_data[split:]

    print(f"Dataset summary: {len(raw_data)} total pairs ({len(train_data)} train, {len(val_data)} val)")

    train_ds = ISLGlossDataset(train_data, tokenizer)
    val_ds = ISLGlossDataset(val_data, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save fine-tuned model and tokenizer
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    print(f"\nSaving fine-tuned ISL T5 model to: {MODEL_SAVE_DIR}")
    model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    print("Model saved successfully!")

    # Run Verification on user's problematic scenarios
    print("\n" + "=" * 65)
    print("          MODEL VERIFICATION ON ISL TEST SEQUENCES")
    print("=" * 65)
    model.eval()

    test_cases = [
        ("I WANT EAT FOOD", "I want to eat food."),
        ("I WHAT WANT EAT FOOD", "I want to eat food. (Denoising transit 'WHAT')"),
        ("I WANT WHAT EAT FOOD", "I want to eat food. (Denoising transit 'WHAT')"),
        ("WHERE GO YOU", "Where are you going?"),
        ("WHAT YOU WANT", "What do you want?"),
        ("GIVE TWO WATER", "Please give me two glasses of water."),
        ("GIVE FOOD", "Please give me some food."),
        ("HELP I", "Please help me!"),
        ("YOU HELP I", "Can you please help me?"),
        ("MY MILK", "This is my milk."),
        ("WE GO", "We are going.")
    ]

    for inp, expected in test_cases:
        prompt = f"translate ISL to English: {inp}"
        in_tokens = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outs = model.generate(
                **in_tokens,
                max_new_tokens=32,
                num_beams=2,
                early_stopping=True
            )
        pred = tokenizer.decode(outs[0], skip_special_tokens=True)
        print(f"Gloss: {inp:26s} -> Output: \"{pred}\" (Expected: \"{expected}\")")

if __name__ == "__main__":
    train_isl_t5(epochs=6, batch_size=8, lr=5e-4)
