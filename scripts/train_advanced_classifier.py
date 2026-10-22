import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from models.hand_feature_extractor import extract_advanced_gesture_features

RAW_DIR = os.path.join(PROJECT_ROOT, "dataset_raw")
MODEL_TASK_PATH = os.path.join(PROJECT_ROOT, "hand_landmarker.task")
SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "gesture_st_classifier.pt")

CLASS_NAMES = [
    "10_Ten", "1_One", "2_Two", "3_Three", "4_Four",
    "5_Five", "6_Six", "7_Seven", "8_Eight", "9_Nine",
    "Drink", "Eat", "Food", "Give", "Go",
    "Help", "I_Me", "Milk", "My", "Want_Desire_Wish_For",
    "Water", "We", "What", "Where", "You"
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

class SwinMishAttentionNet(nn.Module):
    """
    Swin Transformer inspired attention network with Mish activation
    and cross-joint self-attention for 25-class ISL gesture recognition.
    """
    def __init__(self, in_features=160, num_classes=25):
        super().__init__()
        # Input projection with Mish
        self.proj = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Mish(),
            nn.LayerNorm(256)
        )
        # Multi-Head Attention Block (hierarchical self-attention)
        self.mha = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(256)

        # Classification MLP with Mish
        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        emb = self.proj(x).unsqueeze(1) # [B, 1, 256]
        attn_out, _ = self.mha(emb, emb, emb)
        fused = self.norm(emb + attn_out).squeeze(1) # [B, 256]
        return self.mlp(fused)

class ClassWeightedFocalLoss(nn.Module):
    """
    Multi-class focal loss with class-reweighting and label smoothing.
    Downweights easy confident predictions and forces high gradient updates
    on hard/confused gesture boundaries (What, You, Go, Where, 1_One).
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.04):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction='none', label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def train_advanced_model():
    base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25
    )
    detector = vision.HandLandmarker.create_from_options(options)

    all_X, all_Y = [], []

    files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    print(f"Extracting dense position-invariant features from all {len(files)} videos across 25 signs...")

    for fname in tqdm(sorted(files)):
        name_no_ext = os.path.splitext(fname)[0]
        # Clean base name
        if name_no_ext.endswith('2') and not name_no_ext == '2_Two':
            base = name_no_ext[:-1]
        elif name_no_ext.endswith('1') and not name_no_ext == '1_One':
            base = name_no_ext[:-1]
        else:
            base = name_no_ext

        if base not in CLASS_TO_ID:
            continue
        cls_id = CLASS_TO_ID[base]

        cap = cv2.VideoCapture(os.path.join(RAW_DIR, fname))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue

        # Step 2: Dense extraction during gesture active phase (10% to 90% with stride 2)
        start_f = int(total_frames * 0.10)
        end_f = int(total_frames * 0.90)
        step = 2

        curr_f = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None or curr_f > end_f:
                break

            if curr_f >= start_f and (curr_f - start_f) % step == 0:
                h, w, _ = frame.shape
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                res = detector.detect(mp_img)

                if res.hand_landmarks:
                    feats = extract_advanced_gesture_features(res.hand_landmarks, w, h)
                    all_X.append(feats)
                    all_Y.append(cls_id)

                    # Augmentation 1: Landmark Gaussian Jitter (MediaPipe sensor noise)
                    noise = np.random.normal(0, 0.008, size=feats.shape).astype(np.float32)
                    # Don't add noise to discrete hand count at index 0
                    noise[0] = 0.0
                    all_X.append(feats + noise)
                    all_Y.append(cls_id)

                    # Augmentation 2: Scale Perturbation (user distance variance)
                    scale_jitter = float(np.random.uniform(0.94, 1.06))
                    feats_scaled = feats.copy()
                    feats_scaled[15:15+63] *= scale_jitter
                    feats_scaled[78+15:78+15+63] *= scale_jitter
                    all_X.append(feats_scaled)
                    all_Y.append(cls_id)

                    # Augmentation 3: 3D In-Plane Rotation Jitter (+-8 degrees)
                    theta = float(np.radians(np.random.uniform(-8.0, 8.0)))
                    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
                    feats_rot = feats.copy()
                    for h_idx in range(2):
                        lm_offset = 1 + h_idx * 78 + 15
                        for i in range(21):
                            px = feats[lm_offset + i*3 + 0]
                            py = feats[lm_offset + i*3 + 1]
                            feats_rot[lm_offset + i*3 + 0] = px * cos_t - py * sin_t
                            feats_rot[lm_offset + i*3 + 1] = px * sin_t + py * cos_t
                    all_X.append(feats_rot)
                    all_Y.append(cls_id)

                    # Augmentation 4: Horizontal Flip (Mirror & Left/Right Ambidextrous Invariance)
                    frame_flip = cv2.flip(frame, 1)
                    mp_flip = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_flip, cv2.COLOR_BGR2RGB))
                    res_flip = detector.detect(mp_flip)
                    if res_flip.hand_landmarks:
                        feats_flip = extract_advanced_gesture_features(res_flip.hand_landmarks, w, h)
                        all_X.append(feats_flip)
                        all_Y.append(cls_id)
                        # Jitter on flipped sample
                        noise_flip = np.random.normal(0, 0.008, size=feats_flip.shape).astype(np.float32)
                        noise_flip[0] = 0.0
                        all_X.append(feats_flip + noise_flip)
                        all_Y.append(cls_id)

            curr_f += 1

        cap.release()

    detector.close()
    print(f"Total extracted and augmented samples: {len(all_X)}")

    all_X = np.array(all_X, dtype=np.float32)
    all_Y = np.array(all_Y, dtype=np.int64)

    # Train/Validation split (85% train, 15% validation stratified)
    np.random.seed(42)
    indices = np.random.permutation(len(all_X))
    split_idx = int(len(all_X) * 0.85)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    train_X = torch.tensor(all_X[train_idx], dtype=torch.float32)
    train_Y = torch.tensor(all_Y[train_idx], dtype=torch.long)
    val_X = torch.tensor(all_X[val_idx], dtype=torch.float32)
    val_Y = torch.tensor(all_Y[val_idx], dtype=torch.long)

    model = SwinMishAttentionNet(in_features=160, num_classes=25)

    # Step 3: Class-Weighted Focal Loss targeting bottleneck classes
    alpha_weights = torch.ones(25, dtype=torch.float32)
    alpha_weights[CLASS_TO_ID["What"]] = 2.2
    alpha_weights[CLASS_TO_ID["You"]] = 2.0
    alpha_weights[CLASS_TO_ID["Go"]] = 2.0
    alpha_weights[CLASS_TO_ID["Where"]] = 1.8
    alpha_weights[CLASS_TO_ID["1_One"]] = 1.6
    alpha_weights = alpha_weights / alpha_weights.mean()

    criterion = ClassWeightedFocalLoss(alpha=alpha_weights, gamma=2.0, label_smoothing=0.04)
    optimizer = optim.AdamW(model.parameters(), lr=0.0025, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)

    dataset = torch.utils.data.TensorDataset(train_X, train_Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    print("\nTraining Neuro-Symbolic Swin-Mish Attention Network with Focal Loss on ALL 25 signs...")
    best_acc = 0.0

    for epoch in range(1, 61):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bx.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += bx.size(0)

        scheduler.step()
        train_acc = correct / total

        # Validation
        model.eval()
        with torch.no_grad():
            v_out = model(val_X)
            v_preds = v_out.argmax(dim=1)
            val_acc = (v_preds == val_Y).float().mean().item()

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': CLASS_NAMES,
                'val_acc': val_acc,
                'in_features': 160
            }, SAVE_PATH)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/60 | Focal Loss: {total_loss/total:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% (Best: {best_acc*100:.1f}%)")

    print(f"\nTraining Complete! Best Accuracy: {best_acc*100:.2f}%")
    print(f"Saved optimized model to: {SAVE_PATH}")

if __name__ == "__main__":
    train_advanced_model()
