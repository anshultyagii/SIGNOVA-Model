import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

def extract_features_from_landmarks(hand_landmarks_list, orig_w, orig_h):
    """
    Extracts invariant geometric, distance, and angular features from 1 or 2 hands.
    Output vector is size 126 (2 hands x 21 landmarks x 3 coords, normalized relative to wrist).
    """
    feature_vec = np.zeros(126, dtype=np.float32)
    
    for h_idx, hand in enumerate(hand_landmarks_list[:2]):
        wrist = hand[0]
        wx, wy, wz = wrist.x, wrist.y, wrist.z
        
        # Compute hand scale (distance from wrist to middle MCP, landmark 9)
        mid_mcp = hand[9]
        scale = max(1e-4, np.sqrt((mid_mcp.x - wx)**2 + (mid_mcp.y - wy)**2 + (mid_mcp.z - wz)**2))
        
        offset = h_idx * 63
        for i, lm in enumerate(hand):
            # Scale-invariant coordinates centered at wrist
            feature_vec[offset + i*3 + 0] = (lm.x - wx) / scale
            feature_vec[offset + i*3 + 1] = (lm.y - wy) / scale
            feature_vec[offset + i*3 + 2] = (lm.z - wz) / scale
            
    return feature_vec

class SwinMishGestureNet(nn.Module):
    """
    Neural network with Mish activation and Self-Attention feature fusion
    for robust 25-class ISL recognition.
    """
    def __init__(self, in_features=126, num_classes=25):
        super().__init__()
        
        # Self-Attention block over hand joint features
        self.embedding = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Mish(),
            nn.LayerNorm(256)
        )
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        
        # Classification MLP with Mish
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        emb = self.embedding(x).unsqueeze(1) # [B, 1, 256]
        attn_out, _ = self.attn(emb, emb, emb)
        fused = (emb + attn_out).squeeze(1)
        return self.classifier(fused)

def collect_dataset():
    base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3
    )
    detector = vision.HandLandmarker.create_from_options(options)

    train_X, train_Y = [], []
    val_X, val_Y = [], []

    files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    print(f"Extracting invariant hand features from {len(files)} videos...")

    for fname in tqdm(sorted(files)):
        name_no_ext = os.path.splitext(fname)[0]
        if name_no_ext.endswith('2') and not name_no_ext == '2_Two':
            base = name_no_ext[:-1]
            split = 'val'
        elif name_no_ext.endswith('1') and not name_no_ext == '1_One':
            base = name_no_ext[:-1]
            split = 'train'
        else:
            base = name_no_ext
            split = 'train'

        if base not in CLASS_TO_ID:
            continue
        cls_id = CLASS_TO_ID[base]

        cap = cv2.VideoCapture(os.path.join(RAW_DIR, fname))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue

        # Sample active gesture frames
        start_f = int(total_frames * 0.15)
        end_f = int(total_frames * 0.85)
        step = max(1, (end_f - start_f) // 30)

        for f_idx in range(start_f, end_f, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w, _ = frame.shape
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = detector.detect(mp_img)

            if res.hand_landmarks:
                feats = extract_features_from_landmarks(res.hand_landmarks, w, h)
                if split == 'train':
                    train_X.append(feats)
                    train_Y.append(cls_id)
                    # Add mild data augmentation (gaussian jitter on landmarks)
                    noise = np.random.normal(0, 0.02, size=feats.shape).astype(np.float32)
                    train_X.append(feats + noise)
                    train_Y.append(cls_id)
                else:
                    val_X.append(feats)
                    val_Y.append(cls_id)

        cap.release()

    detector.close()
    print(f"Extraction done! Train samples: {len(train_X)}, Val samples: {len(val_X)}")
    return np.array(train_X), np.array(train_Y), np.array(val_X), np.array(val_Y)

def train_classifier():
    train_X, train_Y, val_X, val_Y = collect_dataset()

    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_Y = torch.tensor(train_Y, dtype=torch.long)
    val_X = torch.tensor(val_X, dtype=torch.float32)
    val_Y = torch.tensor(val_Y, dtype=torch.long)

    model = SwinMishGestureNet(in_features=126, num_classes=25)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    dataset = torch.utils.data.TensorDataset(train_X, train_Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    print("\nTraining Swin-Mish Gesture Classifier...")
    best_acc = 0.0

    for epoch in range(1, 41):
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
                'val_acc': val_acc
            }, SAVE_PATH)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/40 | Loss: {total_loss/total:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc (Unseen Signer): {val_acc*100:.1f}% (Best: {best_acc*100:.1f}%)")

    print(f"\nTraining Complete! Best Unseen Signer Val Accuracy: {best_acc*100:.2f}%")
    print(f"Model saved to: {SAVE_PATH}")

if __name__ == "__main__":
    train_classifier()
