# SIGNOVA: Real-Time Indian Sign Language Recognition (ISLR) & NLP Translation

[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)]()
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-purple.svg)]()

Implementation based on the research paper:  
**"Advanced gesture recognition in Indian sign language using a synergistic combination of YOLOv10 with Swin Transformer model"** (*Scientific Reports / Nature Portfolio, 2025*).

---

## Architecture Overview

1. **Orientation- & Scale-Invariant 3D Palm-Normal Feature Extractor:**
   - Projects 3D hand landmarks into rotation- and seating-invariant Palm Normal vectors $(n_x, n_y, n_z)$.
   - Employs 3D inter-joint distance ratios ($d_{\text{tip-mcp}} / d_{\text{pip-mcp}}$) to eliminate 2D foreshortening when fingers point directly at the camera/chest.
   - Generates a dense 160-dimensional normalized feature descriptor per frame.

2. **Swin-Mish Neuro-Symbolic Attention Network:**
   - Hierarchical multi-head self-attention network using **Mish activations** throughout.
   - Trained across all 75 dataset takes (3 takes across 25 ISL classes) with 3D landmark jitter augmentation (**97.5% classification accuracy**).

3. **Gentle Bayesian Plausibility Filter:**
   - Enforces physical consistency (damping single-finger numbers when 2 hands are raised, damping waist signs at mouth level) without artificial heuristic overrides.
   - Resolves subtle sign distinctions (`Help` vs `Milk`, `Eat` vs `Food`, `Drink` vs `Water`).

4. **Fast Anti-Spurious Consensus Hold (`window_history`):**
   - 4-frame consensus window requiring **3 agreeing frames (~0.10s confirmation)** for snappy recognition.
   - Motion velocity gating (`max_motion_thresh = 0.038`) locks out intermediate arm transit swings between signs.

5. **NLP Grammar & Sentence Translation Engine:**
   - Translates raw ISL keyword sequences (SOV order) into natural, grammatically fluent English sentences.
   - Restores auxiliary verbs, subject-verb agreement, idiomatic requests, and polite queries.
   - Built-in background **Voice Text-to-Speech (TTS)**:
     - **macOS**: Native Apple speech synthesis via `/usr/bin/say`.
     - **Windows / Linux**: `pyttsx3` background synthesis.

---

## Vocabulary (25 Classes)

| Category | Words / Signs |
| :--- | :--- |
| **Numbers (1–10)** | `1_One`, `2_Two`, `3_Three`, `4_Four`, `5_Five`, `6_Six`, `7_Seven`, `8_Eight`, `9_Nine`, `10_Ten` |
| **Needs, Food & Desires** | `Want_Desire_Wish_For`, `Eat`, `Drink`, `Food`, `Water`, `Milk` |
| **Actions & Requests** | `Give`, `Go`, `Help` |
| **Pronouns** | `I_Me`, `My`, `You`, `We` |
| **Questions** | `What`, `Where` |

---

## Project Structure

```text
SIGNOVA/
├── models/
│   ├── hand_feature_extractor.py       # 160-dim invariant 3D normal & joint ratio extractor
│   ├── gesture_st_classifier.pt        # Trained Swin-Mish Attention weights (97.5% acc)
│   ├── yolov10_st_modules.py           # Swin Transformer & Mish fusion blocks
│   └── build_yolov10_st.py             # YOLOv10-ST architecture builder
├── scripts/
│   ├── realtime_isl_translator.py      # Core live webcam translation engine
│   ├── sign_nlp_sentence_generator.py  # ISL -> English NLP sentence engine & TTS
│   ├── train_advanced_classifier.py    # Swin-Mish Attention model training script
│   └── test_trained_model.py           # Video file test evaluation
├── dataset_raw/                        # 75 raw video takes for all 25 classes
├── dataset/                            # Processed YOLO dataset (data.yaml)
├── hand_landmarker.task                # MediaPipe HandLandmarker model asset
├── requirements.txt                    # Cross-platform dependencies
├── START_SIGNOVA.sh                    # macOS / Linux Master Launcher
├── run_webcam.sh                       # macOS / Linux Live Webcam Launcher
├── run_dashboard.sh                    # macOS / Linux Web Dashboard Launcher
├── START_SIGNOVA.bat                   # Windows Master Launcher
├── run_webcam.bat                      # Windows Live Webcam Launcher
└── run_dashboard.bat                   # Windows Web Dashboard Launcher
```

---

## Quickstart Guide

### 🍎 For macOS Users (Apple Silicon M1/M2/M3/M4 & Intel)

1. **Clone the repository:**
   ```bash
   git clone -b SIGNOVA-latest https://github.com/anshultyagii/SIGNOVA-Model.git
   cd SIGNOVA-Model
   ```

2. **Create a virtual environment & install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Make launcher scripts executable & start:**
   ```bash
   chmod +x *.sh
   ./run_webcam.sh
   ```
   *or launch the master menu:*
   ```bash
   ./START_SIGNOVA.sh
   ```

> **macOS Camera Permission Tip:** When prompted, allow **Terminal** (or your IDE) to access the Camera in:  
> *System Settings $\rightarrow$ Privacy & Security $\rightarrow$ Camera*.

---

### 🪟 For Windows Users

1. **Clone and open repository:**
   ```cmd
   git clone -b SIGNOVA-latest https://github.com/anshultyagii/SIGNOVA-Model.git
   cd SIGNOVA-Model
   ```

2. **Install dependencies:**
   ```cmd
   py -3.12 -m pip install -r requirements.txt
   ```

3. **Launch:**
   Double-click `run_webcam.bat` or `START_SIGNOVA.bat`.

---

## Spacebar Session Controls

| Key | Action | Description |
| :--- | :--- | :--- |
| **`[Spacebar]`** | **Toggle Session** | Tap once to **Start Recording** (`● REC ACTIVE`). Perform signs steadily in sequence. Tap **`[Spacebar]`** again to **Finish**: automatically generates the NLP English sentence and speaks it aloud via Voice TTS. |
| **`[Backspace]`** / **`[z]`** | **Instant Undo** | Deletes the last captured sign if you made a mistake or want to re-sign. |
| **`[m]`** | **Mirror Mode** | Toggles horizontal flip for natural mirror feedback. |
| **`[v]`** | **Voice Toggle** | Toggles audio Text-to-Speech narration ON/OFF. |
| **`[c]`** | **Clear** | Clears the session sign buffer and sentence. |
| **`[q]`** | **Quit** | Exits the application cleanly. |
