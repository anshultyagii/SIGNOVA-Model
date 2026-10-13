# SignSpeak AI — Baseline Design & Feature Evaluation Report

## 1. Dataset Integrity & Inspection Findings

The 25 feature files in `dataset\processed_corrected\` were verified using `inspect_corrected_dataset.py`.

### Key Metrics
- **Classes:** 25 ISL word signs (`10_Ten`, `1_One`, `2_Two`, `3_Three`, `4_Four`, `5_Five`, `6_Six`, `7_Seven`, `8_Eight`, `9_Nine`, `Drink`, `Eat`, `Food`, `Give`, `Go`, `Help`, `I_Me`, `Milk`, `My`, `Want_Desire_Wish_For`, `Water`, `We`, `What`, `Where`, `You`).
- **Feature Dimension:** Exactly **1,629 floats per frame** across all 25 files.
  - Pose: 33 landmarks × 3 (x, y, z) = 99 features
  - Face Mesh: 468 landmarks × 3 = 1,404 features
  - Left Hand: 21 landmarks × 3 = 63 features
  - Right Hand: 21 landmarks × 3 = 63 features
- **Numerical Validity:** 100% finite values across all 2,056 total frames. **0 NaNs, 0 Infs**.
- **Sequence Lengths:**
  - Minimum: 53 frames (`2_Two`, 2.12 sec)
  - Maximum: 120 frames (`I_Me`, 4.80 sec)
  - Average: 82.2 frames (~3.29 sec at 25 FPS)
- **Active Gesture Window:**
  - Right-handed signs: 19 signs (e.g. `1_One` to `10_Ten`, `Drink`, `Eat`, `Water`, `Food`)
  - Two-handed signs: 6 signs (`Help`, `Milk`, `Want_Desire_Wish_For`, `We`, `Where`, and numbers/signs with secondary hand involvement)
  - Active signing duration ranges from 23 to 58 frames per video.
  - Detailed per-sign metrics are recorded in `dataset_summary.csv`.

---

## 2. Experimental Design: Leakage-Safe Baseline

### The Danger of Naive Splitting
- **Frame-level splitting is invalid:** Randomly splitting individual frames of continuous video into train/test sets causes extreme autocorrelation and identity leakage. Adjacent frames (40 ms apart) share signer identity, lighting, background, and head pose. A model trained this way achieves >98% accuracy simply by memorizing background and signer face features, yielding near-zero real-world generalization.
- **Single-video-per-class constraint ($N=1$):** Standard $k$-fold cross-validation or stratified train/test splitting at the video level is mathematically impossible when each class has only 1 sample.

### Recommended Leakage-Safe Baseline: Exemplar-Based Template Matching
Rather than training a high-capacity neural network or deep classifier on $N=1$ samples, the mathematically sound baseline is an **Exemplar Template Matching System**:
1. **Gallery Enrollment:** The 25 canonical ISL dictionary sequences serve as enrolled reference templates.
2. **Feature Pruning & Normalization:**
   - Normalize hand landmarks relative to their respective wrist coordinates ($L_i - L_{\text{wrist}}$) to make gestures invariant to the signer's position in the frame.
   - Retain upper-body pose (shoulders, elbows, wrists) and reference facial landmarks (nose tip, mouth center) to capture hand-to-face spatial relationships (essential for distinguishing `Eat`, `Drink`, and `Water`).
   - Omit the remaining ~1,390 dense facial mesh landmarks that introduce high-dimensional noise.
3. **Alignment & Distance Metric:**
   - **Dynamic Time Warping (DTW):** Measures non-linear temporal similarity between a query sign sequence and each of the 25 enrolled templates.
   - **Resampled Cosine/Euclidean Distance:** Uniformly resamples the active signing window to $T=30$ frames and computes distance across flattened kinematic trajectories.

---

## 3. Evaluation of Data Augmentation Techniques

### Temporal Resampling
- **Technical Appropriateness:** **Highly Appropriate.**
- **Rationale:** Human signers naturally vary signing cadence and velocity. Resampling the active gesture phase using linear or cubic spline interpolation along the time axis preserves trajectory shapes while matching fixed sequence lengths ($T$).
- **Crucial Rule:** Resampling must be restricted to the active signing segment (`active_start_frame` to `active_end_frame`). It must not interpolate between non-detected (zero) frames and active landmark frames to prevent "ghost" interpolations.

### Spatial / Landmark Perturbations
- **Technical Appropriateness:** **Appropriate with Structural Constraints.**
- **Rationale:**
  - *Wrist-relative translation & scaling ($\pm 2-5\%$):* Simulates variations in hand size and camera distance.
  - *Small Gaussian noise ($\sigma \approx 0.003-0.005$):* Simulates MediaPipe jitter on subtle finger positions.
  - *Caveat:* Independent coordinate scrambling must be avoided; anatomical bone lengths between connected joints (e.g. MCP to PIP to DIP) should be preserved.

---

## 4. Path to a Credible 25-Class Prototype

1. **Step 1: Feature Pruning & Wrist Normalization**
   - Reduce feature vector from 1,629 to ~150 features (wrist-normalized hands + key arm/head anchors).
2. **Step 2: DTW / Exemplar Distance Prototype**
   - Implement a lightweight matching module that compares any input sequence against the 25 dictionary templates.
3. **Step 3: Verification on Independent Query Signs**
   - To credibly evaluate performance, test against unseen recordings or test repetitions rather than self-evaluating on the training templates.
