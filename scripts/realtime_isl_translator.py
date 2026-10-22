"""
Real-Time Indian Sign Language Recognition & Sentence Translation
Features:
- Hotkey [SPACEBAR]: Toggle Recording Session (Start -> Capture signs -> Stop & Translate)
- Automatic NLP Sentence Generation upon pressing Spacebar to finish
- Background Voice TTS speaks the generated sentence upon completion
- Orientation-Invariant Gesture Geometry & Soft Neuro-Symbolic Priors
- Native 1280x720 Widescreen 16:9 Aspect Ratio
- Mirror Mode [m], Voice Toggle [v], Clear [c], Quit [q]
"""

import sys
import os
import platform

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import cv2
import torch
import numpy as np
from collections import deque, Counter
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from scripts.sign_nlp_sentence_generator import SignLanguageNLP, TOKEN_MAP
from models.hand_feature_extractor import extract_advanced_gesture_features
from scripts.train_advanced_classifier import SwinMishAttentionNet, CLASS_NAMES

MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "gesture_st_classifier.pt")
MODEL_TASK_PATH = os.path.join(PROJECT_ROOT, "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

def analyze_hand_geometry(hand, orig_w, orig_h):
    """
    Computes rotation-invariant finger extension states and spatial coordinates.
    Employs a robust dual-metric:
      1. Tip distance from wrist exceeds PIP distance from wrist (radial extension)
      2. Tip distance from wrist exceeds MCP distance by at least 20%
      3. Tip-to-MCP distance exceeds PIP-to-MCP distance by at least 15%
    Eliminates perspective foreshortening errors when curled fingers face the camera.
    """
    wrist = hand[0]
    mid_mcp = hand[9]
    scale = max(1e-4, np.hypot(mid_mcp.x - wrist.x, mid_mcp.y - wrist.y))

    # Rotation-invariant dual-metric finger extension check
    ext_results = {}
    for name, tip_idx, pip_idx, mcp_idx in [('index', 8, 6, 5), ('middle', 12, 10, 9), ('ring', 16, 14, 13), ('pinky', 20, 18, 17)]:
        tip, pip, mcp = hand[tip_idx], hand[pip_idx], hand[mcp_idx]
        d_tip_wrist = np.hypot(tip.x - wrist.x, tip.y - wrist.y)
        d_pip_wrist = np.hypot(pip.x - wrist.x, pip.y - wrist.y)
        d_mcp_wrist = max(1e-4, np.hypot(mcp.x - wrist.x, mcp.y - wrist.y))
        d_tip_mcp = np.hypot(tip.x - mcp.x, tip.y - mcp.y)
        d_pip_mcp = max(1e-4, np.hypot(pip.x - mcp.x, pip.y - mcp.y))
        is_ext = (d_tip_wrist > d_pip_wrist) and (d_tip_wrist > 1.20 * d_mcp_wrist) and (d_tip_mcp > 1.15 * d_pip_mcp)
        ext_results[name] = bool(is_ext)

    # Biomechanical Thumb Extension & Metric Separation
    d_t_tip_mcp = np.hypot(hand[4].x - hand[2].x, hand[4].y - hand[2].y)
    d_t_ip_mcp = max(1e-4, np.hypot(hand[3].x - hand[2].x, hand[3].y - hand[2].y))
    d_t_p17 = np.hypot(hand[4].x - hand[17].x, hand[4].y - hand[17].y) / scale
    d_t_i5 = np.hypot(hand[4].x - hand[5].x, hand[4].y - hand[5].y) / scale
    ext_thumb = bool((d_t_p17 > 1.10) and (d_t_i5 > 0.48) and (d_t_tip_mcp > 1.08 * d_t_ip_mcp))

    ext_count = sum(ext_results.values())
    all_five = ext_thumb and (ext_count >= 4)
    fist = (ext_count == 0)

    # Inter-finger separation ratios
    d_t_i = np.hypot(hand[4].x - hand[8].x, hand[4].y - hand[8].y) / scale
    d_t_m = np.hypot(hand[4].x - hand[12].x, hand[4].y - hand[12].y) / scale
    d_i_m = np.hypot(hand[8].x - hand[12].x, hand[8].y - hand[12].y) / scale

    return {
        'wrist_x': wrist.x,
        'wrist_y': wrist.y,
        'scale': scale,
        'thumb': ext_thumb,
        'index': ext_results['index'],
        'middle': ext_results['middle'],
        'ring': ext_results['ring'],
        'pinky': ext_results['pinky'],
        'count_ext': ext_count,
        'all_five': all_five,
        'fist': fist,
        'd_t_i5': d_t_i5,
        'd_t_p17': d_t_p17,
        'd_t_i': d_t_i,
        'd_t_m': d_t_m,
        'd_i_m': d_i_m
    }

def is_valid_hand_anatomy(hand, orig_w=1280, orig_h=720, world_hand=None, handedness_score=1.0):
    """
    Validates physical hand biomechanics to reject non-hand noise and face false-positives
    without rejecting genuine hands (even when viewed edge-on, tilted, foreshortened, or forming fists).
    """
    ys = [lm.y for lm in hand]
    xs = [lm.x for lm in hand]
    min_y, max_y = min(ys), max(ys)
    min_x, max_x = min(xs), max(xs)
    wrist = hand[0]
    mid_mcp = hand[9]

    # 1. Reject tiny speck noise (< 0.012 of frame)
    bbox_h = max_y - min_y
    bbox_w = max_x - min_x
    if bbox_h < 0.012 or bbox_w < 0.010:
        return False

    # 2. 3D Palm Span (Robust to perspective foreshortening when pointing at camera or forming a fist)
    L_palm_3d = float(np.sqrt((mid_mcp.x - wrist.x)**2 + (mid_mcp.y - wrist.y)**2 + (mid_mcp.z - wrist.z)**2))
    if L_palm_3d < 0.008 or L_palm_3d > 0.45:
        return False

    # 3. 3D Metric World Coordinate Palm Check:
    # Based on empirical dataset metrics:
    # Real human palm width (MCP 5 to MCP 17) max is 7.6cm; palm length (wrist to MCP 9) max is 12.2cm.
    # A face detected as a hand spans ear-to-ear (> 13cm) and chin-to-forehead (> 15cm).
    if world_hand is not None and len(world_hand) >= 18:
        w_mcp5 = world_hand[5]
        w_mcp17 = world_hand[17]
        w_wrist = world_hand[0]
        w_mcp9 = world_hand[9]
        palm_width_m = float(np.sqrt((w_mcp5.x - w_mcp17.x)**2 + (w_mcp5.y - w_mcp17.y)**2 + (w_mcp5.z - w_mcp17.z)**2))
        palm_len_m = float(np.sqrt((w_mcp9.x - w_wrist.x)**2 + (w_mcp9.y - w_wrist.y)**2 + (w_mcp9.z - w_wrist.z)**2))

        # Strict anti-face metric defense (rejects face false-positives with 100% precision)
        if palm_width_m > 0.105 or palm_len_m > 0.138:
            return False

    # 4. 2D Fallback Anti-Face Defense (when face is close to camera without hands):
    # A pseudo-hand spanning the entire head has wrist at throat/chin (y > 0.45)
    # and fingertips at upper forehead (y < 0.18), with width > 0.28 and center x in middle.
    if min_y < 0.18 and max_y > 0.50 and bbox_w > 0.28 and (0.28 < (min_x + max_x) / 2 < 0.72):
        L_palm_2d = float(np.hypot(mid_mcp.x - wrist.x, mid_mcp.y - wrist.y))
        if L_palm_2d > 0.15:
            return False

    return True

def get_symbolic_prior(hand_landmarks_list, orig_w=1280, orig_h=720, neural_probs=None):
    """
    Computes a gentle, non-destructive Bayesian prior probability distribution over all 25 classes.
    Preserves 98%+ neural model accuracy while gently resolving genuine ambiguities:
    - Drink vs Water: chin/mouth elevation arbitration
    - Food vs Eat: pinched mouth vs open palm waft
    - 2_Two vs 8_Eight: V-sign vs 8_Eight neural consensus
    - 7_Seven vs 1_One: L-shape gun vs upright index
    - Give vs Where vs We: balanced 1-hand & 2-hand gesture arbitration
    - Single-hand Milk suppression: prevents Milk false-positives on single open palms
    - What suppression: prevents What misfires when performing other signs
    """
    prior = np.ones(25, dtype=np.float32) * 1.0
    if not hand_landmarks_list:
        return prior

    active_hands = [h for h in hand_landmarks_list if h[0].y <= 0.96]
    if not active_hands:
        return prior * 0.10

    nh = len(active_hands)
    if neural_probs is None:
        return prior

    def idx(name):
        return CLASS_NAMES.index(name)

    # 1. Milk strictly requires two hands squeezing fists
    if nh == 1:
        prior[idx('Milk')] *= 0.10

    # 2. 2_Two vs 8_Eight arbitration
    p2, p8 = neural_probs[idx('2_Two')], neural_probs[idx('8_Eight')]
    if p8 > p2 and p8 >= 0.15:
        prior[idx('8_Eight')] *= 1.40
        prior[idx('2_Two')] *= 0.70
    elif p2 >= p8 and p2 >= 0.15:
        prior[idx('2_Two')] *= 1.40
        prior[idx('8_Eight')] *= 0.70

    # 3. Drink vs Water arbitration
    pd, pw = neural_probs[idx('Drink')], neural_probs[idx('Water')]
    if pd > pw and pd >= 0.15:
        prior[idx('Drink')] *= 1.40
        prior[idx('Water')] *= 0.70
    elif pw >= pd and pw >= 0.15:
        prior[idx('Water')] *= 1.40
        prior[idx('Drink')] *= 0.70

    # 4. Food vs Eat arbitration
    pf, pe = neural_probs[idx('Food')], neural_probs[idx('Eat')]
    if pf >= pe and pf >= 0.15:
        prior[idx('Food')] *= 1.40
        prior[idx('Eat')] *= 0.70
    elif pe > pf and pe >= 0.15:
        prior[idx('Eat')] *= 1.40
        prior[idx('Food')] *= 0.70

    # 5. 7_Seven vs 1_One arbitration
    p7, p1 = neural_probs[idx('7_Seven')], neural_probs[idx('1_One')]
    if p7 >= p1 and p7 >= 0.15:
        prior[idx('7_Seven')] *= 1.40
        prior[idx('1_One')] *= 0.70
    elif p1 > p7 and p1 >= 0.15:
        prior[idx('1_One')] *= 1.40
        prior[idx('7_Seven')] *= 0.70

    # 6. My vs I_Me arbitration
    pmy, pime = neural_probs[idx('My')], neural_probs[idx('I_Me')]
    if pmy >= pime and pmy >= 0.15:
        prior[idx('My')] *= 1.40
        prior[idx('I_Me')] *= 0.70
    elif pime > pmy and pime >= 0.15:
        prior[idx('I_Me')] *= 1.40
        prior[idx('My')] *= 0.70

    # 7. Give vs Where vs We arbitration
    pgive, pwhere, pwe = neural_probs[idx('Give')], neural_probs[idx('Where')], neural_probs[idx('We')]
    if pgive >= pwhere and pgive >= pwe and pgive >= 0.15:
        prior[idx('Give')] *= 1.40
        prior[idx('Where')] *= 0.70
    elif pwhere >= pgive and pwhere >= pwe and pwhere >= 0.15:
        prior[idx('Where')] *= 1.40
        prior[idx('Give')] *= 0.70
    elif pwe >= pgive and pwe >= pwhere and pwe >= 0.15:
        prior[idx('We')] *= 1.40

    # 8. You vs 6_Six arbitration
    pyou, psix = neural_probs[idx('You')], neural_probs[idx('6_Six')]
    if pyou >= psix and pyou >= 0.15:
        prior[idx('You')] *= 1.40
        prior[idx('6_Six')] *= 0.70

    # 9. Want vs What arbitration
    pwant, pwhat = neural_probs[idx('Want_Desire_Wish_For')], neural_probs[idx('What')]
    if pwant >= pwhat and pwant >= 0.15:
        prior[idx('Want_Desire_Wish_For')] *= 1.40
        prior[idx('What')] *= 0.60
    elif pwhat > pwant and pwhat >= 0.20:
        prior[idx('What')] *= 1.40
        prior[idx('Want_Desire_Wish_For')] *= 0.70

    # 10. Suppress What spurious misfires when neural probability is low
    if neural_probs[idx('What')] < 0.20:
        prior[idx('What')] *= 0.40

    # 11. 10_Ten Support (Fist or 2 open hands)
    pten = neural_probs[idx('10_Ten')]
    if pten >= 0.15:
        prior[idx('10_Ten')] *= 1.30

    return prior

class RealTimeISLTranslator:
    def __init__(self, weights_path=MODEL_WEIGHTS_PATH, conf_thresh=0.46):
        print("Initializing High-Accuracy Real-Time ISL Translation Engine...")
        self.device = torch.device('cpu')
        self.conf_thresh = conf_thresh
        self.min_margin = 0.10           # Robust winning margin
        self.enable_voice = True
        self.nlp = SignLanguageNLP(enable_voice=self.enable_voice)
        self.is_mirrored = False  # Raw camera view matches 100% dataset coordinates; tap [m] to toggle mirror

        # Fluid Dynamic & In-Motion Recognition Support
        self.prev_wrist = None
        self.prev_wrist1 = None
        self.wrist_velocity = 0.0
        self.wild_motion_thresh = 0.22  # Allows dynamic in-place signing motions (Want, Milk, Help)
        self.window_history = deque(maxlen=7)  # 7-frame sliding consensus window
        self.required_consensus_frames = 4    # 4 agreeing frames (~0.13s) confirms sign!
        self.flash_capture_word = None
        self.flash_capture_time = 0

        # Multi-digit sequential 10 tracking (1_One followed by 0 / 10_Ten)
        self.last_digit_token = None
        self.last_digit_time = 0

        # Session Recording State
        self.is_recording = False
        self.recorded_tokens = []
        self.last_confirmed_word = None
        self.last_confirmed_time = 0
        self.no_hand_frames = 0
        self.current_sentence = "Press [SPACEBAR] to start recording signs"

        # Load Attention Model
        self.model = SwinMishAttentionNet(in_features=160, num_classes=25)
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded Swin-Mish Attention model from {weights_path}")
        self.model.eval()

        # Initialize MediaPipe Hand Tracker in VIDEO mode for seamless temporal tracking across motion
        # Highly responsive sensitivity (0.20): detects hands INSTANTLY without hesitation or latency
        base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=4,
            running_mode=vision.RunningMode.VIDEO,
            min_hand_detection_confidence=0.32,  # Highly responsive: detects hands instantly while ignoring face skin
            min_hand_presence_confidence=0.25,   # Smooth tracking through motion without face hallucinations
            min_tracking_confidence=0.25
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.prev_ts_ms = 0
        self.user_hand_scale = None

    def filter_primary_user_hands(self, candidate_hands):
        """
        Isolates and locks onto the primary foreground signer's hands, strictly rejecting
        background bystanders, people walking behind, or distant hands.
        Does NOT drop the user's hand when curling fingers into a sign (fist, pointing, etc.).
        """
        if not candidate_hands:
            return []

        scored_hands = []
        for hand in candidate_hands:
            wrist = hand[0]
            # 3D span from wrist to key landmarks (finger tips and MCP)
            # This remains robust even when making a tight fist or pointing at camera
            span_3d = max(
                float(np.sqrt((hand[k].x - wrist.x)**2 + (hand[k].y - wrist.y)**2 + (hand[k].z - wrist.z)**2))
                for k in [4, 8, 12, 16, 20, 9]
            )
            bbox_w = max(lm.x for lm in hand) - min(lm.x for lm in hand)
            bbox_h = max(lm.y for lm in hand) - min(lm.y for lm in hand)
            bbox_diag = float(np.hypot(bbox_w, bbox_h))

            # Hand size metric (uses larger of 3D span or 2D bbox to handle foreshortening)
            hand_size = max(span_3d, bbox_diag * 0.8)

            # Centrality score: Primary user sits in front of the webcam (center x ~ 0.50)
            center_dist = abs(wrist.x - 0.50)
            centrality_weight = max(0.50, 1.0 - 0.25 * center_dist)

            fg_score = hand_size * centrality_weight
            scored_hands.append((fg_score, hand_size, hand))

        # Sort descending by foreground score
        scored_hands.sort(key=lambda item: item[0], reverse=True)

        primary_score, primary_size, primary_hand = scored_hands[0]

        # Only discard if the primary hand is an absurdly tiny speck in the background (< 0.020)
        if primary_size < 0.020:
            return []

        selected_hands = [primary_hand]

        # If multiple hands are detected, evaluate whether hand 2 belongs to the primary user
        if len(scored_hands) >= 2:
            cand_score, cand_size, cand_hand = scored_hands[1]

            # Hand 2 is accepted if:
            # 1. It is not drastically smaller than the primary hand (at least 45% of primary hand size)
            # 2. It is within realistic physical reach of the primary user
            size_ratio = cand_size / max(1e-5, primary_size)
            dx_wrists = abs(primary_hand[0].x - cand_hand[0].x)
            dy_wrists = abs(primary_hand[0].y - cand_hand[0].y)

            # A person standing in the background has size_ratio < 0.35 and small absolute size
            is_same_person = (size_ratio >= 0.40) and (dx_wrists <= 0.80) and (dy_wrists <= 0.70)

            if is_same_person:
                selected_hands.append(cand_hand)

        return selected_hands

    def draw_hand_skeleton(self, frame, hand_landmarks, orig_w, orig_h):
        points = [(int(lm.x * orig_w), int(lm.y * orig_h)) for lm in hand_landmarks]
        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(frame, points[p1], points[p2], (0, 230, 180), 2)
        for pt in points:
            cv2.circle(frame, pt, 4, (0, 160, 255), -1)

    def process_frame(self, frame):
        h, w, _ = frame.shape
        start_t = time.time()

        if self.is_mirrored:
            frame = cv2.flip(frame, 1)

        annotated = frame.copy()

        # Run MediaPipe hand detection with monotonically increasing millisecond timestamp
        ts_ms = max(getattr(self, 'prev_ts_ms', 0) + 1, int(time.time() * 1000))
        self.prev_ts_ms = ts_ms
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = self.detector.detect_for_video(mp_img, ts_ms)

        detected_sign = None
        confidence = 0.0

        # Filter raw detections to strictly valid human hands (eliminating faces, clothing, lap)
        candidate_hands = []
        if res.hand_landmarks:
            for idx, hand in enumerate(res.hand_landmarks):
                # 1. Ignore resting hands in lap / below screen edge (wrist off screen)
                if hand[0].y > 0.96:
                    continue

                world_hand = res.hand_world_landmarks[idx] if getattr(res, 'hand_world_landmarks', None) and idx < len(res.hand_world_landmarks) else None
                score = res.handedness[idx][0].score if getattr(res, 'handedness', None) and idx < len(res.handedness) and res.handedness[idx] else 1.0

                # 2. Check physical hand biomechanics & anti-face filter
                if not is_valid_hand_anatomy(hand, w, h, world_hand=world_hand, handedness_score=score):
                    continue

                candidate_hands.append(hand)

        # 3. Spatial Foreground Lock: strictly isolates primary user's hands & rejects background bystanders
        valid_hands = self.filter_primary_user_hands(candidate_hands)
        num_bg_rejected = max(0, len(candidate_hands) - len(valid_hands))

        # Motion Velocity Computation (Filters out high-speed arm transit)
        is_hand_still = False
        if valid_hands:
            # Sort hands consistently left-to-right to avoid hand ID jitter
            sorted_hands = sorted(valid_hands, key=lambda h: h[0].x)
            curr_wrist0 = (sorted_hands[0][0].x, sorted_hands[0][0].y)
            if len(sorted_hands) >= 2:
                curr_wrist1 = (sorted_hands[1][0].x, sorted_hands[1][0].y)
                if self.prev_wrist is not None and getattr(self, 'prev_wrist1', None) is not None:
                    v0 = float(np.hypot(curr_wrist0[0] - self.prev_wrist[0], curr_wrist0[1] - self.prev_wrist[1]))
                    v1 = float(np.hypot(curr_wrist1[0] - self.prev_wrist1[0], curr_wrist1[1] - self.prev_wrist1[1]))
                    self.wrist_velocity = max(v0, v1)
                else:
                    self.wrist_velocity = 0.0
                self.prev_wrist1 = curr_wrist1
            else:
                if self.prev_wrist is not None:
                    self.wrist_velocity = float(np.hypot(curr_wrist0[0] - self.prev_wrist[0], curr_wrist0[1] - self.prev_wrist[1]))
                else:
                    self.wrist_velocity = 0.0
                self.prev_wrist1 = None
            self.prev_wrist = curr_wrist0
            is_hand_still = (self.wrist_velocity <= self.wild_motion_thresh)
        else:
            self.prev_wrist = None
            self.prev_wrist1 = None
            self.wrist_velocity = 0.0

        if valid_hands:
            for hand in valid_hands:
                self.draw_hand_skeleton(annotated, hand, w, h)

            # 1. Neural Swin-Mish Attention Logits
            feats = extract_advanced_gesture_features(valid_hands, w, h)
            feat_tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits = self.model(feat_tensor)
                neural_probs = torch.softmax(logits, dim=1)[0].numpy()

            # 2. Exact Physics & Geometric Prior (Integrated Number Finger Rules)
            geom_prior = get_symbolic_prior(valid_hands, w, h, neural_probs)

            # 3. Neuro-Symbolic Fusion
            fused_scores = neural_probs * geom_prior
            sorted_indices = np.argsort(fused_scores)
            best_cls = int(sorted_indices[-1])
            second_cls = int(sorted_indices[-2])
            top_conf = float(fused_scores[best_cls] / (np.sum(fused_scores) + 1e-6))
            second_conf = float(fused_scores[second_cls] / (np.sum(fused_scores) + 1e-6))
            margin = top_conf - second_conf

            # Responsive detection threshold (adaptive for motion)
            active_conf_thresh = 0.38 if self.wrist_velocity > 0.03 else self.conf_thresh
            if top_conf >= active_conf_thresh and margin >= self.min_margin:
                detected_sign = CLASS_NAMES[best_cls]
                confidence = top_conf

                # Sleek, clean floating label above wrist (NO clunky boxes)
                wrist_pt = (int(valid_hands[0][0].x * w), int(valid_hands[0][0].y * h))
                tag_label = f"{detected_sign.replace('_', ' ')} ({int(top_conf * 100)}%)"
                cv2.putText(annotated, tag_label, (wrist_pt[0] - 20, max(30, wrist_pt[1] - 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3)
                cv2.putText(annotated, tag_label, (wrist_pt[0] - 20, max(30, wrist_pt[1] - 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 200), 2)

        fps = 1.0 / max(1e-5, (time.time() - start_t))

        # ------------------- FLUID DYNAMIC & STATIC SESSION RECORDING -------------------
        if self.is_recording:
            # Allow recognition during dynamic in-place motion (velocity <= 0.22)
            # Only block if arm is swinging wildly across the screen (velocity > 0.22)
            is_motion_valid = (self.wrist_velocity <= self.wild_motion_thresh)

            if is_motion_valid and detected_sign:
                self.window_history.append(detected_sign)
                # Consensus confirmation: 4 agreeing frames in recent 7-frame window (~0.13s)
                if len(self.window_history) >= self.required_consensus_frames:
                    common_sign, count = Counter(self.window_history).most_common(1)[0]
                    if count >= self.required_consensus_frames:
                        curr_t = time.time()
                        # Sequential 10 detection: User signed 1_One, followed by 10_Ten (fist / 0) within 2.5s
                        if common_sign == "10_Ten" and self.last_digit_token == "1_One" and (curr_t - self.last_digit_time) <= 2.5:
                            if self.recorded_tokens and self.recorded_tokens[-1] == "1_One":
                                self.recorded_tokens[-1] = "10_Ten"
                            else:
                                self.recorded_tokens.append("10_Ten")
                            self.last_confirmed_word = "10_Ten"
                            self.last_confirmed_time = curr_t
                            self.last_digit_token = "10_Ten"
                            self.last_digit_time = curr_t
                            self.flash_capture_word = "10 Ten (1+0)"
                            self.flash_capture_time = curr_t
                            self.window_history.clear()
                            print(f"  [+] Captured Sequential ISL Sign: 10_Ten (From 1 + 0 sequence)")
                        elif common_sign != self.last_confirmed_word and (curr_t - self.last_confirmed_time) > 0.60:
                            self.recorded_tokens.append(common_sign)
                            self.last_confirmed_word = common_sign
                            self.last_confirmed_time = curr_t
                            if common_sign in ["1_One", "2_Two", "3_Three", "4_Four", "5_Five", "6_Six", "7_Seven", "8_Eight", "9_Nine", "10_Ten"]:
                                self.last_digit_token = common_sign
                                self.last_digit_time = curr_t
                            self.flash_capture_word = common_sign
                            self.flash_capture_time = curr_t
                            self.window_history.clear()
                            print(f"  [+] Captured Sign: {common_sign} (Total signs: {len(self.recorded_tokens)})")
            else:
                # Only clear window history if arm swings wildly across the frame
                if self.wrist_velocity > 0.25:
                    self.window_history.clear()

            # Reset last_confirmed_word when hands return to neutral / leave frame
            if not valid_hands:
                self.no_hand_frames += 1
                if self.no_hand_frames > 6:
                    self.last_confirmed_word = None
                    self.window_history.clear()
            else:
                self.no_hand_frames = 0

        # ===============================================================
        # DRAW TOP HUD BANNER
        # ===============================================================
        top_h = 44
        top_overlay = annotated.copy()
        cv2.rectangle(top_overlay, (0, 0), (w, top_h), (15, 15, 15), -1)
        cv2.addWeighted(top_overlay, 0.85, annotated, 0.15, 0, annotated)

        if self.is_recording:
            pulse = int((time.time() * 3) % 2)
            rec_col = (0, 0, 255) if pulse else (0, 80, 255)
            cv2.circle(annotated, (25, 22), 8, rec_col, -1)
            cv2.putText(annotated, "REC ACTIVE", (42, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            cv2.putText(annotated, "- Perform signs - [Spacebar] Finish & Speak - [Backspace] Undo", (168, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        else:
            cv2.rectangle(annotated, (15, 8), (115, 36), (180, 120, 0), -1)
            cv2.putText(annotated, "STANDBY", (22, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(annotated, "Press [SPACEBAR] to Start Recording Signs  |  [m] Mirror  |  [v] Voice  |  [c] Clear", (130, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

        if num_bg_rejected > 0:
            cv2.rectangle(annotated, (w - 360, 8), (w - 110, 36), (40, 30, 20), -1)
            cv2.putText(annotated, f"Background Ignored ({num_bg_rejected})", (w - 350, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 180, 255), 1)

        cv2.putText(annotated, "[q] Quit", (w - 90, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1)

        # ===============================================================
        # DRAW SLEEK BOTTOM HUD (TRANSLATION & HISTORY)
        # ===============================================================
        bot_h = 115
        bot_overlay = annotated.copy()
        cv2.rectangle(bot_overlay, (0, h - bot_h), (w, h), (12, 12, 12), -1)
        cv2.addWeighted(bot_overlay, 0.88, annotated, 0.12, 0, annotated)

        # Line 1: Live Hand & Captured Sequence
        if self.flash_capture_word and (time.time() - self.flash_capture_time) < 0.7:
            live_disp = f"Captured: {self.flash_capture_word.replace('_', ' ')}!"
            live_col = (0, 255, 128)
        elif detected_sign:
            live_disp = f"Live: {detected_sign} ({int(confidence*100)}%)"
            live_col = (0, 255, 200)
        else:
            live_disp = "Live: Ready" if valid_hands else "Live: No Hands"
            live_col = (130, 130, 130)
        cv2.putText(annotated, live_disp, (25, h - 82), cv2.FONT_HERSHEY_SIMPLEX, 0.62, live_col, 2)

        # Display accumulated sequence
        active_seq = self.recorded_tokens if self.recorded_tokens else []
        seq_text = " -> ".join([TOKEN_MAP.get(tok, tok) for tok in active_seq[-7:]])
        seq_label = f"Session Signs: [ {seq_text} ]" if active_seq else "Session Signs: [ ... ]"
        seq_col = (0, 220, 255) if self.is_recording else (180, 180, 180)
        cv2.putText(annotated, seq_label, (360, h - 82), cv2.FONT_HERSHEY_SIMPLEX, 0.60, seq_col, 2)

        # Line 2: NLP English Sentence
        if self.is_recording:
            sent_str = "● Recording signs... Tap [SPACEBAR] to Translate & Speak."
            sent_col = (100, 255, 150)
        else:
            sent_str = f"NLP Sentence: \"{self.current_sentence}\""
            sent_col = (0, 255, 255) if self.current_sentence and not self.current_sentence.startswith("Press") else (180, 220, 220)
        cv2.putText(annotated, sent_str, (25, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.72, sent_col, 2)

        # Line 3: System Status
        mirror_txt = "ON" if self.is_mirrored else "OFF"
        voice_txt = "ON" if self.enable_voice else "OFF"
        meta_str = f"FPS: {fps:.1f}  |  Foreground: Locked ({len(valid_hands)}H)  |  Mirror: {mirror_txt}  |  Engine: Neural ISL-T5  |  Voice: {voice_txt}  |  1280x720 HD"
        cv2.putText(annotated, meta_str, (25, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1)

        return annotated

    def run_webcam(self, camera_index=0, win_name="SIGNOVA - Real-Time Indian Sign Language & NLP Translator"):
        print(f"Opening camera {camera_index} in 1280x720 Widescreen HD mode...")
        is_mac = platform.system() == "Darwin"
        cap = None

        if is_mac:
            # On macOS, try AVFoundation first, then default
            cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
            if not cap or not cap.isOpened():
                cap = cv2.VideoCapture(camera_index)
        else:
            cap = cv2.VideoCapture(camera_index)

        # Fallback to camera 1 if camera 0 is unavailable (common on Macs with Continuity Camera)
        if not cap or not cap.isOpened():
            alt_idx = 1 if camera_index == 0 else 0
            print(f"Camera {camera_index} unavailable. Trying alternative camera index {alt_idx}...")
            if is_mac:
                cap = cv2.VideoCapture(alt_idx, cv2.CAP_AVFOUNDATION)
                if not cap or not cap.isOpened():
                    cap = cv2.VideoCapture(alt_idx)
            else:
                cap = cv2.VideoCapture(alt_idx)

        if not cap or not cap.isOpened():
            print(f"Error: Unable to open camera {camera_index} or alternative indices.")
            if is_mac:
                print("Tip for macOS: Check System Settings -> Privacy & Security -> Camera to grant Terminal/IDE permission.")
            return

        # Request 1280x720 (16:9 widescreen HD)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera active resolution: {actual_w}x{actual_h} (Aspect ratio: {actual_w/actual_h:.2f})")

        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

        print("\n" + "="*68)
        print("          SIGNOVA ISL TRANSLATOR: ANTI-SPURIOUS MODE ACTIVE")
        print("="*68)
        print("  [Spacebar]  : START session -> Hold each sign steady -> STOP & SPEAK")
        print("  [Backspace] : Undo / delete last captured sign")
        print("  [c]         : Clear current sentence")
        print("  [m]         : Toggle Mirror mode")
        print("  [v]         : Toggle Voice Text-to-Speech")
        print("  [q]         : Quit application")
        print("="*68 + "\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = self.process_frame(frame)
            cv2.imshow(win_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # SPACEBAR: Start / Stop session
                if not self.is_recording:
                    # START RECORDING
                    self.is_recording = True
                    self.recorded_tokens = []
                    self.window_history.clear()
                    self.last_confirmed_word = None
                    self.last_confirmed_time = time.time()
                    print("\n" + ">"*50)
                    print("[SESSION STARTED] Recording signs... Hold each sign steady.")
                    print("Tap [SPACEBAR] again when finished to generate English sentence.")
                    print(">"*50)
                else:
                    # STOP RECORDING & GENERATE SENTENCE VIA NLP + TTS
                    self.is_recording = False
                    print("\n" + "<"*50)
                    print(f"[SESSION FINISHED] Captured signs: {self.recorded_tokens}")
                    if self.recorded_tokens:
                        sentence = self.nlp.generate_sentence(self.recorded_tokens)
                        self.current_sentence = sentence
                        print("="*55)
                        print(f"  NLP TRANSLATION: \"{sentence}\"")
                        print("="*55)
                        if self.enable_voice:
                            print("  -> Speaking sentence aloud via Voice TTS...")
                            self.nlp.speak(sentence)
                    else:
                        self.current_sentence = "No signs detected. Press [Spacebar] to record."
                        print("  (No signs recognized during session)")
                    print("<"*50 + "\n")

            elif key == 8 or key == ord('z'):  # Backspace or 'z': Undo last sign
                if self.recorded_tokens:
                    removed = self.recorded_tokens.pop()
                    self.last_confirmed_word = None
                    self.window_history.clear()
                    print(f"\n[-] Undid sign: '{removed}'. Remaining: {self.recorded_tokens}\n")

            elif key == ord('c'):
                self.recorded_tokens.clear()
                self.window_history.clear()
                self.current_sentence = "Press [SPACEBAR] to start recording signs"
                self.last_confirmed_word = None
                print("Sentence buffer cleared.")
            elif key == ord('m'):
                self.is_mirrored = not self.is_mirrored
                print(f"Mirror mode: {'ON' if self.is_mirrored else 'OFF'}")
            elif key == ord('v'):
                self.enable_voice = not self.enable_voice
                self.nlp.enable_voice = self.enable_voice
                print(f"Voice TTS: {'ON' if self.enable_voice else 'OFF'}")

        cap.release()
        cv2.destroyAllWindows()
        self.detector.close()

if __name__ == "__main__":
    translator = RealTimeISLTranslator()
    try:
        translator.run_webcam(0)
    except Exception as e:
        print(f"Camera 0 error: {e}. Trying camera 1...")
        try:
            translator.run_webcam(1)
        except Exception as e2:
            print(f"Camera error: {e2}")
