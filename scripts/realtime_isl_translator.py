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
    Uses distance from MCP to TIP relative to PIP to MCP.
    """
    wrist = hand[0]
    mid_mcp = hand[9]
    scale = max(1e-4, np.sqrt((mid_mcp.x - wrist.x)**2 + (mid_mcp.y - wrist.y)**2))

    # Rotation-invariant finger extension check
    d_idx_tip = np.hypot(hand[8].x - hand[5].x, hand[8].y - hand[5].y)
    d_idx_pip = max(1e-4, np.hypot(hand[6].x - hand[5].x, hand[6].y - hand[5].y))
    ext_index = d_idx_tip > (1.30 * d_idx_pip)

    d_mid_tip = np.hypot(hand[12].x - hand[9].x, hand[12].y - hand[9].y)
    d_mid_pip = max(1e-4, np.hypot(hand[10].x - hand[9].x, hand[10].y - hand[9].y))
    ext_middle = d_mid_tip > (1.30 * d_mid_pip)

    d_rng_tip = np.hypot(hand[16].x - hand[13].x, hand[16].y - hand[13].y)
    d_rng_pip = max(1e-4, np.hypot(hand[14].x - hand[13].x, hand[14].y - hand[13].y))
    ext_ring = d_rng_tip > (1.30 * d_rng_pip)

    d_pnk_tip = np.hypot(hand[20].x - hand[17].x, hand[20].y - hand[17].y)
    d_pnk_pip = max(1e-4, np.hypot(hand[18].x - hand[17].x, hand[18].y - hand[17].y))
    ext_pinky = d_pnk_tip > (1.30 * d_pnk_pip)

    # Thumb extended if tip (4) is far from pinky MCP base (17)
    d_thumb_pinky = np.hypot(hand[4].x - hand[17].x, hand[4].y - hand[17].y) / scale
    ext_thumb = d_thumb_pinky > 1.05

    ext_count = sum([ext_index, ext_middle, ext_ring, ext_pinky])
    all_five = ext_thumb and (ext_count >= 4)
    fist = (ext_count == 0)

    return {
        'wrist_x': wrist.x,
        'wrist_y': wrist.y,
        'scale': scale,
        'thumb': ext_thumb,
        'index': ext_index,
        'middle': ext_middle,
        'ring': ext_ring,
        'pinky': ext_pinky,
        'count_ext': ext_count,
        'all_five': all_five,
        'fist': fist
    }

def get_symbolic_prior(hand_landmarks_list, orig_w, orig_h):
    """
    Computes a gentle Bayesian prior probability distribution over all 25 classes.
    Relies on the high-precision 3D Palm-Normal & Joint Attention network for subtle
    distinctions (e.g. Help vs Milk), using the prior solely for physical plausibility:
    - Multi-hand constraints (e.g., damping single-finger numbers when 2 hands are raised)
    - Macro vertical elevation (mouth/chin vs chest/waist)
    """
    prior = np.ones(25, dtype=np.float32) * 1.0
    if not hand_landmarks_list:
        return prior

    # Ignore passive resting hands at the extreme bottom edge of screen (wrist_y > 0.88)
    active_hands = [h for h in hand_landmarks_list if h[0].y < 0.88]
    if not active_hands:
        active_hands = hand_landmarks_list

    num_hands = len(active_hands)
    wrist_y = active_hands[0][0].y

    # Physical Multi-Hand Consistency
    if num_hands >= 2:
        # Two active hands: suppress single-handed digit numbers
        for num_sign in ["1_One", "2_Two", "3_Three", "4_Four", "5_Five", "6_Six", "7_Seven", "8_Eight", "9_Nine"]:
            prior[CLASS_NAMES.index(num_sign)] *= 0.40
        # Gently boost two-handed gestures
        for two_sign in ["10_Ten", "Where", "Help", "Milk", "Want_Desire_Wish_For", "Give", "We"]:
            prior[CLASS_NAMES.index(two_sign)] *= 1.25
    else:
        # Single active hand: 10_Ten and Where strictly require 2 hands
        prior[CLASS_NAMES.index("10_Ten")] *= 0.30
        prior[CLASS_NAMES.index("Where")] *= 0.30

    # Macro Vertical Elevation Zoning
    if wrist_y < 0.32:
        # Upper face / mouth: suppress waist/chest-only signs
        for waist_sign in ["1_One", "2_Two", "3_Three", "4_Four", "5_Five", "6_Six", "7_Seven", "8_Eight", "9_Nine", "Go", "Where", "10_Ten", "Milk"]:
            prior[CLASS_NAMES.index(waist_sign)] *= 0.35
        # Gently boost mouth-level signs
        for mouth_sign in ["Drink", "Eat", "Water", "Food"]:
            prior[CLASS_NAMES.index(mouth_sign)] *= 1.35
    else:
        # Chest / torso: gently suppress mouth-only signs
        for mouth_sign in ["Drink", "Eat"]:
            prior[CLASS_NAMES.index(mouth_sign)] *= 0.35

    return prior

class RealTimeISLTranslator:
    def __init__(self, weights_path=MODEL_WEIGHTS_PATH, conf_thresh=0.45):
        print("Initializing High-Accuracy Real-Time ISL Translation Engine...")
        self.device = torch.device('cpu')
        self.conf_thresh = conf_thresh
        self.min_margin = 0.10           # Robust winning margin
        self.enable_voice = True
        self.nlp = SignLanguageNLP(enable_voice=self.enable_voice)
        self.is_mirrored = True  # Natural mirror mode by default

        # Fast & Responsive Anti-Spurious Protection
        self.prev_wrist = None
        self.prev_wrist1 = None
        self.wrist_velocity = 0.0
        self.max_motion_thresh = 0.038   # Blocks only rapid transit movements across frame
        self.window_history = deque(maxlen=4)  # 4-frame consensus window
        self.required_hold_frames = 3    # Snappy 3 agreeing frames (~0.10s) confirmation!
        self.flash_capture_word = None
        self.flash_capture_time = 0

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

        # Initialize MediaPipe Hand Tracker
        base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.28,
            min_hand_presence_confidence=0.28
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

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

        # Run MediaPipe hand detection
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = self.detector.detect(mp_img)

        detected_sign = None
        confidence = 0.0

        # Motion Velocity Computation (Filters out high-speed arm transit)
        is_hand_still = False
        if res.hand_landmarks:
            # Sort hands consistently left-to-right to avoid hand ID jitter
            sorted_hands = sorted(res.hand_landmarks, key=lambda h: h[0].x)
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
            is_hand_still = (self.wrist_velocity <= self.max_motion_thresh)
        else:
            self.prev_wrist = None
            self.prev_wrist1 = None
            self.wrist_velocity = 0.0

        if res.hand_landmarks:
            for hand in res.hand_landmarks:
                self.draw_hand_skeleton(annotated, hand, w, h)

            # 1. Neural Swin-Mish Attention Logits
            feats = extract_advanced_gesture_features(res.hand_landmarks, w, h)
            feat_tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits = self.model(feat_tensor)
                neural_probs = torch.softmax(logits, dim=1)[0].numpy()

            # 2. Exact Physics & Geometric Prior
            geom_prior = get_symbolic_prior(res.hand_landmarks, w, h)

            # 3. Neuro-Symbolic Fusion
            fused_scores = neural_probs * geom_prior
            sorted_indices = np.argsort(fused_scores)
            best_cls = int(sorted_indices[-1])
            second_cls = int(sorted_indices[-2])
            top_conf = float(fused_scores[best_cls] / (np.sum(fused_scores) + 1e-6))
            second_conf = float(fused_scores[second_cls] / (np.sum(fused_scores) + 1e-6))
            margin = top_conf - second_conf

            # Responsive detection threshold
            if top_conf >= self.conf_thresh and margin >= self.min_margin:
                detected_sign = CLASS_NAMES[best_cls]
                confidence = top_conf

                # Sleek, clean floating label above wrist (NO clunky boxes)
                wrist_pt = (int(res.hand_landmarks[0][0].x * w), int(res.hand_landmarks[0][0].y * h))
                tag_label = f"{detected_sign.replace('_', ' ')} ({int(top_conf * 100)}%)"
                cv2.putText(annotated, tag_label, (wrist_pt[0] - 20, max(30, wrist_pt[1] - 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3)
                cv2.putText(annotated, tag_label, (wrist_pt[0] - 20, max(30, wrist_pt[1] - 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 200), 2)

        fps = 1.0 / max(1e-5, (time.time() - start_t))

        # ------------------- FAST RESPONSIVE SESSION RECORDING -------------------
        if self.is_recording:
            if is_hand_still and detected_sign:
                self.window_history.append(detected_sign)
                # Fast confirmation (3 agreeing frames in recent 4-frame window ~ 0.10s)
                if len(self.window_history) >= self.required_hold_frames:
                    common_sign, count = Counter(self.window_history).most_common(1)[0]
                    if count >= self.required_hold_frames:
                        curr_t = time.time()
                        if common_sign != self.last_confirmed_word and (curr_t - self.last_confirmed_time) > 0.55:
                            self.recorded_tokens.append(common_sign)
                            self.last_confirmed_word = common_sign
                            self.last_confirmed_time = curr_t
                            self.flash_capture_word = common_sign
                            self.flash_capture_time = curr_t
                            self.window_history.clear()
                            print(f"  [+] Captured Sign: {common_sign} (Total signs: {len(self.recorded_tokens)})")
            else:
                if not is_hand_still:
                    self.window_history.clear()

            # Reset last_confirmed_word when hands return to neutral / leave frame
            if not res.hand_landmarks:
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
            live_col = (0, 255, 200) if is_hand_still else (160, 160, 160)
        else:
            live_disp = "Live: Ready" if res.hand_landmarks else "Live: No Hands"
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
        meta_str = f"FPS: {fps:.1f}  |  Mirror: {mirror_txt}  |  Voice TTS: {voice_txt}  |  Undo: [Backspace]  |  1280x720 HD"
        cv2.putText(annotated, meta_str, (25, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1)

        return annotated

    def run_webcam(self, camera_index=0):
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

        win_name = "SIGNOVA - Real-Time Indian Sign Language & NLP Translator"
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
