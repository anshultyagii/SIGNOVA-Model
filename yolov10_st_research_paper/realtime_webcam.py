"""
YOLOv10-ST Indian Sign Language Translator (Nature Scientific Reports 2025 Edition)
Full Assistive Translation System:
- Real-Time Gesture Recognition with Swin-Mish Attention
- 2D Bounding Boxes matching Figure 4 of Nature Research Paper
- Multi-hand tracking with Background Bystander Rejection
- Hotkey [SPACEBAR] to Record -> Translate -> Speak aloud (NLP + TTS)
- Live HUD with NLP English Sentence Bar and Status Banner
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import cv2
import torch
import numpy as np
from scripts.realtime_isl_translator import RealTimeISLTranslator

class PaperYOLOv10STTranslator(RealTimeISLTranslator):
    """
    Extends the high-accuracy translator with research-paper style
    bounding boxes and visual annotations matching Nature Scientific Reports (2025).
    """
    def __init__(self):
        super().__init__()

    def process_frame(self, frame):
        h, w, _ = frame.shape
        start_t = time.time()

        if self.is_mirrored:
            frame = cv2.flip(frame, 1)

        annotated = frame.copy()

        # Run MediaPipe hand detection with monotonically increasing millisecond timestamp
        ts_ms = max(getattr(self, 'prev_ts_ms', 0) + 1, int(time.time() * 1000))
        self.prev_ts_ms = ts_ms
        import mediapipe as mp
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = self.detector.detect_for_video(mp_img, ts_ms)

        detected_sign = None
        confidence = 0.0

        # Filter raw detections to strictly valid human hands (eliminating faces, clothing, lap)
        candidate_hands = []
        if res.hand_landmarks:
            for idx, hand in enumerate(res.hand_landmarks):
                if hand[0].y > 0.96:
                    continue

                world_hand = res.hand_world_landmarks[idx] if getattr(res, 'hand_world_landmarks', None) and idx < len(res.hand_world_landmarks) else None
                score = res.handedness[idx][0].score if getattr(res, 'handedness', None) and idx < len(res.handedness) and res.handedness[idx] else 1.0

                from scripts.realtime_isl_translator import is_valid_hand_anatomy
                if not is_valid_hand_anatomy(hand, w, h, world_hand=world_hand, handedness_score=score):
                    continue

                candidate_hands.append(hand)

        # Spatial Foreground Lock: strictly isolates primary user's hands & rejects background bystanders
        valid_hands = self.filter_primary_user_hands(candidate_hands)
        num_bg_rejected = max(0, len(candidate_hands) - len(valid_hands))

        # Motion Velocity Computation
        is_hand_still = False
        if valid_hands:
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

        gesture_box = None
        if valid_hands:
            # Draw skeletons
            for hand in valid_hands:
                self.draw_hand_skeleton(annotated, hand, w, h)

            # Compute bounding box enclosing all active hands
            all_xs = [lm.x for hand in valid_hands for lm in hand]
            all_ys = [lm.y for hand in valid_hands for lm in hand]
            min_x, max_x = min(all_xs), max(all_xs)
            min_y, max_y = min(all_ys), max(all_ys)
            pad_x = max(0.02, (max_x - min_x) * 0.12)
            pad_y = max(0.02, (max_y - min_y) * 0.12)

            bx1 = int(max(0.0, min_x - pad_x) * w)
            bx2 = int(min(1.0, max_x + pad_x) * w)
            by1 = int(max(0.0, min_y - pad_y) * h)
            by2 = int(min(1.0, max_y + pad_y) * h)
            gesture_box = (bx1, by1, bx2, by2)

            # 1. Neural Swin-Mish Attention Logits
            from models.hand_feature_extractor import extract_advanced_gesture_features
            from scripts.realtime_isl_translator import get_symbolic_prior, CLASS_NAMES
            feats = extract_advanced_gesture_features(valid_hands, w, h)
            feat_tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits = self.model(feat_tensor)
                neural_probs = torch.softmax(logits, dim=1)[0].numpy()

            # 2. Geometric Prior & Symbolic Fusion
            geom_prior = get_symbolic_prior(valid_hands, w, h, neural_probs)
            fused_scores = neural_probs * geom_prior
            sorted_indices = np.argsort(fused_scores)
            best_cls = int(sorted_indices[-1])
            second_cls = int(sorted_indices[-2])
            top_conf = float(fused_scores[best_cls] / (np.sum(fused_scores) + 1e-6))
            second_conf = float(fused_scores[second_cls] / (np.sum(fused_scores) + 1e-6))
            margin = top_conf - second_conf

            active_conf_thresh = 0.38 if self.wrist_velocity > 0.03 else self.conf_thresh
            if top_conf >= active_conf_thresh and margin >= self.min_margin:
                detected_sign = CLASS_NAMES[best_cls]
                confidence = top_conf

                # Draw Research Paper Style Green Bounding Box (Figure 4)
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 120), 2)
                # Bounding box tag
                tag_str = f"{detected_sign.replace('_', ' ')} {int(confidence*100)}%"
                tag_w = len(tag_str) * 11 + 10
                cv2.rectangle(annotated, (bx1, max(0, by1 - 28)), (bx1 + tag_w, by1), (0, 255, 120), -1)
                cv2.putText(annotated, tag_str, (bx1 + 4, max(18, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        fps = 1.0 / max(1e-5, (time.time() - start_t))

        # ------------------- SESSION RECORDING -------------------
        if self.is_recording:
            is_motion_valid = (self.wrist_velocity <= self.wild_motion_thresh)
            if is_motion_valid and detected_sign:
                self.window_history.append(detected_sign)
                from collections import Counter
                if len(self.window_history) >= self.required_consensus_frames:
                    common_sign, count = Counter(self.window_history).most_common(1)[0]
                    if count >= self.required_consensus_frames:
                        curr_t = time.time()
                        if common_sign != self.last_confirmed_word and (curr_t - self.last_confirmed_time) > 0.60:
                            self.recorded_tokens.append(common_sign)
                            self.last_confirmed_word = common_sign
                            self.last_confirmed_time = curr_t
                            self.flash_capture_word = common_sign
                            self.flash_capture_time = curr_t
                            self.window_history.clear()
                            print(f"  [+] Captured Sign: {common_sign} (Total signs: {len(self.recorded_tokens)})")
            else:
                if self.wrist_velocity > 0.25:
                    self.window_history.clear()

            if not valid_hands:
                self.no_hand_frames += 1
                if self.no_hand_frames > 6:
                    self.last_confirmed_word = None
                    self.window_history.clear()
            else:
                self.no_hand_frames = 0

        # ===============================================================
        # DRAW TOP HUD BANNER (STATUS & CONTROLS)
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

        from scripts.sign_nlp_sentence_generator import TOKEN_MAP
        # Line 1: Live Sign & History
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
        meta_str = f"FPS: {fps:.1f}  |  YOLOv10-ST (Nature 2025)  |  Mirror: {mirror_txt}  |  Engine: Neural ISL-T5  |  Voice: {voice_txt}  |  1280x720 HD"
        cv2.putText(annotated, meta_str, (25, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1)

        return annotated

def main():
    print("=" * 70)
    print("  YOLOv10-ST INDIAN SIGN LANGUAGE & NLP TRANSLATOR (NATURE 2025)")
    print("=" * 70)
    print("Controls:")
    print("  [SPACEBAR] : Start / Stop recording session -> Translate & Speak")
    print("  [m]        : Toggle Mirror mode")
    print("  [v]        : Toggle Voice TTS")
    print("  [c]        : Clear current sentence")
    print("  [q]        : Quit")
    print("=" * 70 + "\n")

    app = PaperYOLOv10STTranslator()
    app.run_webcam(camera_index=0, win_name="SIGNOVA - YOLOv10-ST Indian Sign Language Translator (Nature 2025)")

if __name__ == "__main__":
    main()
