"""
SIGNOVA - Interactive Web Dashboard
Real-Time Indian Sign Language Recognition & NLP Translation System
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import cv2
import torch
import numpy as np
from collections import deque, Counter
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from scripts.sign_nlp_sentence_generator import SignLanguageNLP, TOKEN_MAP
from models.hand_feature_extractor import extract_advanced_gesture_features
from scripts.train_advanced_classifier import SwinMishAttentionNet, CLASS_NAMES
from scripts.realtime_isl_translator import analyze_hand_geometry, get_symbolic_prior, HAND_CONNECTIONS

# Page Configuration
st.set_page_config(
    page_title="SIGNOVA - ISL & NLP Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #a0aec0;
        margin-bottom: 20px;
    }
    .sentence-box {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #38bdf8;
        border-radius: 12px;
        padding: 18px 24px;
        box-shadow: 0 4px 20px rgba(0, 210, 255, 0.15);
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .sentence-text {
        font-size: 1.5rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        background-color: #334155;
        color: #f8fafc;
        font-size: 0.85rem;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-active {
        background-color: #10b981;
        color: #ffffff;
        font-weight: bold;
    }
    .dict-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #334155;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "gesture_st_classifier.pt")
MODEL_TASK_PATH = os.path.join(PROJECT_ROOT, "hand_landmarker.task")
RAW_DIR = os.path.join(PROJECT_ROOT, "dataset_raw")

# Categorization of 25 classes
CATEGORIES = {
    "Numbers (1 to 10)": [
        "1_One", "2_Two", "3_Three", "4_Four", "5_Five",
        "6_Six", "7_Seven", "8_Eight", "9_Nine", "10_Ten"
    ],
    "Needs, Food & Desires": [
        "Want_Desire_Wish_For", "Eat", "Drink", "Food", "Water", "Milk"
    ],
    "Actions & Requests": [
        "Give", "Go", "Help"
    ],
    "Pronouns & Questions": [
        "I_Me", "My", "You", "We", "What", "Where"
    ]
}

SIGN_DESCRIPTIONS = {
    "1_One": "Extend only your index finger straight up, keeping other fingers folded into a fist.",
    "2_Two": "Extend your index and middle fingers in a V-shape with palm facing forward.",
    "3_Three": "Extend index, middle, and ring fingers upward, thumb holding pinky down.",
    "4_Four": "Extend all four fingers (index, middle, ring, pinky) upward with thumb folded across palm.",
    "5_Five": "Spread all five fingers wide open with palm facing forward.",
    "6_Six": "Extend pinky upward or fold thumb touching the pinky finger.",
    "7_Seven": "Extend thumb and index finger or touch thumb to ring finger.",
    "8_Eight": "Extend index and middle or touch thumb to middle finger.",
    "9_Nine": "Touch index finger tip to thumb tip forming a circle with other fingers up.",
    "10_Ten": "Hold up both open hands (5+5) or extend thumb shaking slightly.",
    "Eat": "Bring your fingertips together and tap them towards your mouth/lips.",
    "Drink": "Cup your hand as if holding a small glass and tilt it toward your mouth.",
    "Food": "Cluster fingers together and tap gently near your chin and mouth.",
    "Give": "Hold both hands (or one) with palms facing upward and push outward towards the viewer.",
    "Go": "Point index finger forward and flick hand outward in direction of travel.",
    "Help": "Rest your closed fist on top of an open flat palm and lift both slightly.",
    "I_Me": "Point your index finger inward towards the center of your chest.",
    "Milk": "Hold hand in front of chest and squeeze fingers into a fist in a milking motion.",
    "My": "Place your flat open palm firmly against the center of your chest.",
    "Want_Desire_Wish_For": "Hold both hands in front at waist height, palms up, curling fingers slightly inward.",
    "Water": "Make a 'W' handshape (3 middle fingers extended) and tap near your chin/lips.",
    "We": "Point index finger towards yourself, then sweep hand outward in an arc across the chest.",
    "What": "Hold both hands in front with palms facing upward and shake them side-to-side.",
    "Where": "Hold open hands palms up with an inquisitive side-to-side questioning motion.",
    "You": "Point your index finger straight forward directly toward the person in front of you."
}

@st.cache_resource
def load_models():
    # Load NLP Engine
    nlp = SignLanguageNLP(enable_voice=True)
    
    # Load Neural Classifier
    model = SwinMishAttentionNet(in_features=160, num_classes=25)
    if os.path.exists(MODEL_WEIGHTS_PATH):
        cp = torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(cp['model_state_dict'])
    model.eval()

    # Load MediaPipe Hand Detector
    base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35
    )
    detector = vision.HandLandmarker.create_from_options(options)

    return nlp, model, detector

def draw_skeleton(frame, hand_landmarks, w, h):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for p1, p2 in HAND_CONNECTIONS:
        cv2.line(frame, points[p1], points[p2], (0, 255, 200), 2)
    for pt in points:
        cv2.circle(frame, pt, 4, (0, 165, 255), -1)

if 'confirmed_words' not in st.session_state:
    st.session_state['confirmed_words'] = []
if 'current_sentence' not in st.session_state:
    st.session_state['current_sentence'] = ""

# App UI
st.markdown('<p class="main-title">🤟 SIGNOVA: Indian Sign Language & NLP Translator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">YOLOv10 with Swin Transformer Attention Architecture & Contextual NLP Sentence Generation</p>', unsafe_allow_html=True)

tab_live, tab_dict = st.tabs(["🎥 Live Webcam Studio", "📖 Sign Language Dictionary & Guide"])

# =========================================================================
# TAB 1: LIVE WEBCAM STUDIO
# =========================================================================
with tab_live:
    col_left, col_right = st.columns([2.2, 1.2])

    with col_right:
        st.subheader("⚙️ Control Panel")
        run_camera = st.toggle("🔴 Activate Live Camera", value=False)
        mirror_mode = st.checkbox("Mirror Camera View", value=True)
        voice_tts = st.checkbox("Voice Output (Speak Sentences)", value=True)
        conf_threshold = st.slider("Detection Confidence", min_value=0.20, max_value=0.85, value=0.40, step=0.05)

        st.markdown("---")
        st.subheader("📝 Live Sentence")
        sentence_placeholder = st.empty()
        init_sentence = st.session_state['current_sentence'] if st.session_state['current_sentence'] else "Waiting for sign gestures..."
        sentence_placeholder.markdown(f"""
        <div class="sentence-box">
            <span style="font-size:0.85rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">English Translation</span>
            <div class="sentence-text">"{init_sentence}"</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear Current Sentence", use_container_width=True):
            st.session_state['confirmed_words'] = []
            st.session_state['current_sentence'] = ""
            st.rerun()

        st.markdown("---")
        st.subheader("📜 Gesture Sequence History")
        history_placeholder = st.empty()
        if st.session_state['confirmed_words']:
            badges_html = "".join([f'<span class="badge badge-active">{TOKEN_MAP.get(w, w)}</span>' for w in st.session_state['confirmed_words']])
            history_placeholder.markdown(f"<div>{badges_html}</div>", unsafe_allow_html=True)
        else:
            history_placeholder.info("Perform signs in front of the camera to begin.")

    with col_left:
        video_placeholder = st.empty()
        fps_placeholder = st.empty()

        if not run_camera:
            video_placeholder.info("👈 Toggle 'Activate Live Camera' in the control panel to start live recognition.")
        else:
            nlp, model, detector = load_models()
            nlp.enable_voice = voice_tts

            cap = cv2.VideoCapture(0)
            # Set 16:9 Widescreen HD
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            if not cap.isOpened():
                video_placeholder.error("Error: Could not access webcam. Please ensure your camera is not in use by another program.")
            else:
                last_word = None
                prev_wrist = None
                hold_count = 0
                candidate_sign = None

                try:
                    while run_camera:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Frame read failed.")
                            break

                        if mirror_mode:
                            frame = cv2.flip(frame, 1)

                        h, w, _ = frame.shape
                        start_t = time.time()

                        # MediaPipe Detection
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        res = detector.detect(mp_img)

                        detected_sign = None
                        confidence = 0.0

                        # Motion velocity gating (ignores rapid hand transit between gestures)
                        is_hand_still = False
                        if res.hand_landmarks:
                            curr_wrist = (res.hand_landmarks[0][0].x, res.hand_landmarks[0][0].y)
                            if prev_wrist is not None:
                                wrist_vel = float(np.hypot(curr_wrist[0] - prev_wrist[0], curr_wrist[1] - prev_wrist[1]))
                            else:
                                wrist_vel = 0.0
                            prev_wrist = curr_wrist
                            is_hand_still = (wrist_vel <= 0.024)
                        else:
                            prev_wrist = None

                        if res.hand_landmarks:
                            all_x, all_y = [], []
                            for hand in res.hand_landmarks:
                                draw_skeleton(frame, hand, w, h)
                                for lm in hand:
                                    all_x.append(lm.x * w)
                                    all_y.append(lm.y * h)

                            x1, y1, x2, y2 = 0, 0, 0, 0
                            if all_x and all_y:
                                x1 = max(0, int(min(all_x)) - 20)
                                y1 = max(0, int(min(all_y)) - 20)
                                x2 = min(w, int(max(all_x)) + 20)
                                y2 = min(h, int(max(all_y)) + 20)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 128), 2)

                            # Feature extraction
                            feats = extract_advanced_gesture_features(res.hand_landmarks, w, h)
                            feat_tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)

                            with torch.no_grad():
                                logits = model(feat_tensor)
                                neural_probs = torch.softmax(logits, dim=1)[0].numpy()

                            geom_prior = get_symbolic_prior(res.hand_landmarks, w, h)
                            fused_scores = neural_probs * geom_prior
                            sorted_idx = np.argsort(fused_scores)
                            best_cls = int(sorted_idx[-1])
                            sec_cls = int(sorted_idx[-2])
                            top_conf = float(fused_scores[best_cls] / (np.sum(fused_scores) + 1e-6))
                            sec_conf = float(fused_scores[sec_cls] / (np.sum(fused_scores) + 1e-6))
                            margin = top_conf - sec_conf

                            if top_conf >= conf_threshold and margin >= 0.14:
                                detected_sign = CLASS_NAMES[best_cls]
                                confidence = top_conf
                                
                                # Render label box on frame
                                if all_x and all_y:
                                    tag = f"{detected_sign.replace('_', ' ')}: {int(top_conf*100)}%"
                                    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                                    by = max(0, y1 - th - 10)
                                    cv2.rectangle(frame, (x1, by), (x1 + tw + 12, y1), (16, 185, 129), -1)
                                    cv2.putText(frame, tag, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                        fps = 1.0 / max(1e-5, (time.time() - start_t))

                        # Anti-spurious hold-to-confirm logic
                        if is_hand_still and detected_sign:
                            if detected_sign == candidate_sign:
                                hold_count += 1
                            else:
                                candidate_sign = detected_sign
                                hold_count = 1

                            if hold_count >= 7:
                                if candidate_sign != last_word:
                                    st.session_state['confirmed_words'].append(candidate_sign)
                                    last_word = candidate_sign
                                    st.session_state['current_sentence'] = nlp.generate_sentence(st.session_state['confirmed_words'])
                                    if voice_tts:
                                        nlp.speak(st.session_state['current_sentence'])
                                hold_count = 0
                        else:
                            if not is_hand_still:
                                hold_count = max(0, hold_count - 2)

                        if not res.hand_landmarks:
                            last_word = None
                            candidate_sign = None
                            hold_count = 0

                        # Update UI elements
                        curr_sent = st.session_state['current_sentence']
                        sentence_placeholder.markdown(f"""
                        <div class="sentence-box">
                            <span style="font-size:0.85rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">English Translation</span>
                            <div class="sentence-text">"{curr_sent if curr_sent else '...'}"</div>
                        </div>
                        """, unsafe_allow_html=True)

                        if st.session_state['confirmed_words']:
                            badges_html = "".join([f'<span class="badge badge-active">{TOKEN_MAP.get(w, w)}</span>' for w in st.session_state['confirmed_words']])
                            history_placeholder.markdown(f"<div>{badges_html}</div>", unsafe_allow_html=True)

                        # Render Frame in Streamlit
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        fps_placeholder.caption(f"⚡ Processing Speed: {fps:.1f} FPS  |  Active Resolution: {w}x{h} (16:9 HD)")

                finally:
                    cap.release()

# =========================================================================
# TAB 2: SIGN LANGUAGE DICTIONARY & REFERENCE GUIDE
# =========================================================================
with tab_dict:
    st.subheader("📚 Indian Sign Language Reference Dictionary (25 Words)")
    st.write("Browse any word below to see the exact signing technique and watch reference demonstrations.")

    selected_category = st.radio("Select Category:", list(CATEGORIES.keys()), horizontal=True)

    words_in_cat = CATEGORIES[selected_category]
    cols = st.columns(2)

    for idx, word in enumerate(words_in_cat):
        col = cols[idx % 2]
        with col:
            st.markdown(f"""
            <div class="dict-card">
                <h4 style="color:#38bdf8; margin-top:0;">{word.replace('_', ' ')}</h4>
                <p style="color:#cbd5e1; font-size:0.95rem;">{SIGN_DESCRIPTIONS.get(word, 'Gesture performed with hand and facial cues.')}</p>
            </div>
            """, unsafe_allow_html=True)

            ref_video_path = os.path.join(RAW_DIR, f"{word}.mp4")
            if os.path.exists(ref_video_path):
                st.video(ref_video_path)
            else:
                st.caption(f"Reference take: {word}")
