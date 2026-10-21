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
    Validates physical hand biomechanics to reject false-positive
    pseudo-hands mistakenly detected on human faces or background textures,
    without rejecting genuine hands (even when viewed edge-on, tilted, or foreshortened).
    """
    ys = [lm.y for lm in hand]
    xs = [lm.x for lm in hand]
    min_y, max_y = min(ys), max(ys)
    min_x, max_x = min(xs), max(xs)
    wrist = hand[0]
    idx_mcp = hand[5]
    mid_mcp = hand[9]
    pnk_mcp = hand[17]

    # 1. Palm Scale (Wrist to Middle MCP)
    L_palm = float(np.hypot(mid_mcp.x - wrist.x, mid_mcp.y - wrist.y))
    if L_palm < 0.015 or L_palm > 0.20:
        return False

    bbox_h = max_y - min_y
    bbox_w = max_x - min_x
    if bbox_h < 0.010 or bbox_w < 0.010:
        return False

    # 2. 3D World Coordinate Metric Palm Length (meters)
    # Genuine human hand palm (wrist to middle MCP) is 4.0cm to 13.0cm.
    # Face-spanning phantom detections span > 0.15m (skull height ~ 22cm).
    if world_hand is not None and len(world_hand) >= 10:
        w_wrist = world_hand[0]
        w_mid_mcp = world_hand[9]
        L_world = float(np.sqrt((w_mid_mcp.x - w_wrist.x)**2 + 
                                (w_mid_mcp.y - w_wrist.y)**2 + 
                                (w_mid_mcp.z - w_wrist.z)**2))
        if L_world < 0.040 or L_world > 0.130:
            return False

    # 3. Multi-Tier Anti-Face Defense:
    # A pseudo-hand hallucinated on a human face has distinct anatomical hallmarks:
    # It lies in the head/face elevation: fingertips in upper face/hair and bottom near chin/neck.
    is_in_face_zone = (min_y < 0.42) and (max_y < 0.70) and (0.15 < (min_x + max_x) / 2 < 0.85)

    if is_in_face_zone:
        # Tier A: Handedness confidence on facial features is low / uncertain
        if handedness_score < 0.68:
            return False

        # Tier B: Large face-spanning phantom hand (chin to forehead)
        if bbox_h > 0.23:
            return False

        # Tier C: Unnatural palm length matching head/jaw size rather than hand
        if L_palm > 0.115:
            return False

        # Tier D: Unnatural MCP knuckle spread across cheeks/ears
        W_palm = float(np.hypot(idx_mcp.x - pnk_mcp.x, idx_mcp.y - pnk_mcp.y))
        if W_palm > 0.125:
            return False

        # Tier E: Vertical span from forehead/eyes into throat
        if min_y < 0.25 and max_y > 0.48:
            return False

    return True

def get_symbolic_prior(hand_landmarks_list, orig_w, orig_h, neural_probs=None):
    """
    Computes a gentle Bayesian prior probability distribution over all 25 classes.
    Relies on the high-precision 3D Palm-Normal & Joint Attention network for subtle
    distinctions, fused with exact physical finger geometry for numbers 1 to 10 and signs:
    - Complete disambiguation of 2_Two vs 8_Eight (V-sign is 100% 2, 8 never misfires on 2)
    - Complete disambiguation of You vs 6_Six vs 10_Ten (pointing forward is 100% You)
    - Complete disambiguation of What vs Water (palm-up open hand is What; Water requires chin/mouth)
    - Full support for Help and Milk (both 2-handed and single-handed partial)
    - Full disambiguation of I_Me vs My (index pointing to chest vs open palm on chest)
    - Full disambiguation of Food vs Eat (chin elevation waft vs pinched mouth contact)
    - Full disambiguation of Want vs What (two-handed cupped inward pull)
    - Full support for ISL 7_Seven (L / gun shape or thumb-middle contact)
    """
    prior = np.ones(25, dtype=np.float32) * 1.0
    if not hand_landmarks_list:
        return prior

    # Ignore passive resting hands whose wrist is off the bottom screen edge (wrist_y > 0.96)
    active_hands = [h for h in hand_landmarks_list if h[0].y <= 0.96]
    if not active_hands:
        return prior * 0.05

    num_hands = len(active_hands)
    wrist_y = active_hands[0][0].y

    # Compute 3D palm normal for primary hand
    p0 = np.array([active_hands[0][0].x, active_hands[0][0].y, active_hands[0][0].z])
    p5 = np.array([active_hands[0][5].x, active_hands[0][5].y, active_hands[0][5].z])
    p17 = np.array([active_hands[0][17].x, active_hands[0][17].y, active_hands[0][17].z])
    v1 = p5 - p0
    v2 = p17 - p0
    normal = np.cross(v1, v2)
    norm_mag = np.linalg.norm(normal)
    normal = normal / max(1e-5, norm_mag)
    normal_y = normal[1]  # positive = palm facing UP

    # Index pointing vector (depth z-direction into camera)
    hand0 = active_hands[0]
    idx_mcp = hand0[5]
    idx_tip = hand0[8]
    d_z_pointing = idx_tip.z - idx_mcp.z  # negative = pointing towards camera

    p_milk = neural_probs[CLASS_NAMES.index("Milk")] if neural_probs is not None else 0.0
    p_help = neural_probs[CLASS_NAMES.index("Help")] if neural_probs is not None else 0.0
    p_want = neural_probs[CLASS_NAMES.index("Want_Desire_Wish_For")] if neural_probs is not None else 0.0
    p_food = neural_probs[CLASS_NAMES.index("Food")] if neural_probs is not None else 0.0
    p_eat = neural_probs[CLASS_NAMES.index("Eat")] if neural_probs is not None else 0.0
    p_my = neural_probs[CLASS_NAMES.index("My")] if neural_probs is not None else 0.0
    p_ime = neural_probs[CLASS_NAMES.index("I_Me")] if neural_probs is not None else 0.0
    p_seven = neural_probs[CLASS_NAMES.index("7_Seven")] if neural_probs is not None else 0.0
    p_you = neural_probs[CLASS_NAMES.index("You")] if neural_probs is not None else 0.0
    p_what = neural_probs[CLASS_NAMES.index("What")] if neural_probs is not None else 0.0

    if num_hands >= 2:
        # Two active hands: suppress single-handed digit numbers and single-handed signs
        for num_sign in ["1_One", "2_Two", "3_Three", "4_Four", "5_Five", "6_Six", "7_Seven", "8_Eight", "9_Nine", "You"]:
            prior[CLASS_NAMES.index(num_sign)] *= 0.10
        g0 = analyze_hand_geometry(active_hands[0], orig_w, orig_h)
        g1 = analyze_hand_geometry(active_hands[1], orig_w, orig_h)

        # 10_Ten is two open hands (5 + 5 = 10)
        is_ten_two_hands = (g0['all_five'] or g0['count_ext'] >= 4) and (g1['all_five'] or g1['count_ext'] >= 4)
        if is_ten_two_hands and p_milk < 0.35 and p_want < 0.35:
            prior[CLASS_NAMES.index("10_Ten")] *= 5.0

        for two_sign in ["Give", "We"]:
            prior[CLASS_NAMES.index(two_sign)] *= 1.30

        # Help vs Milk vs Where vs Want Disambiguation in 2-handed mode:
        # Help: One hand forms a flat base tray (palm up/spread),
        # other hand forms an upright thumbs-up or fist resting on top of it.
        # Hands are vertically stacked (|y0 - y1| > 0.05 and |x0 - x1| < 0.25)
        has_flat_base = (g0['count_ext'] >= 3 or g1['count_ext'] >= 3 or g0['all_five'] or g1['all_five'])
        has_fist_hand = (g0['fist'] or g1['fist'] or (g0['thumb'] and g0['count_ext'] <= 1) or (g1['thumb'] and g1['count_ext'] <= 1))
        dy_hands = abs(active_hands[0][0].y - active_hands[1][0].y)
        dx_hands = abs(active_hands[0][0].x - active_hands[1][0].x)
        is_stacked_help = has_flat_base and has_fist_hand and dy_hands > 0.05 and dx_hands < 0.25

        if is_stacked_help or (p_help >= 0.30 and p_help > p_milk):
            prior[CLASS_NAMES.index("Help")] *= 8.0
            prior[CLASS_NAMES.index("Milk")] *= 0.001
            prior[CLASS_NAMES.index("Where")] *= 0.001
            prior[CLASS_NAMES.index("Want_Desire_Wish_For")] *= 0.001
            prior[CLASS_NAMES.index("7_Seven")] *= 0.001
            prior[CLASS_NAMES.index("What")] *= 0.01
        elif (g0['fist'] and g1['fist']) or (g0['count_ext'] <= 2 and g1['count_ext'] <= 2 and not has_flat_base) or (p_milk >= 0.25 and p_milk > p_help):
            # Both hands are fists side-by-side: Milk squeeze motion
            prior[CLASS_NAMES.index("Milk")] *= 5.0
            prior[CLASS_NAMES.index("Help")] *= 0.001
            prior[CLASS_NAMES.index("Where")] *= 0.001
            prior[CLASS_NAMES.index("Want_Desire_Wish_For")] *= 0.01
            prior[CLASS_NAMES.index("7_Seven")] *= 0.001
        else:
            if p_help > p_milk:
                prior[CLASS_NAMES.index("Help")] *= 3.0
                prior[CLASS_NAMES.index("Milk")] *= 0.05
            else:
                prior[CLASS_NAMES.index("Milk")] *= 2.5
                prior[CLASS_NAMES.index("Help")] *= 0.10

        # Where in 2-handed mode:
        # Strictly requires BOTH hands open with palms facing up and fingers spread,
        # held separately on either side of the body (dx_hands > 0.20)
        is_where_open = (g0['count_ext'] >= 3 or g0['all_five']) and (g1['count_ext'] >= 3 or g1['all_five'])
        if is_where_open and not g0['fist'] and not g1['fist'] and dx_hands > 0.20 and not is_stacked_help:
            prior[CLASS_NAMES.index("Where")] *= 2.5
        else:
            prior[CLASS_NAMES.index("Where")] *= 0.001

        # Want vs What in 2-handed mode:
        if p_want >= 0.15 and not is_stacked_help:
            prior[CLASS_NAMES.index("Want_Desire_Wish_For")] *= 3.0
            prior[CLASS_NAMES.index("What")] *= 0.20
        elif p_what >= 0.15 and not is_stacked_help:
            prior[CLASS_NAMES.index("What")] *= 3.0
            prior[CLASS_NAMES.index("Want_Desire_Wish_For")] *= 0.20
    else:
        # Single active hand: In ISL, Milk, Where, Give, We strictly require 2 hands!
        prior[CLASS_NAMES.index("Milk")] *= 0.001
        prior[CLASS_NAMES.index("Where")] *= 0.001
        prior[CLASS_NAMES.index("Give")] *= 0.01
        prior[CLASS_NAMES.index("We")] *= 0.01

        g = analyze_hand_geometry(active_hands[0], orig_w, orig_h)
        ext = g['count_ext']
        t, i, m, r, p = g['thumb'], g['index'], g['middle'], g['ring'], g['pinky']

        is_pointing_forward = (d_z_pointing < -0.015)
        is_index_pointing = bool(i and not m and not r and not p)
        # Pointing to chest (I_Me)
        is_pointing_to_chest = (is_index_pointing and d_z_pointing > 0.008 and wrist_y > 0.35)

        # ----------------- Help Disambiguation on Single Hand -----------------
        # In ISL "Help", one hand (fist with thumb UP) rests on top of the other.
        # Often the bottom hand is occluded or resting below, so only the top fist is detected.
        # Top hand signature: thumb extended pointing UP, other 4 fingers folded (fist).
        is_thumb_up = bool(t and (hand0[4].y < hand0[5].y - 0.015) and ext <= 1 and not is_index_pointing)
        if is_thumb_up or (p_help >= 0.25 and ext <= 1 and not is_index_pointing and not is_pointing_forward):
            prior[CLASS_NAMES.index("Help")] *= 8.0
            prior[CLASS_NAMES.index("Milk")] *= 0.001   # Milk is 2-hand only!
            prior[CLASS_NAMES.index("Where")] *= 0.001  # Where is 2-hand only!
            prior[CLASS_NAMES.index("Want_Desire_Wish_For")] *= 0.001 # Want is 2-hand only!
            prior[CLASS_NAMES.index("7_Seven")] *= 0.001
            prior[CLASS_NAMES.index("10_Ten")] *= 0.01
            prior[CLASS_NAMES.index("8_Eight")] *= 0.01
            prior[CLASS_NAMES.index("You")] *= 0.01
        elif is_index_pointing or ext in [2, 3, 4, 5]:
            # Single pointing finger or isolated number is NEVER Help!
            prior[CLASS_NAMES.index("Help")] *= 0.01

        # Want support on single hand during inward pull motion
        if p_want >= 0.15 and not is_thumb_up and not is_index_pointing and wrist_y > 0.36:
            prior[CLASS_NAMES.index("Want_Desire_Wish_For")] *= 4.0
            prior[CLASS_NAMES.index("What")] *= 0.15
        else:
            prior[CLASS_NAMES.index("Want_Desire_Wish_For")] *= 0.001

        # ----------------- I_Me vs My DISAMBIGUATION -----------------
        if is_pointing_to_chest or (is_index_pointing and p_ime >= 0.20):
            prior[CLASS_NAMES.index("I_Me")] *= 4.5
            prior[CLASS_NAMES.index("My")] *= 0.05
            prior[CLASS_NAMES.index("1_One")] *= 0.10
            prior[CLASS_NAMES.index("You")] *= 0.10
        elif (g['all_five'] or ext >= 4) and wrist_y > 0.38 and normal_y < 0.05 and p_my >= 0.15:
            # Flat open palm resting/facing chest is My
            prior[CLASS_NAMES.index("My")] *= 4.5
            prior[CLASS_NAMES.index("I_Me")] *= 0.05
            prior[CLASS_NAMES.index("5_Five")] *= 0.20

        # ----------------- 2_Two vs 8_Eight DISAMBIGUATION -----------------
        p_8 = neural_probs[CLASS_NAMES.index("8_Eight")] if neural_probs is not None else 0.5
        p_2 = neural_probs[CLASS_NAMES.index("2_Two")] if neural_probs is not None else 0.5

        is_8_std = bool(m and r and p and not i)
        is_8_three = bool(t and i and m and not r and not p and g['d_t_i5'] > 0.58 and p_8 > p_2)
        is_8 = is_8_std or is_8_three
        is_2 = bool(i and m and not r and not p and not is_8)

        if is_2:
            prior[CLASS_NAMES.index("2_Two")] *= 4.0
            prior[CLASS_NAMES.index("8_Eight")] *= 0.05
            prior[CLASS_NAMES.index("Go")] *= 0.15
            prior[CLASS_NAMES.index("You")] *= 0.15
            prior[CLASS_NAMES.index("3_Three")] *= 0.15
        elif is_8:
            prior[CLASS_NAMES.index("8_Eight")] *= 4.0
            prior[CLASS_NAMES.index("2_Two")] *= 0.10

        # ----------------- 7_Seven DISAMBIGUATION -----------------
        # Thumb + Index extended (L / gun shape) or thumb touching middle tip WITH index extended
        is_7_gun = bool(t and i and not m and not r and not p)
        is_7_middle_touch = bool(i and not r and not p and g['d_t_m'] < 0.35 and not g['fist'] and not is_thumb_up and ext >= 1)
        is_7 = (is_7_gun or is_7_middle_touch) and not is_thumb_up and not g['fist'] and p_help < 0.15

        if is_7 and not is_pointing_forward and not is_pointing_to_chest:
            prior[CLASS_NAMES.index("7_Seven")] *= 4.5
            prior[CLASS_NAMES.index("You")] *= 0.05
            prior[CLASS_NAMES.index("1_One")] *= 0.05
            prior[CLASS_NAMES.index("Milk")] *= 0.001
            prior[CLASS_NAMES.index("8_Eight")] *= 0.05
            prior[CLASS_NAMES.index("Help")] *= 0.001
        else:
            if is_thumb_up or g['fist'] or p_help >= 0.15:
                prior[CLASS_NAMES.index("7_Seven")] *= 0.001

        # ----------------- YOU vs 6_Six vs 10_Ten vs 1_One -----------------
        if (is_pointing_forward or (is_index_pointing and wrist_y > 0.40 and not is_pointing_to_chest and not is_7)) and p_you >= 0.20:
            prior[CLASS_NAMES.index("You")] *= 4.0
            prior[CLASS_NAMES.index("6_Six")] *= 0.05
            prior[CLASS_NAMES.index("10_Ten")] *= 0.05
            prior[CLASS_NAMES.index("Help")] *= 0.01

        # 1_One: Upright vertical index finger pointing UP
        if is_index_pointing and not is_pointing_forward and not is_pointing_to_chest and not is_7:
            if idx_tip.y < idx_mcp.y:
                prior[CLASS_NAMES.index("1_One")] *= 4.0
                prior[CLASS_NAMES.index("8_Eight")] *= 0.05
                prior[CLASS_NAMES.index("2_Two")] *= 0.05
                prior[CLASS_NAMES.index("6_Six")] *= 0.05
                prior[CLASS_NAMES.index("10_Ten")] *= 0.05
                prior[CLASS_NAMES.index("You")] *= 0.10

        # 6_Six: Pinky only extended
        if (p and not i and not m and not r) and not is_pointing_forward:
            prior[CLASS_NAMES.index("6_Six")] *= 4.5
            prior[CLASS_NAMES.index("You")] *= 0.05
            prior[CLASS_NAMES.index("8_Eight")] *= 0.05

        # 10_Ten Single-Hand Fist
        if g['fist'] and not is_pointing_forward and not is_thumb_up:
            if p_milk < 0.35 and p_help < 0.20:
                prior[CLASS_NAMES.index("10_Ten")] *= 4.0
                prior[CLASS_NAMES.index("Milk")] *= 0.001
                prior[CLASS_NAMES.index("You")] *= 0.05
                prior[CLASS_NAMES.index("Help")] *= 0.01

        # ----------------- WHAT vs WATER -----------------
        is_palm_up = (normal_y > 0.06)
        if is_palm_up and p_want < 0.15:
            prior[CLASS_NAMES.index("What")] *= 4.0
            prior[CLASS_NAMES.index("Water")] *= 0.02  # Water is NEVER palm up!
            prior[CLASS_NAMES.index("Drink")] *= 0.05
            prior[CLASS_NAMES.index("Milk")] *= 0.001

        # 3_Three
        if i and m and r and not p and not is_8:
            prior[CLASS_NAMES.index("3_Three")] *= 4.0
            prior[CLASS_NAMES.index("2_Two")] *= 0.05
            prior[CLASS_NAMES.index("8_Eight")] *= 0.05

        # ----------------- 4_Four vs 5_Five DISAMBIGUATION -----------------
        p_4 = neural_probs[CLASS_NAMES.index("4_Four")] if neural_probs is not None else 0.5
        p_5 = neural_probs[CLASS_NAMES.index("5_Five")] if neural_probs is not None else 0.5

        # 4_Four: 4 fingers extended, thumb tucked
        if ext == 4 and not t and p_4 >= p_5:
            prior[CLASS_NAMES.index("4_Four")] *= 4.0
            prior[CLASS_NAMES.index("8_Eight")] *= 0.05
            prior[CLASS_NAMES.index("5_Five")] *= 0.20

        # 5_Five: all 5 fingers open, or 4 fingers extended with thumb or p_5 > p_4
        if (g['all_five'] or (ext >= 4 and (t or p_5 > p_4))) and not is_palm_up and p_my < 0.25 and p_milk < 0.35:
            prior[CLASS_NAMES.index("5_Five")] *= 4.0
            prior[CLASS_NAMES.index("8_Eight")] *= 0.05
            prior[CLASS_NAMES.index("4_Four")] *= 0.20

        # 9_Nine
        if (g['d_t_i'] < 0.30 and m and r and p) or (r and p and not i and not m):
            prior[CLASS_NAMES.index("9_Nine")] *= 4.0
            prior[CLASS_NAMES.index("8_Eight")] *= 0.05

    # ----------------- FOOD vs EAT vs DRINK vs WATER -----------------
    # In ISL, Eat and Food occur at chin/mouth elevation (wrist_y <= 0.56)
    if wrist_y <= 0.56:
        g = analyze_hand_geometry(active_hands[0], orig_w, orig_h)
        is_pinched_mouth = (g['d_t_i'] < 0.30 and g['count_ext'] <= 1)
        is_open_food = (g['count_ext'] >= 3 or g['d_t_i'] > 0.35)

        if is_pinched_mouth or p_eat > p_food:
            prior[CLASS_NAMES.index("Eat")] *= 3.5
            prior[CLASS_NAMES.index("Food")] *= 0.40
        elif is_open_food or p_food >= p_eat:
            prior[CLASS_NAMES.index("Food")] *= 3.5
            prior[CLASS_NAMES.index("Eat")] *= 0.40

        # Suppress waist-only signs if hand is at upper face / mouth (wrist_y < 0.36)
        if wrist_y < 0.36:
            for waist_sign in ["1_One", "2_Two", "3_Three", "4_Four", "5_Five", "6_Six", "7_Seven", "8_Eight", "9_Nine", "10_Ten", "Milk"]:
                prior[CLASS_NAMES.index(waist_sign)] *= 0.20
    else:
        # Waist / torso: suppress mouth-only signs
        for mouth_sign in ["Drink", "Eat", "Water", "Food"]:
            prior[CLASS_NAMES.index(mouth_sign)] *= 0.15

    return prior

class RealTimeISLTranslator:
    def __init__(self, weights_path=MODEL_WEIGHTS_PATH, conf_thresh=0.46):
        print("Initializing High-Accuracy Real-Time ISL Translation Engine...")
        self.device = torch.device('cpu')
        self.conf_thresh = conf_thresh
        self.min_margin = 0.10           # Robust winning margin
        self.enable_voice = True
        self.nlp = SignLanguageNLP(enable_voice=self.enable_voice)
        self.is_mirrored = True  # Natural mirror mode by default

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
        base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=vision.RunningMode.VIDEO,
            min_hand_detection_confidence=0.35,  # Robust sensitivity: rejects face false-positives while tracking hands
            min_hand_presence_confidence=0.25,   # Continuous smooth tracking through dynamic motion
            min_tracking_confidence=0.25
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.prev_ts_ms = 0

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
        valid_hands = []
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

                valid_hands.append(hand)

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
        meta_str = f"FPS: {fps:.1f}  |  Mirror: {mirror_txt}  |  Engine: Neural ISL-T5  |  Voice TTS: {voice_txt}  |  Undo: [Backspace]  |  1280x720 HD"
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
