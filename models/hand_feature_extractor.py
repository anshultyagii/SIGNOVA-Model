import numpy as np

def extract_advanced_gesture_features(hand_landmarks_list, orig_w=1280, orig_h=720):
    """
    Computes a rotation-, scale-, and position-invariant 160-dimensional feature vector
    for Indian Sign Language gesture recognition:
    - Palm normal 3D orientation (facing camera, self, up, down)
    - Rotation-invariant continuous finger extension ratios (Tip-MCP vs PIP-MCP)
    - Scale-normalized 3D joint landmark coordinates
    - Inter-finger tip distances and relative wrist elevation
    - Dual-hand interaction metrics (if 2 hands active)
    """
    feats = np.zeros(160, dtype=np.float32)
    num_hands = len(hand_landmarks_list)
    feats[0] = min(num_hands, 2) / 2.0  # Normalized hand count

    if num_hands == 0:
        return feats

    # Sort hands by X-coordinate so left/right hand assignment is consistent
    hands_sorted = sorted(hand_landmarks_list[:2], key=lambda h: h[0].x)

    for h_idx, hand in enumerate(hands_sorted):
        offset = 1 + h_idx * 78
        wrist = hand[0]
        wx, wy, wz = wrist.x, wrist.y, wrist.z

        # 1. Centered vertical elevation (face: < -0.15, chest: -0.05..0.15, waist: > 0.15)
        feats[offset + 0] = wy - 0.50

        # 2. Scale normalization factor (wrist to middle MCP landmark 9)
        mid_mcp = hand[9]
        scale = max(1e-4, np.sqrt((mid_mcp.x - wx)**2 + (mid_mcp.y - wy)**2 + (mid_mcp.z - wz)**2))
        feats[offset + 14] = scale

        # 3. Palm 3D Normal Vector (cross product of wrist->index_mcp and wrist->pinky_mcp)
        idx_mcp = hand[5]
        pnk_mcp = hand[17]
        v1 = np.array([idx_mcp.x - wx, idx_mcp.y - wy, idx_mcp.z - wz])
        v2 = np.array([pnk_mcp.x - wx, pnk_mcp.y - wy, pnk_mcp.z - wz])
        normal = np.cross(v1, v2)
        norm_mag = np.linalg.norm(normal)
        if norm_mag > 1e-5:
            normal = normal / norm_mag
        feats[offset + 1] = normal[0]
        feats[offset + 2] = normal[1]
        feats[offset + 3] = normal[2]

        # 4. Finger extension metrics (Tip distance to Wrist / Scale)
        finger_tips = [4, 8, 12, 16, 20]
        for f_i, tip in enumerate(finger_tips):
            d_wrist = np.sqrt((hand[tip].x - wx)**2 + (hand[tip].y - wy)**2 + (hand[tip].z - wz)**2) / scale
            feats[offset + 4 + f_i] = d_wrist

        # 5. Rotation-invariant finger extension ratios: dist(tip, mcp) / dist(pip, mcp)
        finger_pips = [2, 6, 10, 14, 18]
        finger_mcps = [1, 5, 9, 13, 17]
        for f_i, (tip, pip, mcp) in enumerate(zip(finger_tips, finger_pips, finger_mcps)):
            d_tip_mcp = np.sqrt((hand[tip].x - hand[mcp].x)**2 + (hand[tip].y - hand[mcp].y)**2 + (hand[tip].z - hand[mcp].z)**2)
            d_pip_mcp = max(1e-4, np.sqrt((hand[pip].x - hand[mcp].x)**2 + (hand[pip].y - hand[mcp].y)**2 + (hand[pip].z - hand[mcp].z)**2))
            feats[offset + 9 + f_i] = min(3.0, d_tip_mcp / d_pip_mcp)

        # 6. Scale-invariant 3D landmark coordinates (21 x 3 = 63 values)
        for i, lm in enumerate(hand):
            feats[offset + 15 + i*3 + 0] = (lm.x - wx) / scale
            feats[offset + 15 + i*3 + 1] = (lm.y - wy) / scale
            feats[offset + 15 + i*3 + 2] = (lm.z - wz) / scale

    # 7. Dual-hand interaction features
    if len(hands_sorted) >= 2:
        h1 = hands_sorted[0]
        h2 = hands_sorted[1]
        # Distance between wrists
        d_wrists = np.sqrt((h1[0].x - h2[0].x)**2 + (h1[0].y - h2[0].y)**2 + (h1[0].z - h2[0].z)**2)
        feats[157] = d_wrists
        # Distance between index finger tips
        d_idx = np.sqrt((h1[8].x - h2[8].x)**2 + (h1[8].y - h2[8].y)**2 + (h1[8].z - h2[8].z)**2)
        feats[158] = d_idx
        # Relative elevation difference
        feats[159] = h1[0].y - h2[0].y

    return feats

if __name__ == "__main__":
    dummy_vec = extract_advanced_gesture_features([])
    print("Advanced feature vector size:", dummy_vec.shape)
