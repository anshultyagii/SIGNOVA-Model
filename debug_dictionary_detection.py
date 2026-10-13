"""
SignSpeak AI - Downloaded ISL Dictionary Dataset Diagnostic
Inspects raw dictionary videos in dataset/raw, checks existing extractions in dataset/processed,
runs fresh per-video MediaPipe Holistic analysis, saves annotated preview frames,
and determines usability of each downloaded sign video.
"""

from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

RAW_DIR = Path(r"F:\SignSpeakAI\dataset\raw")
PROCESSED_DIR = Path(r"F:\SignSpeakAI\dataset\processed")
OUTPUT_DIR = Path(r"F:\SignSpeakAI\debug_output\dictionary_previews")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles


def inspect_existing_processed():
    """Inspect existing .npz files in dataset/processed without modifying them."""
    print("=" * 80)
    print("EXISTING DATASET/PROCESSED EXTRACTION SUMMARY")
    print("=" * 80)
    
    npz_files = sorted(PROCESSED_DIR.glob("*_landmarks.npz"))
    if not npz_files:
        print("No .npz files found in dataset/processed.")
        return {}

    existing_stats = {}
    for f in npz_files:
        try:
            with np.load(f) as data:
                presence = data["presence"]
                features = data["features"]
                total = len(presence)
                pose_cnt = int(presence[:, 0].sum()) if presence.shape[1] > 0 else 0
                face_cnt = int(presence[:, 1].sum()) if presence.shape[1] > 1 else 0
                lh_cnt = int(presence[:, 2].sum()) if presence.shape[1] > 2 else 0
                rh_cnt = int(presence[:, 3].sum()) if presence.shape[1] > 3 else 0
                either_cnt = int(np.logical_or(presence[:, 2], presence[:, 3]).sum())
                
                existing_stats[f.stem.replace("_landmarks", "")] = {
                    "total_frames": total,
                    "features_shape": features.shape,
                    "pose_frames": pose_cnt,
                    "face_frames": face_cnt,
                    "lh_frames": lh_cnt,
                    "rh_frames": rh_cnt,
                    "either_hand_frames": either_cnt,
                    "either_pct": (either_cnt / total * 100) if total > 0 else 0
                }
        except Exception as e:
            print(f"Error reading {f.name}: {e}")
            
    return existing_stats


def analyze_raw_video(video_path: Path, save_previews: bool = True):
    """Analyze a single raw video with a FRESH MediaPipe Holistic instance."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Could not open {video_path.name}"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    preview_sub = OUTPUT_DIR / video_path.stem
    if save_previews:
        preview_sub.mkdir(parents=True, exist_ok=True)

    frame_records = []
    
    # FRESH Holistic instance specifically for this video
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        frame_idx = 0
        best_frame = None
        best_frame_score = -1
        best_frame_idx = -1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            orig_h, orig_w = frame.shape[:2]
            # Match 960-width scaling used by original pipeline
            if orig_w > 960:
                proc_frame = cv2.resize(frame, (960, int(orig_h * 960 / orig_w)))
            else:
                proc_frame = frame

            rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            has_pose = results.pose_landmarks is not None
            has_face = results.face_landmarks is not None
            has_lh = results.left_hand_landmarks is not None
            has_rh = results.right_hand_landmarks is not None
            has_either = has_lh or has_rh
            has_both = has_lh and has_rh

            # Check wrist positions in pose if available
            wrist_in_frame = False
            wrists_near_edge = False
            if has_pose:
                lw = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
                rw = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
                
                # Check visibility and bounds
                l_vis = lw.visibility > 0.5
                r_vis = rw.visibility > 0.5
                wrist_in_frame = l_vis or r_vis

                for w in [lw, rw]:
                    if w.visibility > 0.3:
                        if w.x < 0.05 or w.x > 0.95 or w.y > 0.95 or w.y < 0.05:
                            wrists_near_edge = True

            score = (2 if has_both else (1 if has_either else 0)) + (0.5 if wrist_in_frame else 0)
            if score > best_frame_score:
                best_frame_score = score
                best_frame_idx = frame_idx

            frame_records.append({
                "frame_idx": frame_idx,
                "has_pose": has_pose,
                "has_face": has_face,
                "has_lh": has_lh,
                "has_rh": has_rh,
                "has_either": has_either,
                "has_both": has_both,
                "wrist_in_frame": wrist_in_frame,
                "wrists_near_edge": wrists_near_edge
            })

            # Save preview frames (start, mid, end, and peak gesture frame)
            if save_previews and (frame_idx in [0, total_frames // 4, total_frames // 2, int(total_frames * 0.75)] or frame_idx == best_frame_idx):
                annotated = proc_frame.copy()
                if results.pose_landmarks:
                    mp_draw.draw_landmarks(
                        annotated, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_style.get_default_pose_landmarks_style()
                    )
                if results.left_hand_landmarks:
                    mp_draw.draw_landmarks(
                        annotated, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(0, 200, 0), thickness=2)
                    )
                if results.right_hand_landmarks:
                    mp_draw.draw_landmarks(
                        annotated, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 128, 255), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(0, 100, 255), thickness=2)
                    )
                
                status_text = f"Frame {frame_idx}/{total_frames} | LH:{'Y' if has_lh else 'N'} RH:{'Y' if has_rh else 'N'} Pose:{'Y' if has_pose else 'N'}"
                cv2.putText(annotated, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imwrite(str(preview_sub / f"frame_{frame_idx:03d}.jpg"), annotated)

            frame_idx += 1

    cap.release()

    total_proc = len(frame_records)
    pose_cnt = sum(1 for r in frame_records if r["has_pose"])
    face_cnt = sum(1 for r in frame_records if r["has_face"])
    lh_cnt = sum(1 for r in frame_records if r["has_lh"])
    rh_cnt = sum(1 for r in frame_records if r["has_rh"])
    either_cnt = sum(1 for r in frame_records if r["has_either"])
    both_cnt = sum(1 for r in frame_records if r["has_both"])
    wrist_cnt = sum(1 for r in frame_records if r["wrist_in_frame"])

    # Find active signing window (first and last frame where either hand is detected)
    hand_frames = [r["frame_idx"] for r in frame_records if r["has_either"]]
    if hand_frames:
        active_start = hand_frames[0]
        active_end = hand_frames[-1]
        active_duration = active_end - active_start + 1
        # In the active window, what is hand detection rate?
        window_records = frame_records[active_start:active_end + 1]
        window_either_cnt = sum(1 for r in window_records if r["has_either"])
        window_rate = (window_either_cnt / len(window_records) * 100) if window_records else 0
    else:
        active_start, active_end, active_duration, window_rate = -1, -1, 0, 0

    return {
        "video_name": video_path.name,
        "stem": video_path.stem,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_proc,
        "duration": duration,
        "pose_cnt": pose_cnt,
        "face_cnt": face_cnt,
        "lh_cnt": lh_cnt,
        "rh_cnt": rh_cnt,
        "either_cnt": either_cnt,
        "both_cnt": both_cnt,
        "wrist_cnt": wrist_cnt,
        "either_pct": (either_cnt / total_proc * 100) if total_proc > 0 else 0,
        "active_start": active_start,
        "active_end": active_end,
        "active_duration": active_duration,
        "window_rate": window_rate
    }


def main():
    existing_stats = inspect_existing_processed()
    
    videos = sorted(RAW_DIR.glob("*.mp4"))
    print(f"\nFound {len(videos)} raw dictionary videos in {RAW_DIR}.\n")
    print("=" * 115)
    print(f"{'Video Name':<25} | {'Frames':<7} | {'Pose':<6} | {'LH':<5} | {'RH':<5} | {'Either':<7} | {'Active Window':<15} | {'Status':<12}")
    print("=" * 115)

    results = []
    for v in videos:
        res = analyze_raw_video(v, save_previews=True)
        results.append(res)
        
        name = res["video_name"]
        tf = res["total_frames"]
        pose_pct = f"{res['pose_cnt']/tf*100:.0f}%" if tf > 0 else "0%"
        lh = res["lh_cnt"]
        rh = res["rh_cnt"]
        either = f"{res['either_cnt']} ({res['either_pct']:.0f}%)"
        
        if res["active_start"] >= 0:
            act = f"F{res['active_start']}-F{res['active_end']} ({res['active_duration']}f)"
        else:
            act = "None (0f)"

        # Usability determination:
        # A dictionary video is ~2-4s (50-100 frames). The actual sign takes 1-2s (25-50 frames).
        # Rest frames at start/end have hands lowered.
        if res["either_cnt"] >= 20 or (res["either_pct"] >= 25.0 and res["either_cnt"] >= 15):
            status = "USABLE"
        elif res["either_cnt"] >= 10:
            status = "MARGINAL"
        else:
            status = "UNUSABLE"

        print(f"{name:<25} | {tf:<7} | {pose_pct:<6} | {lh:<5} | {rh:<5} | {either:<7} | {act:<15} | {status:<12}")

    print("=" * 115)
    print("\nDETAILED ANALYSIS & USABILITY SUMMARY:")
    
    usable = [r for r in results if r["either_cnt"] >= 20 or (r["either_pct"] >= 25.0 and r["either_cnt"] >= 15)]
    marginal = [r for r in results if 10 <= r["either_cnt"] < 20 and not (r["either_pct"] >= 25.0 and r["either_cnt"] >= 15)]
    unusable = [r for r in results if r["either_cnt"] < 10]

    print(f"\n  • USABLE videos ({len(usable)}/{len(results)}):")
    for r in usable:
        hand_type = "Two-handed" if r["both_cnt"] > 10 else ("Right-handed" if r["rh_cnt"] > r["lh_cnt"] else "Left-handed")
        print(f"      - {r['video_name']}: {r['either_cnt']}/{r['total_frames']} frames ({r['either_pct']:.1f}%), {hand_type}, active window {r['active_duration']} frames")

    print(f"\n  • MARGINAL videos ({len(marginal)}/{len(results)}):")
    for r in marginal:
        print(f"      - {r['video_name']}: {r['either_cnt']}/{r['total_frames']} frames ({r['either_pct']:.1f}%)")

    print(f"\n  • UNUSABLE videos ({len(unusable)}/{len(results)}):")
    for r in unusable:
        print(f"      - {r['video_name']}: {r['either_cnt']}/{r['total_frames']} frames ({r['either_pct']:.1f}%) - Hands not detected or out of frame")

    # Comparison with dataset/processed
    print("\n" + "=" * 80)
    print("COMPARISON: FRESH HOLISTIC INSTANCE vs EXISTING DATASET/PROCESSED")
    print("=" * 80)
    for r in results:
        stem = r["stem"]
        if stem in existing_stats:
            ex = existing_stats[stem]
            diff_either = r["either_cnt"] - ex["either_hand_frames"]
            diff_rh = r["rh_cnt"] - ex["rh_frames"]
            diff_lh = r["lh_cnt"] - ex["lh_frames"]
            diff_str = f"diff either: {diff_either:+d} (RH: {diff_rh:+d}, LH: {diff_lh:+d})"
            print(f"  {stem:<25}: Fresh={r['either_cnt']} frames vs Processed={ex['either_hand_frames']} frames | {diff_str}")


if __name__ == "__main__":
    main()
