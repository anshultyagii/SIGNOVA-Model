# -*- coding: utf-8 -*-
"""
SignSpeak AI - Confusion Diagnostic Script
==========================================
Loads every gallery template and computes:
  1. The pairwise DTW distance matrix (all 25 x 25 classes).
  2. A per-class leave-one-out "self-recognition" rate.
  3. A deep dive on the Drink vs Water pair.

Run from the project root:
    .venv/Scripts/python.exe scripts/diagnose_confusion.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.pipeline import ExemplarRecognitionEngine
from src.recognition.matcher import _pack_sequence, _resample_matrix, _batch_frame_distances, DTW_RESAMPLE_LEN

GALLERY_DIR = PROJECT_ROOT / "dataset" / "processed_corrected"

# --------------------------- helpers -----------------------------------------

def load_all_templates(gallery_dir: Path):
    """Load raw NPZ data and normalized sequences for all gallery files."""
    from src.recognition.features import normalize_sequence
    engine = ExemplarRecognitionEngine(gallery_dir=gallery_dir, trim_to_active=True)
    # Also load raw NPZ for statistics
    raw = {}
    npz_files = sorted(gallery_dir.glob("*_landmarks.npz"))
    for f in npz_files:
        with np.load(f) as data:
            label = str(data["label"]) if "label" in data else f.stem.replace("_landmarks", "")
            raw[label] = {
                "features": data["features"].copy(),
                "presence": data["presence"].copy(),
            }
    return engine, raw


def compute_dtw(engine, label_a, label_b):
    seq_a = engine.matcher.templates[label_a]
    seq_b = engine.matcher.templates[label_b]
    return engine.matcher.compute_dtw_distance(seq_a, seq_b)


# --------------------------- main --------------------------------------------

def main():
    print("=" * 72)
    print("SignSpeak AI — Confusion Diagnostic")
    print("=" * 72)

    engine, raw = load_all_templates(GALLERY_DIR)
    labels = sorted(engine.matcher.templates.keys())
    n = len(labels)
    print(f"\n{n} templates loaded.\n")

    # -- 1. Template statistics ------------------------------------------------
    print("-" * 72)
    print("TEMPLATE STATISTICS (after trim_to_active normalization)")
    print("-" * 72)
    print(f"  {'Label':<28}  {'RawF':>5}  {'TrimF':>5}  {'LH%':>5}  {'RH%':>5}  {'Both%':>5}")
    print("  " + "-" * 60)
    for lbl in labels:
        pres = raw[lbl]["presence"]
        raw_frames = pres.shape[0]
        trim_frames = engine.matcher.templates[lbl].length
        lh_pct = 100.0 * pres[:, 2].sum() / raw_frames
        rh_pct = 100.0 * pres[:, 3].sum() / raw_frames
        both_pct = 100.0 * np.logical_and(pres[:, 2], pres[:, 3]).sum() / raw_frames
        print(f"  {lbl:<28}  {raw_frames:>5}  {trim_frames:>5}  {lh_pct:>4.0f}%  {rh_pct:>4.0f}%  {both_pct:>4.0f}%")

    # -- 2. Deep dive: Drink vs Water -----------------------------------------
    print("\n" + "-" * 72)
    print("DEEP DIVE: Drink vs Water")
    print("-" * 72)
    for lbl in ("Drink", "Water"):
        if lbl not in engine.matcher.templates:
            print(f"  WARNING: {lbl} not found in gallery!")
            continue
        seq = engine.matcher.templates[lbl]
        packed = getattr(seq, "_packed", _pack_sequence(seq))
        resampled = _resample_matrix(packed, DTW_RESAMPLE_LEN)

        pres = raw[lbl]["presence"]
        lh_frames = int(pres[:, 2].sum())
        rh_frames = int(pres[:, 3].sum())
        both_frames = int(np.logical_and(pres[:, 2], pres[:, 3]).sum())

        # Pose spread (how much the upper body moves across the trimmed sequence)
        # Packed row layout: pose=[0:21], rh_shape=[21:84], rh_pos=[84:87], rh_flag=[87],
        #                    lh_shape=[88:151], lh_pos=[151:154], lh_flag=[154]
        pose_mat = resampled[:, 0:21]       # (30, 21)
        rh_pos   = resampled[:, 84:87]      # (30, 3)
        lh_pos   = resampled[:, 151:154]    # (30, 3)
        rh_shape = resampled[:, 21:84]      # (30, 63)
        lh_shape = resampled[:, 88:151]     # (30, 63)

        pose_spread = float(np.std(pose_mat))
        rh_pos_spread = float(np.std(rh_pos))
        lh_pos_spread = float(np.std(lh_pos))
        rh_shape_spread = float(np.std(rh_shape))
        lh_shape_spread = float(np.std(lh_shape))

        # Mean rh/lh flag across resampled frames (how often hand is present)
        rh_flag_mean = float(resampled[:, 87].mean())
        lh_flag_mean = float(resampled[:, 154].mean())

        print(f"\n  [{lbl}]  raw={pres.shape[0]}f  trimmed={seq.length}f  "
              f"LH={lh_frames}f  RH={rh_frames}f  BOTH={both_frames}f")
        print(f"    Resampled shape: {resampled.shape}")
        print(f"    RH present fraction (resampled): {rh_flag_mean:.2f}")
        print(f"    LH present fraction (resampled): {lh_flag_mean:.2f}")
        print(f"    Pose std-dev:     {pose_spread:.4f}")
        print(f"    RH position std:  {rh_pos_spread:.4f}")
        print(f"    LH position std:  {lh_pos_spread:.4f}")
        print(f"    RH shape std:     {rh_shape_spread:.4f}")
        print(f"    LH shape std:     {lh_shape_spread:.4f}")

    if "Drink" in engine.matcher.templates and "Water" in engine.matcher.templates:
        drink_water_dist = compute_dtw(engine, "Drink", "Water")
        water_drink_dist = compute_dtw(engine, "Water", "Drink")

        seq_d = engine.matcher.templates["Drink"]
        seq_w = engine.matcher.templates["Water"]
        mat_d = _resample_matrix(getattr(seq_d, "_packed", _pack_sequence(seq_d)), DTW_RESAMPLE_LEN)
        mat_w = _resample_matrix(getattr(seq_w, "_packed", _pack_sequence(seq_w)), DTW_RESAMPLE_LEN)

        # Component-wise distances (averaged across DTW path — approximate via direct resampled distance)
        # Use batch frame distance on resampled matrices
        frame_dist_matrix = _batch_frame_distances(mat_d, mat_w)  # (30, 30)
        # DTW alignment — diagonal gives approximate frame-by-frame breakdown
        diag_dists = np.diag(frame_dist_matrix)
        print(f"\n  DTW distance Drink→Water: {drink_water_dist:.4f}")
        print(f"  DTW distance Water→Drink: {water_drink_dist:.4f}")
        print(f"  Diagonal frame distances (min/mean/max): "
              f"{diag_dists.min():.4f} / {diag_dists.mean():.4f} / {diag_dists.max():.4f}")

    # -- 3. Full pairwise distance matrix -------------------------------------
    print("\n" + "-" * 72)
    print("PAIRWISE DTW DISTANCE MATRIX (lower = more similar)")
    print("-" * 72)
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            if i == j:
                dist_matrix[i, j] = 0.0
            elif j > i:
                d = compute_dtw(engine, la, lb)
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

    # Print header
    short = [lbl[:10] for lbl in labels]
    col_w = 7
    header = " " * 22 + "".join(f"{s:>{col_w}}" for s in short)
    print(header)
    for i, la in enumerate(labels):
        row_str = f"  {la:<20}  " + "".join(
            f"{'--':>{col_w}}" if i == j else f"{dist_matrix[i,j]:>{col_w}.3f}"
            for j in range(n)
        )
        print(row_str)

    # -- 4. Nearest neighbours (confusion candidates) -------------------------
    print("\n" + "-" * 72)
    print("NEAREST NEIGHBOURS (top-2 most similar templates per class)")
    print("-" * 72)
    print(f"  {'Label':<28}  {'#1 Match':<22}  {'Dist1':>6}  {'#2 Match':<22}  {'Dist2':>6}")
    print("  " + "-" * 85)
    for i, la in enumerate(labels):
        row = dist_matrix[i].copy()
        row[i] = np.inf  # exclude self
        sorted_j = np.argsort(row)
        j1, j2 = sorted_j[0], sorted_j[1]
        print(f"  {la:<28}  {labels[j1]:<22}  {dist_matrix[i,j1]:>6.3f}  "
              f"{labels[j2]:<22}  {dist_matrix[i,j2]:>6.3f}")

    # -- 5. Self-recognition leave-one-out (gallery vs gallery) ---------------
    print("\n" + "-" * 72)
    print("SELF-RECOGNITION: Does each template's nearest neighbour == itself?")
    print("(A class should be most different from everything else.)")
    print("-" * 72)
    correct = 0
    for i, la in enumerate(labels):
        row = dist_matrix[i].copy()
        row[i] = np.inf
        best_j = int(np.argmin(row))
        best_lbl = labels[best_j]
        ok = "✓" if best_lbl == la else f"X → {best_lbl}"
        print(f"  {la:<28}  nearest={best_lbl:<22}  dist={dist_matrix[i,best_j]:.3f}  {ok}")
        if best_lbl == la:
            correct += 1
    print(f"\n  Self-recognition rate: {correct}/{n} classes have self as nearest neighbour "
          f"(trivially 0 since self is excluded — metric is: is nearest == same class?)")
    print("  NOTE: Correct interpretation — nearest neighbour should be DIFFERENT class.")
    print("        We want large inter-class distances (especially Drink↔Water).")

    print("\n" + "=" * 72)
    print("Diagnostic complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
