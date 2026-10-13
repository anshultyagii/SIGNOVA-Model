"""
SignSpeak AI - Gallery Pipeline Sanity Check Script
Compares reference templates against the enrolled gallery to verify pipeline correctness.
IMPORTANT: This is a mathematical pipeline sanity check, NOT a model accuracy evaluation.
"""

from pathlib import Path
import sys
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.pipeline import ExemplarRecognitionEngine
from src.recognition.features import normalize_sequence

GALLERY_DIR = PROJECT_ROOT / "dataset" / "processed_corrected"


def run_sanity_check():
    print("=" * 95)
    print(" " * 18 + "[PIPELINE SANITY CHECK - NOT AN ACCURACY EVALUATION]")
    print("=" * 95)
    print("NOTE:")
    print("  • This test evaluates mathematical pipeline consistency, normalization, and DTW distance.")
    print("  • Comparing templates against themselves verifies that identical inputs achieve distance 0.0")
    print("    and that top candidates are logically ranked.")
    print("  • This does NOT demonstrate real-world generalization, cross-signer robustness, or accuracy.")
    print("=" * 95)

    if not GALLERY_DIR.exists():
        print(f"ERROR: Gallery directory not found: {GALLERY_DIR}")
        sys.exit(1)

    print(f"Initializing Recognition Engine from: {GALLERY_DIR}")
    engine = ExemplarRecognitionEngine(gallery_dir=GALLERY_DIR, trim_to_active=True)
    labels = engine.get_gallery_labels()
    print(f"Loaded {len(labels)} enrolled sign templates in gallery.\n")

    print("-" * 95)
    print(f"{'Query Sign':<22} | {'Rank 1 Match':<20} | {'Score':<8} | {'Dist':<7} | {'Rank 2':<18} | {'Margin':<7}")
    print("-" * 95)

    all_identity_match = True
    margins = []

    for label in labels:
        npz_path = GALLERY_DIR / f"{label}_landmarks.npz"
        res = engine.recognize_npz_file(npz_path, top_k=3)

        top_pred = res["top_prediction"]
        top_cand = res["candidates"][0]
        rank2_cand = res["candidates"][1] if len(res["candidates"]) > 1 else None

        score = top_cand["similarity"]
        dist = top_cand["distance"]
        r2_label = rank2_cand["label"] if rank2_cand else "None"
        margin = (top_cand["similarity"] - rank2_cand["similarity"]) if rank2_cand else 0.0
        margins.append(margin)

        is_correct = (top_pred == label)
        if not is_correct or abs(dist) > 1e-4:
            all_identity_match = False

        status_flag = "✓" if is_correct else "✗ FAIL"
        print(
            f"{label:<22} | {top_pred:<20} | {score:<8.4f} | {dist:<7.4f} | {r2_label:<18} | {margin:<7.4f}"
        )

    print("-" * 95)
    print("\nTEMPORAL WARPING ROBUSTNESS TEST (Half-speed / Stride test):")
    print("  Testing whether DTW correctly aligns temporal variations (odd frames vs even frames)...")

    temporal_correct = 0
    for label in labels:
        npz_path = GALLERY_DIR / f"{label}_landmarks.npz"
        with np.load(npz_path) as data:
            feats = data["features"]
            pres = data["presence"]

        # Subsample every second frame (simulating a sign executed at 2x frame rate / half duration)
        sub_feats = feats[::2]
        sub_pres = pres[::2]

        res = engine.recognize_features(sub_feats, sub_pres, query_name=f"{label}_stride2", top_k=3)
        if res["top_prediction"] == label:
            temporal_correct += 1

    print(f"  Result: {temporal_correct}/{len(labels)} signs correctly matched under 2x speed temporal decimation.")

    print("\n" + "=" * 95)
    print("PIPELINE INTEGRITY SUMMARY:")
    print(f"  • Enrolled gallery classes: {len(labels)}")
    print(f"  • Identity self-match verified (dist = 0.0000): {'PASS' if all_identity_match else 'FAIL'}")
    print(f"  • Average similarity margin over 2nd candidate: {np.mean(margins):.4f}")
    print(f"  • DTW temporal alignment invariance: {'PASS' if temporal_correct >= 24 else 'REVIEW'}")
    print("=" * 95)


if __name__ == "__main__":
    run_sanity_check()
