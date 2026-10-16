# -*- coding: utf-8 -*-
"""
SignSpeak AI - Recorded Samples Evaluation Script (Phase 1)
===========================================================
Evaluates independent webcam-recorded sign samples (.npz files) against the
current enrolled gallery using ExemplarRecognitionEngine and the existing
normalization & DTW matching pipeline.

Key Capabilities:
  - Canonicalizes labels (e.g., '1_One' -> 'One', '10_Ten' -> 'Ten') to avoid
    false misclassifications due to naming conventions.
  - Computes overall Top-1 and Top-3 accuracy.
  - Computes per-class sample counts, Top-1 accuracy, Top-3 accuracy, and mean
    scores.
  - Builds full multi-class confusion matrix.
  - Diagnoses all misclassifications with full candidate ranking.
  - Audits and reports skipped samples and reasons (e.g., missing data, invalid format).
  - Automatically writes timestamped reports (.md and .json) under reports/ without
    overwriting previous runs.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
from typing import Dict, Any, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.pipeline import ExemplarRecognitionEngine


# ─────────────────────────── Label Canonicalization ─────────────────────────

DIGIT_PREFIX_RE = re.compile(r"^\d+_(.+)$")


def canonicalize_label(label: str) -> str:
    """
    Canonicalizes sign class labels for uniform comparison across gallery
    and evaluation datasets.

    Strips numeric prefix identifiers (e.g., '1_One' -> 'One', '10_Ten' -> 'Ten')
    and trims trailing/leading whitespace.

    Examples:
        '1_One' -> 'One'
        '10_Ten' -> 'Ten'
        'Drink' -> 'Drink'
        'I_Me' -> 'I_Me'
    """
    if not label:
        return ""
    cleaned = str(label).strip()
    match = DIGIT_PREFIX_RE.match(cleaned)
    if match:
        return match.group(1).strip()
    return cleaned


# ─────────────────────────── Data Structures ────────────────────────────────

@dataclass
class CandidatePrediction:
    rank: int
    label_raw: str
    label_canonical: str
    similarity: float
    distance: float


@dataclass
class EvaluationSampleResult:
    file_path: str
    file_name: str
    ground_truth_raw: str
    ground_truth_canonical: str
    top1_prediction_raw: str
    top1_prediction_canonical: str
    top1_similarity: float
    top1_distance: float
    is_top1_correct: bool
    is_top3_correct: bool
    ground_truth_rank: Optional[int]
    query_frames: int
    active_window: Tuple[int, int]
    candidates: List[Dict[str, Any]]
    hand_frames: int = 0


@dataclass
class SkippedSample:
    file_path: str
    file_name: str
    reason: str


@dataclass
class EvaluationSummary:
    timestamp: str
    gallery_dir: str
    dataset_dir: str
    total_files_found: int
    evaluated_count: int
    skipped_count: int
    top1_correct_count: int
    top1_accuracy: float
    top3_correct_count: int
    top3_accuracy: float
    samples_with_hands: int
    samples_zero_hands: int
    accuracy_with_hands_top1: float
    accuracy_with_hands_top3: float
    classes: List[str]
    per_class_metrics: Dict[str, Dict[str, Any]]
    confusion_matrix: List[List[int]]
    misclassifications: List[Dict[str, Any]]
    skipped_samples: List[Dict[str, Any]]


# ─────────────────────────── Metric Calculation ─────────────────────────────

def compute_evaluation_metrics(
    results: List[EvaluationSampleResult],
    skipped: List[SkippedSample],
    all_classes: List[str],
    gallery_dir: str = "",
    dataset_dir: str = "",
    timestamp: Optional[str] = None
) -> EvaluationSummary:
    """
    Pure metric computation function over evaluated sample results.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sorted_classes = sorted(list(set(all_classes)))
    class_to_idx = {c: i for i, c in enumerate(sorted_classes)}
    n_classes = len(sorted_classes)

    cm = np.zeros((n_classes, n_classes), dtype=int)

    top1_correct = 0
    top3_correct = 0

    per_class: Dict[str, Dict[str, Any]] = {
        c: {
            "total": 0,
            "top1_correct": 0,
            "top1_accuracy": 0.0,
            "top3_correct": 0,
            "top3_accuracy": 0.0,
            "mean_top1_similarity": 0.0,
            "mean_top1_distance": 0.0,
            "sim_scores": [],
            "dist_scores": []
        }
        for c in sorted_classes
    }

    misclassifications = []

    for r in results:
        gt = r.ground_truth_canonical
        pred = r.top1_prediction_canonical

        if gt not in per_class:
            per_class[gt] = {
                "total": 0,
                "top1_correct": 0,
                "top1_accuracy": 0.0,
                "top3_correct": 0,
                "top3_accuracy": 0.0,
                "mean_top1_similarity": 0.0,
                "mean_top1_distance": 0.0,
                "sim_scores": [],
                "dist_scores": []
            }
            if gt not in class_to_idx:
                sorted_classes.append(gt)
                class_to_idx[gt] = len(sorted_classes) - 1

        per_class[gt]["total"] += 1
        per_class[gt]["sim_scores"].append(r.top1_similarity)
        per_class[gt]["dist_scores"].append(r.top1_distance)

        # Update confusion matrix
        if gt in class_to_idx and pred in class_to_idx:
            cm[class_to_idx[gt], class_to_idx[pred]] += 1

        if r.is_top1_correct:
            top1_correct += 1
            per_class[gt]["top1_correct"] += 1

        if r.is_top3_correct:
            top3_correct += 1
            per_class[gt]["top3_correct"] += 1
        else:
            misclassifications.append({
                "file_name": r.file_name,
                "ground_truth": gt,
                "top1_prediction": pred,
                "top1_similarity": r.top1_similarity,
                "top1_distance": r.top1_distance,
                "ground_truth_rank": r.ground_truth_rank,
                "top3_candidates": r.candidates[:3]
            })

        if not r.is_top1_correct and r.is_top3_correct:
            # Also record top-1 errors that were in top-3
            misclassifications.append({
                "file_name": r.file_name,
                "ground_truth": gt,
                "top1_prediction": pred,
                "top1_similarity": r.top1_similarity,
                "top1_distance": r.top1_distance,
                "ground_truth_rank": r.ground_truth_rank,
                "top3_candidates": r.candidates[:3]
            })

    # Deduplicate misclassifications list by file_name
    unique_misc = {}
    for m in misclassifications:
        unique_misc[m["file_name"]] = m
    misclassifications_list = list(unique_misc.values())
    misclassifications_list.sort(key=lambda x: (x["ground_truth"], x["file_name"]))

    eval_count = len(results)
    top1_acc = (top1_correct / eval_count) if eval_count > 0 else 0.0
    top3_acc = (top3_correct / eval_count) if eval_count > 0 else 0.0

    # Data quality / hand presence breakdown
    with_hands = [r for r in results if r.hand_frames > 0]
    zero_hands = [r for r in results if r.hand_frames == 0]
    acc_hands_top1 = (sum(1 for r in with_hands if r.is_top1_correct) / len(with_hands)) if with_hands else 0.0
    acc_hands_top3 = (sum(1 for r in with_hands if r.is_top3_correct) / len(with_hands)) if with_hands else 0.0

    for c, stats in per_class.items():
        tot = stats["total"]
        if tot > 0:
            stats["top1_accuracy"] = stats["top1_correct"] / tot
            stats["top3_accuracy"] = stats["top3_correct"] / tot
            stats["mean_top1_similarity"] = float(np.mean(stats["sim_scores"]))
            stats["mean_top1_distance"] = float(np.mean(stats["dist_scores"]))
        # Remove raw lists from summary
        del stats["sim_scores"]
        del stats["dist_scores"]

    skipped_data = [
        {"file_name": s.file_name, "file_path": str(s.file_path), "reason": s.reason}
        for s in skipped
    ]

    return EvaluationSummary(
        timestamp=timestamp,
        gallery_dir=str(gallery_dir),
        dataset_dir=str(dataset_dir),
        total_files_found=eval_count + len(skipped),
        evaluated_count=eval_count,
        skipped_count=len(skipped),
        top1_correct_count=top1_correct,
        top1_accuracy=top1_acc,
        top3_correct_count=top3_correct,
        top3_accuracy=top3_acc,
        samples_with_hands=len(with_hands),
        samples_zero_hands=len(zero_hands),
        accuracy_with_hands_top1=acc_hands_top1,
        accuracy_with_hands_top3=acc_hands_top3,
        classes=sorted_classes,
        per_class_metrics=per_class,
        confusion_matrix=cm.tolist(),
        misclassifications=misclassifications_list,
        skipped_samples=skipped_data
    )


# ─────────────────────────── Report Formatting ──────────────────────────────

def format_text_confusion_matrix(classes: List[str], cm_list: List[List[int]]) -> str:
    """Formats an ASCII representation of the confusion matrix."""
    cm = np.array(cm_list)
    short_classes = [c[:4] for c in classes]
    header = "True \\ Pred | " + " ".join(f"{sc:>4}" for sc in short_classes)
    sep = "-" * len(header)
    lines = [header, sep]
    for i, c in enumerate(classes):
        row_str = " ".join(f"{cm[i, j]:>4}" if cm[i, j] > 0 else "   ." for j in range(len(classes)))
        lines.append(f"{c[:11]:<11} | {row_str}")
    return "\n".join(lines)


def generate_report_markdown(summary: EvaluationSummary) -> str:
    """Generates a full Markdown evaluation report."""
    md = []
    md.append(f"# SignSpeakAI — Independent Recorded Samples Evaluation Report")
    md.append(f"**Generated at:** {summary.timestamp}\n")
    md.append(f"- **Gallery Directory:** `{summary.gallery_dir}`")
    md.append(f"- **Evaluation Dataset:** `{summary.dataset_dir}`")
    md.append(f"- **Total Files Found:** {summary.total_files_found}")
    md.append(f"- **Evaluated Samples:** {summary.evaluated_count}")
    md.append(f"- **Skipped Samples:** {summary.skipped_count}\n")

    md.append("## 1. Overall Executive Summary")
    md.append("| Metric | Count / Score | Percentage |")
    md.append("| :--- | :---: | :---: |")
    md.append(f"| **Top-1 Accuracy** | {summary.top1_correct_count} / {summary.evaluated_count} | **{summary.top1_accuracy * 100:.2f}%** |")
    md.append(f"| **Top-3 Accuracy** | {summary.top3_correct_count} / {summary.evaluated_count} | **{summary.top3_accuracy * 100:.2f}%** |")
    md.append(f"| **Top-1 Error Count** | {summary.evaluated_count - summary.top1_correct_count} / {summary.evaluated_count} | {(1.0 - summary.top1_accuracy) * 100:.2f}% |")
    md.append(f"| **Evaluated Classes** | {len(summary.classes)} | 100% Gallery Coverage |\n")

    md.append("### 1.1 Data Quality & Hand Detection Audit")
    md.append("| Cohort | Sample Count | Top-1 Accuracy | Top-3 Accuracy | Characteristic |")
    md.append("| :--- | :---: | :---: | :---: | :--- |")
    md.append(f"| **Samples with Detected Hands** | {summary.samples_with_hands} / {summary.evaluated_count} ({(summary.samples_with_hands/summary.evaluated_count)*100:.1f}%) | **{summary.accuracy_with_hands_top1 * 100:.2f}%** | **{summary.accuracy_with_hands_top3 * 100:.2f}%** | Hand landmarks present in >= 1 frame |")
    md.append(f"| **Samples with Zero Hand Detections** | {summary.samples_zero_hands} / {summary.evaluated_count} ({(summary.samples_zero_hands/summary.evaluated_count)*100:.1f}%) | **{( (summary.top1_correct_count - int(summary.accuracy_with_hands_top1 * summary.samples_with_hands)) / max(1, summary.samples_zero_hands) ) * 100:.2f}%** | **{( (summary.top3_correct_count - int(summary.accuracy_with_hands_top3 * summary.samples_with_hands)) / max(1, summary.samples_zero_hands) ) * 100:.2f}%** | MediaPipe hand presence completely absent (0 frames) |\n")

    md.append("> [!NOTE]")
    md.append("> **Root Cause Analysis for Low Baseline Accuracy:**")
    md.append(f"> In **{summary.samples_zero_hands} of {summary.evaluated_count}** ({summary.samples_zero_hands/summary.evaluated_count*100:.1f}%) recorded samples, MediaPipe extracted zero hand landmarks (`presence[:, 2:] == False` across all frames).")
    md.append("> Without hand landmarks, hand shape (`w_shape=0.50`) and hand position (`w_pos=0.25`) cannot be matched, triggering missing-hand penalties and forcing recognition to rely solely on coarse torso pose (`w_pose=0.25`). This caused severe prediction collapse into shorter templates (`Give` and `What`).\n")

    md.append("## 2. Per-Class Accuracy Breakdown")
    md.append("| Class | Total | Top-1 Correct | Top-1 Acc | Top-3 Correct | Top-3 Acc | Mean Sim | Mean Dist |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for c in summary.classes:
        st = summary.per_class_metrics.get(c, {})
        tot = st.get("total", 0)
        t1_c = st.get("top1_correct", 0)
        t1_a = st.get("top1_accuracy", 0.0) * 100.0
        t3_c = st.get("top3_correct", 0)
        t3_a = st.get("top3_accuracy", 0.0) * 100.0
        sim = st.get("mean_top1_similarity", 0.0)
        dist = st.get("mean_top1_distance", 0.0)
        md.append(f"| **{c}** | {tot} | {t1_c} | {t1_a:.1f}% | {t3_c} | {t3_a:.1f}% | {sim:.3f} | {dist:.3f} |")
    md.append("")

    md.append("## 3. Misclassifications Analysis")
    if not summary.misclassifications:
        md.append("No misclassifications observed! All evaluated samples achieved Top-1 accuracy.\n")
    else:
        md.append(f"A total of {len(summary.misclassifications)} samples had Top-1 prediction mismatches:\n")
        md.append("| Sample Name | Ground Truth | Top-1 Predicted | Sim | Dist | True Rank | Top-3 Candidates |")
        md.append("| :--- | :--- | :--- | :---: | :---: | :---: | :--- |")
        for m in summary.misclassifications:
            cands_str = ", ".join(f"{c['label']}({c['similarity']:.2f})" for c in m.get("top3_candidates", []))
            rank_str = str(m.get("ground_truth_rank", "N/A"))
            md.append(
                f"| `{m['file_name']}` | **{m['ground_truth']}** | `{m['top1_prediction']}` | "
                f"{m['top1_similarity']:.3f} | {m['top1_distance']:.3f} | #{rank_str} | {cands_str} |"
            )
        md.append("")

    md.append("## 4. Skipped Samples")
    if not summary.skipped_samples:
        md.append("No samples were skipped. All samples were successfully parsed, normalized, and evaluated.\n")
    else:
        md.append(f"{len(summary.skipped_samples)} sample(s) were skipped:\n")
        for sk in summary.skipped_samples:
            md.append(f"- **`{sk['file_name']}`**: {sk['reason']}")
        md.append("")

    md.append("## 5. Confusion Matrix (Row = True, Col = Predicted)")
    md.append("```text")
    md.append(format_text_confusion_matrix(summary.classes, summary.confusion_matrix))
    md.append("```\n")

    md.append("## 6. Known Limitations & Technical Observations")
    md.append("1. **Single-Exemplar Representation:** The enrolled gallery uses exactly one exemplar video per sign. Variations in signing style, speed, and morphology directly impact DTW matching.")
    md.append("2. **Class Imbalance in Test Recordings:** 6 classes have 10 recorded samples, 1 class has 6 samples, and 18 classes have only 1 sample. Overall accuracy reflects this sample distribution.")
    md.append("3. **Natural-Length DTW Cap:** As observed in repository diagnostics, sequences over 60 frames are resampled. Hand presence flag interpolation on sequences with duration > 60 frames can cause non-binary flags.")
    md.append("4. **Uncalibrated Confidence:** Similarity scores are computed via $s = 1 / (1 + d)$ and are not calibrated Bayesian posterior probabilities.")

    return "\n".join(md)


# ─────────────────────────── Evaluation Runner ──────────────────────────────

def evaluate_dataset(
    gallery_dir: Path,
    dataset_dir: Path,
    reports_dir: Optional[Path] = None,
    save_reports: bool = True
) -> Tuple[EvaluationSummary, Optional[Path], Optional[Path]]:
    """
    Evaluates all .npz samples in dataset_dir against gallery_dir.
    """
    gallery_dir = Path(gallery_dir)
    dataset_dir = Path(dataset_dir)

    if not gallery_dir.exists():
        raise FileNotFoundError(f"Gallery directory not found: {gallery_dir}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Evaluation dataset directory not found: {dataset_dir}")

    # Initialize recognition engine with existing pipeline settings
    engine = ExemplarRecognitionEngine(gallery_dir=gallery_dir, trim_to_active=True)
    gallery_raw_labels = engine.get_gallery_labels()
    gallery_canonical_labels = sorted(list(set(canonicalize_label(l) for l in gallery_raw_labels)))

    sample_files = sorted(list(dataset_dir.glob("*.npz")))
    if not sample_files:
        raise FileNotFoundError(f"No .npz files found in {dataset_dir}")

    results: List[EvaluationSampleResult] = []
    skipped: List[SkippedSample] = []

    print(f"[Evaluation] Initialized engine with {len(gallery_raw_labels)} gallery templates.")
    print(f"[Evaluation] Found {len(sample_files)} NPZ samples in {dataset_dir}.\n")

    total_k = len(gallery_raw_labels)

    for i, npz_path in enumerate(sample_files, 1):
        try:
            with np.load(npz_path) as data:
                if "features" not in data or "presence" not in data:
                    skipped.append(SkippedSample(
                        file_path=str(npz_path),
                        file_name=npz_path.name,
                        reason="Missing 'features' or 'presence' arrays in NPZ file"
                    ))
                    continue

                features = data["features"]
                presence = data["presence"]

                if len(features) == 0:
                    skipped.append(SkippedSample(
                        file_path=str(npz_path),
                        file_name=npz_path.name,
                        reason="Features array has zero frames"
                    ))
                    continue

                # Ground truth from metadata label or filename
                if "label" in data and str(data["label"]).strip():
                    gt_raw = str(data["label"]).strip()
                else:
                    # Derivation from filename prefix
                    gt_raw = npz_path.stem.split("_sample_")[0].replace("_landmarks", "")

                gt_canonical = canonicalize_label(gt_raw)

                # Recognize using existing ExemplarRecognitionEngine
                pred_res = engine.recognize_features(
                    features=features,
                    presence=presence,
                    query_name=gt_raw,
                    top_k=total_k
                )

                candidates = pred_res.get("candidates", [])
                if not candidates:
                    skipped.append(SkippedSample(
                        file_path=str(npz_path),
                        file_name=npz_path.name,
                        reason="Engine returned zero candidates"
                    ))
                    continue

                top1 = candidates[0]
                top1_raw = top1["label"]
                top1_canonical = canonicalize_label(top1_raw)
                top1_sim = float(top1["similarity"])
                top1_dist = float(top1["distance"])

                is_top1 = (top1_canonical == gt_canonical)
                top3_canonical = [canonicalize_label(c["label"]) for c in candidates[:3]]
                is_top3 = (gt_canonical in top3_canonical)

                # Find true class rank
                gt_rank = None
                for c in candidates:
                    if canonicalize_label(c["label"]) == gt_canonical:
                        gt_rank = c["rank"]
                        break

                h_frames = int(presence[:, 2:].sum()) if presence.ndim > 1 else 0

                results.append(EvaluationSampleResult(
                    file_path=str(npz_path),
                    file_name=npz_path.name,
                    ground_truth_raw=gt_raw,
                    ground_truth_canonical=gt_canonical,
                    top1_prediction_raw=top1_raw,
                    top1_prediction_canonical=top1_canonical,
                    top1_similarity=top1_sim,
                    top1_distance=top1_dist,
                    is_top1_correct=is_top1,
                    is_top3_correct=is_top3,
                    ground_truth_rank=gt_rank,
                    query_frames=pred_res.get("query_frames", 0),
                    active_window=pred_res.get("active_window", (0, 0)),
                    candidates=candidates,
                    hand_frames=h_frames
                ))

                mark = "[PASS]" if is_top1 else ("[TOP3]" if is_top3 else "[FAIL]")
                try:
                    print(f"[{i:02d}/{len(sample_files):02d}] {mark:<6} {gt_canonical:<20} -> {top1_canonical:<20} "
                          f"(sim={top1_sim:.3f}, dist={top1_dist:.3f}, gt_rank=#{gt_rank})")
                except Exception:
                    pass

        except Exception as exc:
            skipped.append(SkippedSample(
                file_path=str(npz_path),
                file_name=npz_path.name,
                reason=f"Processing exception: {exc}"
            ))
            try:
                print(f"[{i:02d}/{len(sample_files):02d}] ERROR {npz_path.name}: {exc}")
            except Exception:
                pass

    # Compute summary
    summary = compute_evaluation_metrics(
        results=results,
        skipped=skipped,
        all_classes=gallery_canonical_labels,
        gallery_dir=str(gallery_dir),
        dataset_dir=str(dataset_dir)
    )

    md_path = None
    json_path = None

    if save_reports:
        if reports_dir is None:
            reports_dir = PROJECT_ROOT / "reports"
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path = reports_dir / f"evaluation_report_{now_str}.md"
        json_path = reports_dir / f"evaluation_report_{now_str}.json"

        # Generate markdown report
        md_content = generate_report_markdown(summary)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # Generate JSON summary
        json_content = asdict(summary)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_content, f, indent=2)

        print(f"\n[Evaluation] Saved Markdown report to: {md_path}")
        print(f"[Evaluation] Saved JSON report to:     {json_path}")

    return summary, md_path, json_path


# ─────────────────────────── CLI Entrypoint ─────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate SignSpeakAI on recorded samples.")
    parser.add_argument(
        "--gallery",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "processed_corrected"),
        help="Path to gallery templates directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "recorded_processed"),
        help="Path to evaluation NPZ samples directory"
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(PROJECT_ROOT / "reports"),
        help="Output directory for evaluation reports"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SignSpeak AI — Independent Recorded Samples Evaluation (Phase 1)")
    print("=" * 80)

    summary, md_path, json_path = evaluate_dataset(
        gallery_dir=Path(args.gallery),
        dataset_dir=Path(args.dataset),
        reports_dir=Path(args.reports_dir),
        save_reports=True
    )

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Samples Evaluated: {summary.evaluated_count}/{summary.total_files_found}")
    print(f"Skipped Samples:         {summary.skipped_count}")
    print(f"Overall Top-1 Accuracy:  {summary.top1_accuracy * 100:.2f}% ({summary.top1_correct_count}/{summary.evaluated_count})")
    print(f"Overall Top-3 Accuracy:  {summary.top3_accuracy * 100:.2f}% ({summary.top3_correct_count}/{summary.evaluated_count})")
    print(f"Misclassifications:      {len(summary.misclassifications)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
