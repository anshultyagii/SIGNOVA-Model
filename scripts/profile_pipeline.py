# -*- coding: utf-8 -*-
"""
SignSpeak AI - Pipeline Profiler
================================
Measures latency and throughput across:
  1. Image decoding & format conversion
  2. MediaPipe Holistic feature extraction
  3. Posture analysis & motion energy calculation
  4. DTW Exemplar gallery matching
  5. T5 neural model vs rule translator inference
"""

import time
import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recognition.pipeline import ExemplarRecognitionEngine
from src.recognition.stream_adapter import LiveWebcamRecognitionSession
from src.translation.model_translator import ConfigurableISLTranslator


def profile_pipeline():
    print("=" * 60, flush=True)
    print("SignSpeakAI Recognition & Translation Profiling Report", flush=True)
    print("=" * 60, flush=True)

    # 1. Pipeline Initialization
    t0 = time.perf_counter()
    gallery_dir = Path(__file__).resolve().parent.parent / "dataset" / "processed_corrected"
    engine = ExemplarRecognitionEngine(gallery_dir=gallery_dir if gallery_dir.exists() else None)
    session = LiveWebcamRecognitionSession(engine=engine)
    init_time = (time.perf_counter() - t0) * 1000
    print(f"[Init] Recognition engine & session loaded in {init_time:.2f} ms", flush=True)

    # Create dummy 640x480 BGR webcam frame
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    # Draw simple human silhouette so detectors engage
    cv2.circle(frame, (320, 150), 40, (200, 200, 200), -1)  # Head
    cv2.rectangle(frame, (260, 190), (380, 350), (160, 160, 160), -1)  # Torso
    cv2.circle(frame, (220, 280), 20, (180, 180, 180), -1)  # Left hand
    cv2.circle(frame, (420, 280), 20, (180, 180, 180), -1)  # Right hand

    # Warmup
    session.process_frame(frame)

    # 2. Frame Processing Benchmark (20 iterations)
    proc_times = []
    for _ in range(20):
        t_start = time.perf_counter()
        session.process_frame(frame)
        proc_times.append((time.perf_counter() - t_start) * 1000)

    mean_proc = np.mean(proc_times)
    median_proc = np.median(proc_times)
    p95_proc = np.percentile(proc_times, 95)
    fps = 1000.0 / mean_proc
    print(f"[Frame Processing Stage]", flush=True)
    print(f"  Median Latency : {median_proc:.2f} ms", flush=True)
    print(f"  Mean Latency   : {mean_proc:.2f} ms", flush=True)
    print(f"  P95 Latency    : {p95_proc:.2f} ms", flush=True)
    print(f"  Throughput     : {fps:.1f} frames/sec", flush=True)

    # 3. DTW Matching Benchmark (Cached Gallery)
    if engine.gallery_loaded:
        sample_feat = np.random.randn(25, 1629).astype(np.float32)
        sample_pres = np.ones((25, 4), dtype=bool)
        dtw_times = []
        for _ in range(25):
            t_start = time.perf_counter()
            engine.recognize_features(sample_feat, sample_pres, top_k=3)
            dtw_times.append((time.perf_counter() - t_start) * 1000)
        mean_dtw = np.mean(dtw_times)
        median_dtw = np.median(dtw_times)
        p95_dtw = np.percentile(dtw_times, 95)
        dtw_throughput = 1000.0 / mean_dtw
        print(f"\n[DTW Exemplar Matcher (25 Gallery Templates)]", flush=True)
        print(f"  Median Latency : {median_dtw:.2f} ms", flush=True)
        print(f"  Mean Latency   : {mean_dtw:.2f} ms", flush=True)
        print(f"  P95 Latency    : {p95_dtw:.2f} ms", flush=True)
        print(f"  Throughput     : {dtw_throughput:.1f} matches/sec", flush=True)

    # 4. Translator Inference Benchmark
    translator = ConfigurableISLTranslator(mode="model")
    seq = ["I_Me", "Want_Desire_Wish_For", "Water"]

    # Neural inference
    t5_times = []
    for _ in range(15):
        t_start = time.perf_counter()
        translator.translate_detailed(seq)
        t5_times.append((time.perf_counter() - t_start) * 1000)
    mean_t5 = np.mean(t5_times)
    median_t5 = np.median(t5_times)
    p95_t5 = np.percentile(t5_times, 95)
    t5_throughput = 1000.0 / mean_t5
    print(f"\n[T5-Small Neural Translator (Fine-Tuned v2)]", flush=True)
    print(f"  Median Latency : {median_t5:.2f} ms", flush=True)
    print(f"  Mean Latency   : {mean_t5:.2f} ms", flush=True)
    print(f"  P95 Latency    : {p95_t5:.2f} ms", flush=True)
    print(f"  Throughput     : {t5_throughput:.1f} sentences/sec", flush=True)

    # Rule inference
    translator.set_mode("rules")
    rule_times = []
    for _ in range(25):
        t_start = time.perf_counter()
        translator.translate_detailed(seq)
        rule_times.append((time.perf_counter() - t_start) * 1000)
    mean_rule = np.mean(rule_times)
    median_rule = np.median(rule_times)
    p95_rule = np.percentile(rule_times, 95)
    rule_throughput = 1000.0 / mean_rule
    print(f"\n[Rule-Based Translator Baseline]", flush=True)
    print(f"  Median Latency : {median_rule:.4f} ms", flush=True)
    print(f"  Mean Latency   : {mean_rule:.4f} ms", flush=True)
    print(f"  P95 Latency    : {p95_rule:.4f} ms", flush=True)
    print(f"  Throughput     : {rule_throughput:.0f} sentences/sec", flush=True)

    # End-to-end estimated sign recognition pipeline
    e2e_mean = mean_proc + mean_dtw + (mean_t5 / 3.0)  # translation debounced per sign
    print(f"\n[End-to-End Pipeline Performance]", flush=True)
    print(f"  Est. E2E Latency per Gesture : {mean_proc * 15 + mean_dtw + mean_t5:.2f} ms", flush=True)
    print(f"  Max Real-Time Throughput     : {min(fps, dtw_throughput):.1f} signs/sec equivalent", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    profile_pipeline()
