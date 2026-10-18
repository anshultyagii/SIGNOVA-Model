"""
Script to test the trained YOLOv10-ST model and NLP Sentence Translator on test videos.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import torch
from scripts.realtime_isl_translator import RealTimeISLTranslator

def test_on_video(video_filename="Eat2.mp4"):
    raw_video_path = os.path.join(PROJECT_ROOT, "dataset_raw", video_filename)
    if not os.path.exists(raw_video_path):
        print(f"Error: {raw_video_path} not found.")
        return
        
    weights_path = os.path.join(PROJECT_ROOT, "runs", "yolov10_st_isl", "weights", "best.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(PROJECT_ROOT, "runs", "yolov10_st_isl", "weights", "last.pt")
        
    print(f"Testing on: {raw_video_path}")
    print(f"Using weights: {weights_path}")
    
    translator = RealTimeISLTranslator(weights_path=weights_path if os.path.exists(weights_path) else os.path.join(PROJECT_ROOT, "models", "gesture_st_classifier.pt"))
    
    os.makedirs(os.path.join(PROJECT_ROOT, "runs", "test_outputs"), exist_ok=True)
    out_video_path = os.path.join(PROJECT_ROOT, "runs", "test_outputs", f"annotated_{video_filename}")
    
    translator.run_video_file(raw_video_path, output_path=out_video_path)
    print(f"Final recognized sign words: {translator.confirmed_words}")
    print(f"Generated NLP Sentence: \"{translator.current_sentence}\"")
    print(f"Saved video to: {out_video_path}")

if __name__ == "__main__":
    test_on_video("Eat2.mp4")
