"""
SignSpeak AI - Recognition Package
Exemplar-based Indian Sign Language gesture recognition engine.
"""

from .features import (
    NormalizedFrame,
    NormalizedSequence,
    normalize_sequence,
    parse_raw_landmarks_frame,
    frame_distance
)
from .matcher import (
    DTWSequenceMatcher,
    PredictionResult
)
from .pipeline import (
    ExemplarRecognitionEngine
)

__all__ = [
    "NormalizedFrame",
    "NormalizedSequence",
    "normalize_sequence",
    "parse_raw_landmarks_frame",
    "frame_distance",
    "DTWSequenceMatcher",
    "PredictionResult",
    "ExemplarRecognitionEngine"
]
