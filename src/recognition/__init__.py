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
from .gloss_buffer import (
    GlossSequenceBuffer,
    GlossItem,
    canonicalize_gloss
)

__all__ = [
    "NormalizedFrame",
    "NormalizedSequence",
    "normalize_sequence",
    "parse_raw_landmarks_frame",
    "frame_distance",
    "DTWSequenceMatcher",
    "PredictionResult",
    "ExemplarRecognitionEngine",
    "GlossSequenceBuffer",
    "GlossItem",
    "canonicalize_gloss"
]
