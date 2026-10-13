"""
SignSpeak AI - High-Level Recognition Pipeline Engine
Provides a modular facade for loading the template gallery,
processing raw query inputs (from .npz or live streams), and returning predictions.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from .features import (
    normalize_sequence,
    NormalizedSequence
)
from .matcher import (
    DTWSequenceMatcher,
    PredictionResult
)


class ExemplarRecognitionEngine:
    """
    Modular engine for exemplar-based sign language recognition.
    Independent of the input source (files or real-time camera).
    """

    def __init__(
        self,
        gallery_dir: Optional[Path] = None,
        trim_to_active: bool = True,
        w_pose: float = 0.25,
        w_shape: float = 0.50,
        w_pos: float = 0.25,
        missing_hand_penalty: float = 2.0
    ):
        self.matcher = DTWSequenceMatcher(
            w_pose=w_pose,
            w_shape=w_shape,
            w_pos=w_pos,
            missing_hand_penalty=missing_hand_penalty
        )
        self.trim_to_active = trim_to_active
        self.gallery_loaded = False

        if gallery_dir is not None:
            self.load_gallery(gallery_dir)

    def load_gallery(self, gallery_dir: Path) -> int:
        """Loads and pre-processes all reference templates."""
        count = self.matcher.load_gallery_from_npz_dir(
            gallery_dir,
            trim_to_active=self.trim_to_active
        )
        self.gallery_loaded = True
        return count

    def get_gallery_labels(self) -> List[str]:
        """Returns the list of enrolled sign labels."""
        return sorted(list(self.matcher.templates.keys()))

    def recognize_features(
        self,
        features: np.ndarray,
        presence: np.ndarray,
        query_name: str = "query",
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Recognizes sign from raw feature arrays (T, 1629) and presence flags (T, 4).
        """
        if not self.gallery_loaded:
            raise RuntimeError("Gallery has not been loaded. Call load_gallery first.")

        norm_query = normalize_sequence(
            features=features,
            presence=presence,
            label=query_name,
            video_name=query_name,
            trim_to_active=self.trim_to_active
        )

        candidates = self.matcher.predict(norm_query, top_k=top_k)

        return {
            "query_name": query_name,
            "query_frames": norm_query.length,
            "active_window": norm_query.active_window,
            "top_prediction": candidates[0].label if candidates else None,
            "candidates": [c.to_dict() for c in candidates]
        }

    def recognize_npz_file(
        self,
        npz_path: Path,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Recognizes sign from a saved .npz feature file.
        """
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")

        with np.load(path) as data:
            features = data["features"]
            presence = data["presence"]
            label = str(data["label"]) if "label" in data else path.stem.replace("_landmarks", "")

        return self.recognize_features(
            features=features,
            presence=presence,
            query_name=label,
            top_k=top_k
        )
