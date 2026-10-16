# -*- coding: utf-8 -*-
"""
SignSpeak AI - Gloss Sequence Buffer Module
===========================================
Accumulates individually recognized ISL signs into an ordered gloss sequence.

Features:
  - Ordered storage of recognized sign tokens with timestamps and confidence scores.
  - Duplicate suppression: prevents spurious repeated additions of the same ongoing sign.
  - Idle finalization: automatically completes and emits a sequence after a configurable
    idle / rest duration (default 1.5s).
  - Validation: rejects empty labels, invalid markers ('-', 'unknown'), and low-confidence
    predictions below a configurable threshold.
  - Deterministic time injection: supports explicit timestamps or a custom time provider
    for testing.
  - Clean state management: explicit add, check_idle, finalize, and reset methods.
"""

from dataclasses import dataclass
import re
import time
from typing import Callable, List, Optional

# Regex to canonicalize labels (e.g. '1_One' -> 'One', '10_Ten' -> 'Ten')
DIGIT_PREFIX_RE = re.compile(r"^\d+_(.+)$")


def canonicalize_gloss(label: str) -> str:
    """Strips leading numeric prefix and trims whitespace."""
    if not label:
        return ""
    cleaned = str(label).strip()
    match = DIGIT_PREFIX_RE.match(cleaned)
    if match:
        return match.group(1).strip()
    return cleaned


@dataclass
class GlossItem:
    """Represents a single recognized gloss token in a sequence."""
    gloss: str
    confidence: float = 1.0
    timestamp: float = 0.0

    def to_dict(self):
        return {
            "gloss": self.gloss,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }


class GlossSequenceBuffer:
    """
    Buffers recognized sign tokens into an ordered gloss sequence.
    Emits completed sentences after a pause/idle timeout.
    """

    def __init__(
        self,
        idle_timeout_seconds: float = 1.5,
        min_confidence: float = 0.50,
        duplicate_cooldown_seconds: float = 1.2,
        canonicalize: bool = True,
        time_provider: Optional[Callable[[], float]] = None
    ):
        """
        Args:
            idle_timeout_seconds: Idle period (seconds) after which current sequence is finalized.
            min_confidence: Minimum similarity score required to accept a prediction.
            duplicate_cooldown_seconds: Minimum time (seconds) required before the same gloss
                                       can be added again consecutively.
            canonicalize: Whether to strip numeric prefixes ('1_One' -> 'One').
            time_provider: Callable returning current time (defaults to time.time).
        """
        self.idle_timeout_seconds = float(idle_timeout_seconds)
        self.min_confidence = float(min_confidence)
        self.duplicate_cooldown_seconds = float(duplicate_cooldown_seconds)
        self.canonicalize = canonicalize
        self._time_provider = time_provider or time.time

        self._current_sequence: List[GlossItem] = []
        self._completed_sequences: List[List[str]] = []
        self._last_added_time: float = 0.0
        self._last_activity_time: float = 0.0
        self._gesture_boundary_active: bool = False

    def mark_gesture_boundary(self):
        """
        Marks that a new distinct gesture onset has occurred.
        Allows the next accepted prediction to be recorded even if identical
        to the preceding gloss, distinguishing deliberate consecutive repetitions
        from frame-level burst duplicates.
        """
        self._gesture_boundary_active = True

    def _now(self, timestamp: Optional[float] = None) -> float:
        """Returns injected timestamp or current time."""
        if timestamp is not None:
            return float(timestamp)
        return float(self._time_provider())

    def add_gloss(
        self,
        label: Optional[str],
        confidence: float = 1.0,
        timestamp: Optional[float] = None
    ) -> bool:
        """
        Attempts to add a recognized sign to the active sequence.

        Returns:
            bool: True if the gloss was added; False if rejected (invalid, low confidence,
                  or duplicate).
        """
        t = self._now(timestamp)

        # 1. Validate label
        if label is None:
            return False
        raw_label = str(label).strip()
        if not raw_label or raw_label in ("-", "unknown", "None", "PAUSED"):
            return False

        clean_label = canonicalize_gloss(raw_label) if self.canonicalize else raw_label
        if not clean_label:
            return False

        # 2. Check confidence threshold
        if confidence < self.min_confidence:
            return False

        # 3. Duplicate suppression:
        # If the same gloss is added consecutively without an intervening gesture boundary,
        # and within duplicate_cooldown_seconds, reject it as a burst duplicate.
        if self._current_sequence and not self._gesture_boundary_active:
            last_item = self._current_sequence[-1]
            if last_item.gloss == clean_label:
                elapsed_since_same = t - last_item.timestamp
                if elapsed_since_same < self.duplicate_cooldown_seconds:
                    # Update activity time so sign duration is tracked, but do not duplicate
                    self._last_activity_time = t
                    return False

        # 4. Add to current sequence
        item = GlossItem(gloss=clean_label, confidence=float(confidence), timestamp=t)
        self._current_sequence.append(item)
        self._last_added_time = t
        self._last_activity_time = t
        self._gesture_boundary_active = False
        return True

    def check_idle_timeout(self, timestamp: Optional[float] = None) -> Optional[List[str]]:
        """
        Checks if the buffer has been idle longer than idle_timeout_seconds.
        If so, finalizes and returns the completed gloss list; otherwise returns None.
        """
        if not self._current_sequence:
            return None

        t = self._now(timestamp)
        if (t - self._last_activity_time) >= self.idle_timeout_seconds:
            return self.finalize()

        return None

    def record_activity(self, timestamp: Optional[float] = None):
        """
        Notifies the buffer of ongoing signing activity (e.g. hands active in frame),
        preventing premature idle timeout while the signer is preparing the next sign.
        """
        self._last_activity_time = self._now(timestamp)

    def finalize(self) -> List[str]:
        """
        Manually finalizes the current sequence, stores it in history, and returns it.
        """
        if not self._current_sequence:
            return []

        emitted = [item.gloss for item in self._current_sequence]
        self._completed_sequences.append(emitted)
        self._current_sequence = []
        return emitted

    def remove_last_gloss(self) -> Optional[GlossItem]:
        """
        Removes and returns the most recently added gloss from the active sequence.
        Safely updates timestamps and returns None if the sequence is empty.
        """
        if not self._current_sequence:
            return None

        removed = self._current_sequence.pop()
        if self._current_sequence:
            self._last_added_time = self._current_sequence[-1].timestamp
            self._last_activity_time = self._current_sequence[-1].timestamp
        else:
            self._last_added_time = 0.0
            self._last_activity_time = 0.0
        self._gesture_boundary_active = False
        return removed

    def reset(self):
        """Clears the active sequence and activity timers."""
        self._current_sequence.clear()
        self._last_added_time = 0.0
        self._last_activity_time = 0.0
        self._gesture_boundary_active = False

    def clear_history(self):
        """Clears both current sequence and history of completed sequences."""
        self.reset()
        self._completed_sequences.clear()

    @property
    def current_sequence(self) -> List[str]:
        """Returns the list of gloss names currently in the active buffer."""
        return [item.gloss for item in self._current_sequence]

    @property
    def current_items(self) -> List[GlossItem]:
        """Returns the list of GlossItem objects in the active buffer."""
        return list(self._current_sequence)

    @property
    def last_gloss(self) -> Optional[str]:
        """Returns the most recently added gloss in the active buffer, or None."""
        if self._current_sequence:
            return self._current_sequence[-1].gloss
        return None

    @property
    def completed_sequences(self) -> List[List[str]]:
        """Returns history of all emitted sequences."""
        return list(self._completed_sequences)

    @property
    def last_completed_sequence(self) -> Optional[List[str]]:
        """Returns the most recently finalized sequence, or None."""
        if self._completed_sequences:
            return self._completed_sequences[-1]
        return None

    @property
    def is_empty(self) -> bool:
        """True if the active sequence has no glosses."""
        return len(self._current_sequence) == 0

    def __len__(self) -> int:
        return len(self._current_sequence)

    def __repr__(self) -> str:
        seq = " -> ".join(self.current_sequence) if self._current_sequence else "(empty)"
        return f"GlossSequenceBuffer(active=[{seq}], completed={len(self._completed_sequences)})"
