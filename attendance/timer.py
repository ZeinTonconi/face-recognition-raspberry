"""
attendance/timer.py — Tracks how long the same person has been in frame.
"""

import time
from config import CONFIRM_SECONDS, CONFIRM_THRESHOLD


class ConfirmationTimer:
    """
    Measures continuous presence of the same person.

    Call update(name, confidence) every time a new prediction arrives.
    Returns True the moment CONFIRM_SECONDS has elapsed with the same name
    AND the confidence is above CONFIRM_THRESHOLD the whole time.
    Call reset() when the face disappears or changes.
    """

    def __init__(self):
        self._name      : str | None = None
        self._start_time: float      = 0.0

    def update(self, name: str, confidence: float) -> bool:
        """
        Returns True when the confirmation threshold is reached.
        Resets automatically if name changes or confidence drops.
        """
        if name == "Unknown" or confidence < CONFIRM_THRESHOLD:
            self.reset()
            return False

        now = time.monotonic()
        if name != self._name:
            self._name       = name
            self._start_time = now
            return False

        return (now - self._start_time) >= CONFIRM_SECONDS

    def reset(self) -> None:
        self._name       = None
        self._start_time = 0.0

    def progress(self) -> float:
        """Elapsed seconds for the current person (0 if no one is tracked)."""
        if self._name is None:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def current_name(self) -> str | None:
        return self._name
