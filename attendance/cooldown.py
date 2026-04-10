"""
attendance/cooldown.py — Prevents recording the same person multiple times.
"""

import time
from config import COOLDOWN_MINUTES


class CooldownRegistry:
    """
    Tracks when each person was last recorded.

    A person is "in cooldown" for COOLDOWN_MINUTES after being recorded,
    preventing duplicate attendance entries when they linger in front of
    the camera.
    """

    def __init__(self):
        self._timestamps: dict[str, float] = {}

    def in_cooldown(self, name: str) -> bool:
        if name not in self._timestamps:
            return False
        elapsed = time.monotonic() - self._timestamps[name]
        return elapsed < COOLDOWN_MINUTES * 60

    def record(self, name: str) -> None:
        """Mark a person as just recorded."""
        self._timestamps[name] = time.monotonic()

    def remaining_seconds(self, name: str) -> float:
        """How many seconds of cooldown remain for a person (0 if none)."""
        if name not in self._timestamps:
            return 0.0
        elapsed = time.monotonic() - self._timestamps[name]
        return max(0.0, COOLDOWN_MINUTES * 60 - elapsed)
