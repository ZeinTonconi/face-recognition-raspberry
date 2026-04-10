"""
core/embedder.py — Face embedding using dlib / face_recognition.

get_embedding() is called both from the main process (build_dataset.py)
and from the worker process (workers/embed_worker.py).
"""

import logging
import cv2
import numpy as np
import face_recognition

from config import EMBEDDING_MODEL

log = logging.getLogger(__name__)


def get_embedding(face_bgr: np.ndarray) -> np.ndarray | None:
    """
    Compute a 128-dim embedding for a BGR face crop.
    Returns a float32 array of shape (128,), or None on failure.
    """
    face_rgb  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    h, w      = face_rgb.shape[:2]
    encodings = face_recognition.face_encodings(
        face_rgb,
        known_face_locations=[(0, w, h, 0)],
        num_jitters=1,
        model=EMBEDDING_MODEL,
    )
    if not encodings:
        log.warning("face_recognition returned no encoding.")
        return None
    return np.array(encodings[0], dtype="float32")
