"""
workers/embed_worker.py — Runs face embedding + KNN in a separate process.

Using a process (not a thread) means dlib's heavy C++ work never competes
with the main loop for the GIL, keeping the camera preview at full fps.
"""

import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def _embed_and_predict(
    face_bgr_bytes: bytes,
    shape: tuple,
    model_file: str,
    unknown_threshold: float,
    embedding_model: str,
) -> tuple[str | None, float]:
    """
    Runs inside a worker process.
    Receives the face as raw bytes (picklable), returns (name, confidence).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import cv2
    import numpy as np
    import pickle
    import face_recognition

    face     = np.frombuffer(face_bgr_bytes, dtype=np.uint8).reshape(shape)
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    h, w     = face_rgb.shape[:2]

    encodings = face_recognition.face_encodings(
        face_rgb,
        known_face_locations=[(0, w, h, 0)],
        num_jitters=1,
        model=embedding_model,
    )
    if not encodings:
        return None, 0.0

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    vec = np.array(encodings[0], dtype="float32").reshape(1, -1)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        idx   = int(np.argmax(probs))
        return str(model.classes_[idx]), float(probs[idx]) * 100.0
    return str(model.predict(vec)[0]), 100.0


class EmbedWorker:
    """
    Wraps a single-worker ProcessPoolExecutor.

    Usage:
        worker = EmbedWorker()
        worker.submit(face_crop, token, on_result_callback)
        # on_result(name, conf, token) called when done
        worker.shutdown()
    """

    def __init__(self):
        self._executor = ProcessPoolExecutor(max_workers=1)
        self._future   = None

    def busy(self) -> bool:
        return self._future is not None and not self._future.done()

    def submit(
        self,
        face_bgr: np.ndarray,
        token: int,
        on_result,          # callable(name, conf, token)
        model_file: str,
        unknown_threshold: float,
        embedding_model: str,
    ) -> None:
        """Submit a face crop for embedding. Ignored if worker is busy."""
        if self.busy():
            return

        self._future = self._executor.submit(
            _embed_and_predict,
            face_bgr.tobytes(),
            face_bgr.shape,
            model_file,
            unknown_threshold,
            embedding_model,
        )

        def _done(fut, t=token):
            try:
                name, conf = fut.result()
                on_result(name, conf, t)
            except Exception:
                pass

        self._future.add_done_callback(_done)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
