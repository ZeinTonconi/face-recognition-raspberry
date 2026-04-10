"""
core/detector.py — Face detection, cropping, and bounding-box tracking.

Uses MediaPipe for detection (fast, runs on every frame).
"""

import cv2
import numpy as np
import mediapipe as mp

from config import FACE_SIZE, IOU_THRESHOLD, MIN_DETECT_CONFIDENCE

# One shared MediaPipe instance for the whole process
_mp_det   = mp.solutions.face_detection
_detector = _mp_det.FaceDetection(
    model_selection=0,
    min_detection_confidence=MIN_DETECT_CONFIDENCE,
)


def detect_faces(frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detect all faces in a BGR frame.
    Returns a list of (x, y, w, h) boxes, or [] if none found.
    """
    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = _detector.process(rgb)
    if not results.detections:
        return []

    ih, iw = frame_bgr.shape[:2]
    boxes  = []
    for det in results.detections:
        bb = det.location_data.relative_bounding_box
        x  = max(0, int(bb.xmin * iw))
        y  = max(0, int(bb.ymin * ih))
        w  = min(iw - x, int(bb.width  * iw))
        h  = min(ih - y, int(bb.height * ih))
        if w > 0 and h > 0:
            boxes.append((x, y, w, h))
    return boxes


def crop_face(frame_bgr: np.ndarray,
              box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop and resize a face region to FACE_SIZE. Returns BGR image."""
    x, y, w, h = box
    return cv2.resize(frame_bgr[y : y + h, x : x + w], FACE_SIZE)


def iou(box_a: tuple, box_b: tuple) -> float:
    """Intersection-over-Union between two (x, y, w, h) boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix    = max(ax, bx)
    iy    = max(ay, by)
    ix2   = min(ax + aw, bx + bw)
    iy2   = min(ay + ah, by + bh)
    inter = max(0, ix2 - ix) * max(0, iy2 - iy)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def select_tracked_box(
    candidates: list[tuple],
    prev_box: tuple | None,
) -> tuple | None:
    """
    Pick the best box to keep tracking.
    - No previous box → pick the largest face.
    - Previous box    → pick candidate with highest IoU (if above threshold).
    """
    if not candidates:
        return None
    if prev_box is None:
        areas = [w * h for (_, _, w, h) in candidates]
        return candidates[int(np.argmax(areas))]

    ious = [iou(prev_box, c) for c in candidates]
    best = int(np.argmax(ious))
    return candidates[best] if ious[best] > IOU_THRESHOLD else None


def box_centre(box: tuple) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2, y + h / 2


def box_distance(a: tuple | None, b: tuple | None) -> float:
    """Euclidean distance between the centres of two boxes."""
    if a is None or b is None:
        return float("inf")
    ax, ay = box_centre(a)
    bx, by = box_centre(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
