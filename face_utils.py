"""
face_utils.py — Shared utilities for the face recognition attendance system.

Imported by every other script. Contains NO business logic — only reusable
primitives for detection, embedding, tracking, and display.

Dependencies: opencv-python, mediapipe, face_recognition, numpy
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
import cv2
import numpy as np
import mediapipe as mp
import face_recognition

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FACE_SIZE             = (160, 160)
IOU_THRESHOLD         = 0.45
MIN_DETECT_CONFIDENCE = 0.5

# ---------------------------------------------------------------------------
# MediaPipe detector — one shared instance for the whole process
# ---------------------------------------------------------------------------
_mp_det   = mp.solutions.face_detection
_detector = _mp_det.FaceDetection(
    model_selection=0,
    min_detection_confidence=MIN_DETECT_CONFIDENCE,
)

log = logging.getLogger(__name__)


# ===========================================================================
# Face detection & cropping
# ===========================================================================

def detect_faces(frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detect all faces in a BGR frame.

    Returns a list of (x, y, w, h) bounding boxes in pixel coordinates,
    or an empty list if no face is found.
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
    """
    Crop the face region from frame and resize it to FACE_SIZE.

    Returns a BGR image of shape (FACE_SIZE[1], FACE_SIZE[0], 3).
    """
    x, y, w, h = box
    crop = frame_bgr[y : y + h, x : x + w]
    return cv2.resize(crop, FACE_SIZE)


# ===========================================================================
# Embedding — face_recognition (dlib ResNet, 128-dim)
# Works efficiently on CPU, including Raspberry Pi 4.
# ===========================================================================

def get_embedding(face_bgr: np.ndarray) -> np.ndarray | None:
    """
    Compute a 128-dim embedding for a BGR face crop using dlib ResNet.

    Returns a 1-D float32 array of shape (128,), or None on failure.
    The image is converted BGR → RGB before being passed to face_recognition.
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(
        face_rgb,
        known_face_locations=[(0, face_rgb.shape[1], face_rgb.shape[0], 0)],
        num_jitters=1,
        model="small",   # "small" is faster; use "large" for higher accuracy
    )
    if not encodings:
        log.warning("face_recognition returned no encoding.")
        return None
    return np.array(encodings[0], dtype="float32")


# ===========================================================================
# Bounding-box tracking
# ===========================================================================

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
    Choose which detected box to continue tracking.

    - No previous box  →  pick the largest face (closest to camera).
    - Previous box     →  pick candidate with highest IoU, only if it
                          exceeds IOU_THRESHOLD; otherwise return None.
    """
    if not candidates:
        return None
    if prev_box is None:
        areas = [w * h for (_, _, w, h) in candidates]
        return candidates[int(np.argmax(areas))]

    ious = [iou(prev_box, c) for c in candidates]
    best = int(np.argmax(ious))
    return candidates[best] if ious[best] > IOU_THRESHOLD else None


# ===========================================================================
# On-screen drawing helpers
# ===========================================================================

_GREEN     = (0,   210,   0)
_YELLOW    = (0,   210, 255)
_RED       = (0,     0, 220)
_BLUE      = (255,  140,  0)
_WHITE     = (255, 255, 255)
_BLACK     = (0,     0,   0)
_FONT      = cv2.FONT_HERSHEY_SIMPLEX


def _text_with_bg(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float   = 0.65,
    thickness: int = 1,
    fg: tuple      = _WHITE,
    bg: tuple      = _BLACK,
) -> None:
    """Draw text with a filled background rectangle for legibility."""
    (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    x, y = origin
    pad  = 4
    cv2.rectangle(frame,
                  (x - pad,      y - th - pad),
                  (x + tw + pad, y + baseline + pad),
                  bg, cv2.FILLED)
    cv2.putText(frame, text, (x, y), _FONT, scale, fg, thickness, cv2.LINE_AA)


def draw_face_box(
    frame:      np.ndarray,
    box:        tuple[int, int, int, int],
    label:      str,
    confidence: float | None = None,
    known:      bool         = True,
    confirmed:  bool         = False,
) -> None:
    """
    Draw a bounding box and name label on *frame* in-place.

    Args:
        box:        (x, y, w, h)
        label:      Person name or "Unknown"
        confidence: 0–100 float appended to label when provided
        known:      Green box for recognised faces, red for unknown
        confirmed:  Blue box when attendance has been recorded
    """
    x, y, w, h = box

    if confirmed:
        color = _BLUE
    elif known:
        color = _GREEN
    else:
        color = _RED

    text = f"{label}  {confidence:.1f}%" if confidence is not None else label
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    _text_with_bg(frame, text, (x, max(y - 8, 14)), fg=_YELLOW, bg=_BLACK)


def draw_hud(frame: np.ndarray, lines: list[str]) -> None:
    """
    Render instruction / status lines in the top-left corner of *frame*.
    Each string in *lines* appears on its own row.
    """
    for i, line in enumerate(lines):
        _text_with_bg(frame, line, (10, 24 + i * 26),
                      scale=0.6, fg=_WHITE, bg=(30, 30, 30))


# ===========================================================================
# Camera utilities
# ===========================================================================

def list_cameras(max_index: int = 10) -> list[int]:
    """Return indices of all responding cameras (up to max_index)."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
        cap.release()
    return available


def pick_camera() -> int:
    """
    Print available cameras and let the user choose one interactively.
    Returns the chosen index, defaulting to 0 on bad input.
    """
    available = list_cameras()
    if not available:
        raise RuntimeError("No cameras found. Check your hardware.")
    print("Available cameras:", available)
    raw=0
    # raw = input("Camera index to use (default 0): ").strip()
    try:
        idx = int(raw)
    except ValueError:
        idx = 0
    if idx not in available:
        print(f"Index {idx} not available — defaulting to 0.")
        idx = 0
    return idx