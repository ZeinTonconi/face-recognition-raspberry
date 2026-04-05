"""
recognize_live.py — Real-time face recognition, throttled to model speed.

Design: single synchronous loop.
  1. Read frame from camera.
  2. Detect face.
  3. If a face is present AND it has moved enough → run DeepFace + KNN.
  4. Clear the label immediately when no face is detected.
  5. Display result and wait for next frame.

This means the label is ALWAYS the result of the current frame's prediction —
no stale labels from a previous person. The tradeoff is that the loop runs at
~3-5 fps (limited by DeepFace), which is perfectly acceptable for an
attendance system where people walk past a camera one at a time.

Controls:  ESC or Q to quit.

Tuning:
    UNKNOWN_THRESHOLD   min KNN confidence % to show a name (default 55)
    BOX_MOVE_THRESHOLD  pixels a box must move to re-run the model (default 15)
                        set lower to re-run more often, higher to skip small movements
"""

import os
import sys
import platform
import pickle

import numpy as np
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from face_utils import (
    detect_faces,
    crop_face,
    get_embedding,
    select_tracked_box,
    draw_face_box,
    draw_hud,
    pick_camera,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_FILE         = os.path.join("models", "knn_model.pkl")
UNKNOWN_THRESHOLD  = 55.0   # confidence % — below this → "Unknown"
BOX_MOVE_THRESHOLD = 15     # pixels — skip model if face barely moved

_CAP_BACKEND = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model():
    if not os.path.exists(MODEL_FILE):
        print(f"Model not found: {MODEL_FILE}")
        print("Run train_model.py first.")
        sys.exit(1)
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded — classes: {model.classes_.tolist()}")
    return model


def predict(model, embedding: np.ndarray) -> tuple[str, float]:
    vec = embedding.reshape(1, -1)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        idx   = int(np.argmax(probs))
        return model.classes_[idx], float(probs[idx]) * 100.0
    return str(model.predict(vec)[0]), 100.0


# ---------------------------------------------------------------------------
# Box helpers
# ---------------------------------------------------------------------------

def box_centre(box: tuple) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2, y + h / 2


def box_moved(a: tuple | None, b: tuple | None,
              threshold: int = BOX_MOVE_THRESHOLD) -> bool:
    if a is None or b is None:
        return True
    ax, ay = box_centre(a)
    bx, by = box_centre(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 > threshold


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    model = load_model()

    camera_idx = pick_camera()
    cap = cv2.VideoCapture(camera_idx, _CAP_BACKEND)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always grab the freshest frame

    tracked_box    = None
    last_pred_box  = None   # box position when we last ran the model
    current_name   = None   # None means no face present
    current_conf   = None
    current_known  = True

    print("Live recognition running — press ESC or Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read error.")
            break

        # ── Detect & track ───────────────────────────────────────────────
        boxes       = detect_faces(frame)
        tracked_box = select_tracked_box(boxes, tracked_box)

        if tracked_box is None:
            # No face in frame — clear label immediately
            current_name  = None
            current_conf  = None
            current_known = True
            last_pred_box = None
        else:
            # Run model only when the face has moved enough to matter
            if box_moved(tracked_box, last_pred_box):
                face = crop_face(frame, tracked_box)
                emb  = get_embedding(face)

                if emb is not None:
                    name, conf = predict(model, emb)
                    if conf >= UNKNOWN_THRESHOLD:
                        current_name  = name
                        current_conf  = conf
                        current_known = True
                    else:
                        current_name  = "Unknown"
                        current_conf  = None
                        current_known = False

                last_pred_box = tracked_box

        # ── Draw ─────────────────────────────────────────────────────────
        if tracked_box is not None and current_name is not None:
            draw_face_box(frame, tracked_box,
                          current_name, current_conf, current_known)
        elif tracked_box is not None:
            # Face detected but model not run yet on this position
            draw_face_box(frame, tracked_box, "...", None, True)

        draw_hud(frame, ["ESC / Q to quit"])
        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
