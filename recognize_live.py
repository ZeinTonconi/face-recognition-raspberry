"""
recognize_live.py — Real-time face recognition with attendance recording.

Architecture:
  - Main thread       : camera read + MediaPipe detection + drawing (~30 fps)
  - Separate process  : face_recognition embedding + KNN (no GIL contention)
  - HTTP thread       : POST requests to attendance server (non-blocking)

Flow:
  1. Face detected + identified with enough confidence.
  2. Confirmation timer starts (green progress bar under the box).
  3. Same person held in frame for CONFIRM_SECONDS → POST to server.
  4. Box turns blue, "Recorded" shown on screen for a few seconds.
  5. Person enters cooldown for COOLDOWN_MINUTES — won't be sent again.
  6. Face disappears or jumps → label clears instantly, timer resets.

Controls:  ESC or Q to quit.

Tuning:
  UNKNOWN_THRESHOLD    min KNN confidence % to count as a real match
  CONFIRM_SECONDS      seconds same person must stay in frame to trigger POST
  COOLDOWN_MINUTES     minutes before same person can be recorded again
  BOX_JUMP_THRESHOLD   pixel jump that resets the confirmation timer
  BOX_MOVE_THRESHOLD   pixel movement needed to submit a new embedding
  SERVER_URL           endpoint to POST attendance to
  DEVICE_ID            identifier for this camera / Pi unit
"""

import os
import sys
import platform
import pickle
import threading
import queue
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
import cv2
import requests

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from face_utils import (
    detect_faces,
    crop_face,
    select_tracked_box,
    draw_face_box,
    draw_hud,
    pick_camera,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_FILE         = os.path.join("models", "knn_model.pkl")
UNKNOWN_THRESHOLD  = 55.0
CONFIRM_THRESHOLD  = 75.0
CONFIRM_SECONDS    = 3.0
COOLDOWN_MINUTES   = 10
BOX_JUMP_THRESHOLD = 80
BOX_MOVE_THRESHOLD = 40    # higher = fewer submissions = smoother fps
SERVER_URL         = "http://localhost:8000/attendance"
DEVICE_ID          = "windows-dev-1"
HTTP_TIMEOUT       = 5

_CAP_BACKEND = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2


# ---------------------------------------------------------------------------
# Process worker function — runs in a completely separate process.
# No GIL contention with the main loop whatsoever.
# Loads the model fresh each call (cheap for KNN, unavoidable with processes).
# ---------------------------------------------------------------------------

def _embed_and_predict(face_bgr_bytes: bytes,
                       shape: tuple,
                       model_file: str,
                       unknown_threshold: float):
    """
    Called in a worker process.
    Receives face image as raw bytes (picklable), returns (name, confidence).
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import numpy as np
    import pickle
    import cv2
    import face_recognition

    face = np.frombuffer(face_bgr_bytes, dtype=np.uint8).reshape(shape)
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    encodings = face_recognition.face_encodings(
        face_rgb,
        known_face_locations=[(0, face_rgb.shape[1], face_rgb.shape[0], 0)],
        num_jitters=1,
        model="small",
    )
    if not encodings:
        return None, 0.0

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    vec = np.array(encodings[0], dtype="float32").reshape(1, -1)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        idx   = int(np.argmax(probs))
        return model.classes_[idx], float(probs[idx]) * 100.0
    return str(model.predict(vec)[0]), 100.0


# ---------------------------------------------------------------------------
# HTTP worker thread
# ---------------------------------------------------------------------------

class HttpWorker(threading.Thread):

    def __init__(self, on_success, on_failure):
        super().__init__(daemon=True)
        self._q         = queue.Queue()
        self._running   = True
        self.on_success = on_success
        self.on_failure = on_failure

    def send(self, name: str, confidence: float) -> None:
        payload = {
            "name":       name,
            "confidence": round(confidence, 2),
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "device_id":  DEVICE_ID,
        }
        self._q.put(payload)

    def run(self) -> None:
        while self._running:
            try:
                payload = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            name = payload["name"]
            try:
                resp = requests.post(SERVER_URL, json=payload,
                                     timeout=HTTP_TIMEOUT)
                if resp.status_code in (200, 201, 409):
                    self.on_success(name)
                else:
                    self.on_failure(name, f"HTTP {resp.status_code}")
            except requests.exceptions.ConnectionError:
                self.on_failure(name, "Server unreachable")
            except requests.exceptions.Timeout:
                self.on_failure(name, "Request timed out")
            except Exception as e:
                self.on_failure(name, str(e))

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# Confirmation timer
# ---------------------------------------------------------------------------

class ConfirmationTimer:

    def __init__(self):
        self._name      : str | None = None
        self._start_time: float      = 0.0

    def update(self, name: str) -> bool:
        """Returns True the moment CONFIRM_SECONDS is reached."""
        if name == "Unknown":
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
        if self._name is None:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def current_name(self) -> str | None:
        return self._name


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------

class FPS:
    def __init__(self, window: int = 30):
        self._times: list[float] = []
        self._window = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


# ---------------------------------------------------------------------------
# Box helpers
# ---------------------------------------------------------------------------

def box_centre(box: tuple) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2, y + h / 2


def box_distance(a: tuple | None, b: tuple | None) -> float:
    if a is None or b is None:
        return float("inf")
    ax, ay = box_centre(a)
    bx, by = box_centre(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.path.exists(MODEL_FILE):
        print(f"Model not found: {MODEL_FILE}")
        print("Run pipeline.bat first.")
        sys.exit(1)

    # ── Shared prediction state ──────────────────────────────────────────
    _lock          = threading.Lock()
    _result_name   = None
    _result_conf   = None
    _current_token = 0

    def on_embed_result(name: str | None, conf: float, token: int) -> None:
        nonlocal _result_name, _result_conf
        with _lock:
            if token != _current_token:
                return
            if name is not None and conf >= UNKNOWN_THRESHOLD:
                _result_name = name
                _result_conf = conf
            else:
                _result_name = "Unknown"
                _result_conf = None

    # ── Cooldown registry ────────────────────────────────────────────────
    _cooldown: dict[str, float] = {}

    def in_cooldown(name: str) -> bool:
        if name not in _cooldown:
            return False
        return (time.monotonic() - _cooldown[name]) < COOLDOWN_MINUTES * 60

    # ── HTTP callbacks ───────────────────────────────────────────────────
    _http_status = {"msg": "", "ok": True}

    def on_http_success(name: str) -> None:
        _cooldown[name]     = time.monotonic()
        _http_status["msg"] = f"Recorded: {name}"
        _http_status["ok"]  = True
        print(f"[HTTP] Attendance recorded for {name}")

    def on_http_failure(name: str, reason: str) -> None:
        _http_status["msg"] = f"POST failed: {reason}"
        _http_status["ok"]  = False
        print(f"[HTTP] Failed to record {name}: {reason}")

    # ── Workers ──────────────────────────────────────────────────────────
    executor    = ProcessPoolExecutor(max_workers=1)
    http_worker = HttpWorker(on_http_success, on_http_failure)
    http_worker.start()

    _future = None   # holds the current pending process future

    camera_idx = pick_camera()
    cap = cv2.VideoCapture(camera_idx, _CAP_BACKEND)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    tracked_box     = None
    prev_box        = None
    last_submit_box = None
    timer           = ConfirmationTimer()
    fps_counter     = FPS()

    _confirmed_name            = None
    _confirmed_at              = 0.0
    CONFIRMED_DISPLAY_SECONDS  = 3.0

    print("Live recognition running — press ESC or Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read error.")
            break

        fps = fps_counter.tick()
        now = time.monotonic()

        # ── Detect & track ───────────────────────────────────────────────
        boxes       = detect_faces(frame)
        tracked_box = select_tracked_box(boxes, tracked_box)

        # ── Reset on face disappear or jump ─────────────────────────────
        face_gone   = tracked_box is None
        face_jumped = box_distance(tracked_box, prev_box) > BOX_JUMP_THRESHOLD

        if face_gone or face_jumped:
            with _lock:
                _current_token += 1
                _result_name    = None
                _result_conf    = None
            timer.reset()
            last_submit_box = None

        prev_box = tracked_box

        # ── Submit to process worker ─────────────────────────────────────
        if tracked_box is not None:
            if (_future is None or _future.done()) and \
               box_distance(tracked_box, last_submit_box) > BOX_MOVE_THRESHOLD:

                face = crop_face(frame, tracked_box)
                with _lock:
                    tok = _current_token

                _future = executor.submit(
                    _embed_and_predict,
                    face.tobytes(), face.shape, MODEL_FILE, UNKNOWN_THRESHOLD,
                )

                def _handle(fut, t=tok):
                    try:
                        name, conf = fut.result()
                        on_embed_result(name, conf, t)
                    except Exception:
                        pass

                _future.add_done_callback(_handle)
                last_submit_box = tracked_box

        # ── Read latest prediction ───────────────────────────────────────
        with _lock:
            name = _result_name
            conf = _result_conf

        # ── Confirmation timer ───────────────────────────────────────────
        just_confirmed = False
        if name and name != "Unknown" and conf is not None and conf >= CONFIRM_THRESHOLD:
            just_confirmed = timer.update(name)
        else:
            timer.reset()

        if just_confirmed and not in_cooldown(name):
            http_worker.send(name, conf or 0.0)
            _confirmed_name = name
            _confirmed_at   = now
            timer.reset()

        # ── Draw ─────────────────────────────────────────────────────────
        if tracked_box is not None:
            recently_confirmed = (
                _confirmed_name is not None
                and _confirmed_name == name
                and (now - _confirmed_at) < CONFIRMED_DISPLAY_SECONDS
            )

            display_name = name if name is not None else "..."
            draw_face_box(
                frame, tracked_box,
                display_name, conf,
                known     = (name is not None and name != "Unknown"),
                confirmed = recently_confirmed,
            )

            # Progress bar (only when not in cooldown)
            if name and name != "Unknown" and not in_cooldown(name):
                progress  = min(timer.progress(), CONFIRM_SECONDS)
                bar_width = tracked_box[2]
                filled    = int(bar_width * progress / CONFIRM_SECONDS)
                bx        = tracked_box[0]
                by        = tracked_box[1] + tracked_box[3] + 4
                cv2.rectangle(frame, (bx, by), (bx + bar_width, by + 6),
                              (60, 60, 60), cv2.FILLED)
                cv2.rectangle(frame, (bx, by), (bx + filled, by + 6),
                              (0, 210, 0), cv2.FILLED)

            # Cooldown indicator
            if name and in_cooldown(name):
                elapsed   = time.monotonic() - _cooldown.get(name, 0)
                remaining = max(0, COOLDOWN_MINUTES * 60 - elapsed)
                hud = [
                    f"FPS: {fps:.0f}",
                    f"Cooldown: {int(remaining // 60)}m {int(remaining % 60)}s",
                    "ESC / Q to quit",
                ]
            else:
                hud = [f"FPS: {fps:.0f}", "ESC / Q to quit"]

            draw_hud(frame, hud)

            # HTTP status at bottom of screen
            if _http_status["msg"]:
                color = (0, 210, 0) if _http_status["ok"] else (0, 0, 220)
                h, w  = frame.shape[:2]
                cv2.putText(frame, _http_status["msg"],
                            (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            color, 1, cv2.LINE_AA)
        else:
            draw_hud(frame, [f"FPS: {fps:.0f}", "ESC / Q to quit"])

        cv2.imshow("Face Recognition — Attendance", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break

    http_worker.stop()
    executor.shutdown(wait=False)
    cap.release()
    cv2.destroyAllWindows()

def start_recognition():
    main()

def recognize_streamlit():
    import streamlit as st
    import cv2
    import pickle
    import numpy as np

    from face_utils import detect_faces, crop_face, draw_face_box
    from recognize_live import _embed_and_predict, MODEL_FILE, UNKNOWN_THRESHOLD

    st.title("Face Recognition")

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    # Load model once
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detect_faces(frame)

        for box in boxes:
            face = crop_face(frame, box)

            name, conf = _embed_and_predict(
                face.tobytes(),
                face.shape,
                MODEL_FILE,
                UNKNOWN_THRESHOLD
            )

            if name is None or conf < UNKNOWN_THRESHOLD:
                name = "Unknown"
                conf = None

            draw_face_box(frame, box, name, conf,
                          known=(name != "Unknown"))

        frame_placeholder.image(frame, channels="BGR")

    cap.release()

# Required on Windows for ProcessPoolExecutor
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()