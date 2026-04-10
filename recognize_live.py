"""
recognize_live.py — Live face recognition with in-app registration.

Controls:
  ESC / Q  — quit
  R        — register a new person without leaving the screen

Registration flow (press R):
  1. Type the person's name, press Enter.
  2. The person looks at the camera — face is captured automatically.
  3. Augmentation + embedding + retraining runs in a background thread.
  4. Model hot-swaps silently. Recognition continues throughout.
"""

import os
import sys
import platform
import threading
import multiprocessing
import time
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from config import (
    MODEL_FILE, UNKNOWN_THRESHOLD, EMBEDDING_MODEL,
    BOX_JUMP_THRESHOLD, BOX_MOVE_THRESHOLD, CONFIRM_SECONDS,
)
from core.detector  import detect_faces, crop_face, select_tracked_box, box_distance
from core.drawing   import (draw_face_box, draw_progress_bar, draw_hud,
                             draw_status, draw_registration_overlay)
from workers.embed_worker import EmbedWorker
from workers.http_worker  import HttpWorker
from attendance.timer     import ConfirmationTimer
from attendance.cooldown  import CooldownRegistry
from attendance.registrar import register_person


# ---------------------------------------------------------------------------
# Camera picker
# ---------------------------------------------------------------------------

def pick_camera() -> tuple[int, int]:
    backend   = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
    available = []
    for i in range(10):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
        cap.release()
    if not available:
        print("No cameras found.")
        sys.exit(1)
    print("Available cameras:", available)
    raw = input("Camera index (default 0): ").strip()
    try:
        idx = int(raw)
    except ValueError:
        idx = 0
    return (idx if idx in available else 0), backend


# ---------------------------------------------------------------------------
# Model hot-swap wrapper
# ---------------------------------------------------------------------------

class ModelRef:
    """
    Thread-safe reference to the current KNN model.
    Recognition reads via .get(), registration writes via .swap().
    """

    def __init__(self, model):
        self._model = model
        self._lock  = threading.Lock()

    def get(self):
        with self._lock:
            return self._model

    def swap(self, new_model) -> None:
        with self._lock:
            self._model = new_model
        print("[Model] Hot-swapped to new model.")


# ---------------------------------------------------------------------------
# Registration state machine
# ---------------------------------------------------------------------------

class RegistrationState:
    """
    Tracks where we are in the registration flow.

    States:
      idle        — normal recognition mode
      typing      — user is typing the name
      capturing   — waiting for a clear face to auto-capture
      processing  — background thread running augment/embed/train
      done        — success, showing confirmation briefly
      error       — something failed
    """

    IDLE       = "idle"
    TYPING     = "typing"
    CAPTURING  = "capturing"
    PROCESSING = "processing"
    DONE       = "done"
    ERROR      = "error"

    def __init__(self):
        self.state   = self.IDLE
        self.name    = ""
        self.message = ""

    def is_idle(self)       -> bool: return self.state == self.IDLE
    def is_typing(self)     -> bool: return self.state == self.TYPING
    def is_capturing(self)  -> bool: return self.state == self.CAPTURING
    def is_processing(self) -> bool: return self.state == self.PROCESSING
    def is_done(self)       -> bool: return self.state == self.DONE
    def is_error(self)      -> bool: return self.state == self.ERROR
    def is_active(self)     -> bool: return self.state != self.IDLE


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Load initial model ───────────────────────────────────────────────
    if not os.path.exists(MODEL_FILE):
        print(f"Model not found: {MODEL_FILE}")
        print("Run pipeline.bat first, or press R inside the app to register people.")
        # Start with no model — recognition will show "..." until one is trained
        model_ref = ModelRef(None)
    else:
        import pickle
        with open(MODEL_FILE, "rb") as f:
            model_ref = ModelRef(pickle.load(f))
        print(f"Model loaded.")

    # ── Shared prediction state ──────────────────────────────────────────
    _pred_lock     = threading.Lock()
    _result_name   = None
    _result_conf   = None
    _current_token = 0

    def on_embed_result(name, conf, token):
        nonlocal _result_name, _result_conf
        with _pred_lock:
            if token != _current_token:
                return
            if name is not None and conf >= UNKNOWN_THRESHOLD:
                _result_name = name
                _result_conf = conf
            else:
                _result_name = "Unknown"
                _result_conf = None

    def reset_prediction():
        nonlocal _current_token, _result_name, _result_conf
        with _pred_lock:
            _current_token += 1
            _result_name    = None
            _result_conf    = None

    # ── HTTP callbacks ───────────────────────────────────────────────────
    _http_status = {"msg": "", "ok": True}

    def on_http_success(name):
        cooldown.record(name)
        _http_status.update(msg=f"Recorded: {name}", ok=True)
        print(f"[HTTP] Recorded: {name}")

    def on_http_failure(name, reason):
        _http_status.update(msg=f"POST failed: {reason}", ok=False)
        print(f"[HTTP] Failed ({name}): {reason}")

    # ── Workers & state objects ──────────────────────────────────────────
    embed_worker = EmbedWorker()
    http_worker  = HttpWorker(on_http_success, on_http_failure)
    timer        = ConfirmationTimer()
    cooldown     = CooldownRegistry()
    reg          = RegistrationState()
    http_worker.start()

    camera_idx, backend = pick_camera()
    cap = cv2.VideoCapture(camera_idx, backend)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    tracked_box     = None
    prev_box        = None
    last_submit_box = None
    confirmed_name  = None
    confirmed_at    = 0.0
    done_at         = 0.0
    CONFIRMED_SHOW  = 3.0
    DONE_SHOW       = 2.0   # seconds to show "done" overlay before returning to idle

    print("Running — R to register, ESC or Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read error.")
            break

        now  = time.monotonic()
        key  = cv2.waitKey(1) & 0xFF

        # ================================================================
        # REGISTRATION MODE
        # ================================================================
        if not reg.is_idle():

            # ── TYPING state — collect name from keyboard ────────────────
            if reg.is_typing():
                if key == 27:                           # ESC → cancel
                    reg.state = RegistrationState.IDLE
                    reg.name  = ""
                elif key == 13:                         # Enter → confirm
                    if reg.name.strip():
                        reg.state = RegistrationState.CAPTURING
                    # else: ignore empty name
                elif key == 8:                          # Backspace
                    reg.name = reg.name[:-1]
                elif 32 <= key <= 126:                  # printable char
                    reg.name += chr(key)

                draw_registration_overlay(frame, "typing", name=reg.name)

            # ── CAPTURING state — show live preview, wait for SPACE ─────
            elif reg.is_capturing():
                boxes     = detect_faces(frame)
                face_crop = None

                if boxes:
                    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
                    bx, by, bw, bh = boxes[0]
                    face_crop = crop_face(frame, boxes[0])
                    # Green box while face is detected
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh),
                                  (0, 210, 0), 2)

                if key == 27:                           # ESC → cancel
                    reg.state = RegistrationState.IDLE
                    reg.name  = ""

                elif key in (32, ord("c")) and face_crop is not None:  # SPACE / C
                    reg.state   = RegistrationState.PROCESSING
                    reg.message = "Starting..."

                    def _run_registration(name, face, ref):
                        try:
                            def on_progress(msg):
                                reg.message = msg
                            new_model = register_person(name, face, on_progress)
                            ref.swap(new_model)
                            reg.state      = RegistrationState.DONE
                            done_at_ref[0] = time.monotonic()
                        except Exception as exc:
                            reg.state   = RegistrationState.ERROR
                            reg.message = str(exc)

                    done_at_ref = [0.0]
                    threading.Thread(
                        target=_run_registration,
                        args=(reg.name, face_crop, model_ref),
                        daemon=True,
                    ).start()

                draw_registration_overlay(
                    frame, "capturing",
                    ready=face_crop is not None,
                )

            # ── PROCESSING state — show progress message ─────────────────
            elif reg.is_processing():
                draw_registration_overlay(frame, "processing",
                                          message=reg.message)

            # ── DONE state — show success briefly then return to idle ─────
            elif reg.is_done():
                draw_registration_overlay(frame, "done", name=reg.name)
                if time.monotonic() - done_at_ref[0] > DONE_SHOW:
                    reset_prediction()
                    reg.state = RegistrationState.IDLE
                    reg.name  = ""

            # ── ERROR state — show error until any key pressed ────────────
            elif reg.is_error():
                draw_registration_overlay(frame, "error",
                                          message=reg.message)
                if key != 255:      # any key
                    reg.state   = RegistrationState.IDLE
                    reg.name    = ""
                    reg.message = ""

            cv2.imshow("Attendance", frame)
            continue   # skip normal recognition logic while registering

        # ================================================================
        # NORMAL RECOGNITION MODE
        # ================================================================

        # ── Detect & track ───────────────────────────────────────────────
        boxes       = detect_faces(frame)
        tracked_box = select_tracked_box(boxes, tracked_box)

        if tracked_box is None or box_distance(tracked_box, prev_box) > BOX_JUMP_THRESHOLD:
            reset_prediction()
            timer.reset()
            last_submit_box = None

        prev_box = tracked_box

        # ── Submit to embed worker ───────────────────────────────────────
        current_model = model_ref.get()
        if (current_model is not None
                and tracked_box is not None
                and box_distance(tracked_box, last_submit_box) > BOX_MOVE_THRESHOLD):
            with _pred_lock:
                tok = _current_token
            embed_worker.submit(
                crop_face(frame, tracked_box), tok, on_embed_result,
                MODEL_FILE, UNKNOWN_THRESHOLD, EMBEDDING_MODEL,
            )
            last_submit_box = tracked_box

        # ── Read latest prediction ───────────────────────────────────────
        with _pred_lock:
            name = _result_name
            conf = _result_conf

        # ── Confirmation timer ───────────────────────────────────────────
        if name and name != "Unknown" and conf is not None:
            if timer.update(name, conf) and not cooldown.in_cooldown(name):
                http_worker.send(name, conf)
                confirmed_name = name
                confirmed_at   = now
                timer.reset()
        else:
            timer.reset()

        # ── Draw ─────────────────────────────────────────────────────────
        if tracked_box is not None:
            recently_confirmed = (
                confirmed_name == name
                and (now - confirmed_at) < CONFIRMED_SHOW
            )
            display = name if name is not None else "..."
            draw_face_box(frame, tracked_box, display, conf,
                          known     = (name is not None and name != "Unknown"),
                          confirmed = recently_confirmed)

            if name and name != "Unknown" and not cooldown.in_cooldown(name):
                draw_progress_bar(frame, tracked_box,
                                  timer.progress(), CONFIRM_SECONDS)

            remaining = cooldown.remaining_seconds(name) if name else 0
            hud = ["R = register   ESC / Q = quit"]
            if remaining > 0:
                hud.append(f"Cooldown: {int(remaining // 60)}m {int(remaining % 60)}s")
            draw_hud(frame, hud)
        else:
            draw_hud(frame, ["R = register   ESC / Q = quit"])

        draw_status(frame, _http_status["msg"], _http_status["ok"])
        cv2.imshow("Attendance", frame)

        # ── Key handling ─────────────────────────────────────────────────
        if key in (27, ord("q"), ord("Q")):
            break
        if key == ord("r") or key == ord("R"):
            reg.state = RegistrationState.TYPING
            reg.name  = ""

    embed_worker.shutdown()
    http_worker.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
