"""
core/drawing.py — OpenCV drawing helpers for the live recognition window.
"""

import cv2
import numpy as np

_GREEN  = (0,   210,   0)
_YELLOW = (0,   210, 255)
_RED    = (0,     0, 220)
_BLUE   = (255,  140,   0)
_WHITE  = (255, 255, 255)
_BLACK  = (0,     0,   0)
_FONT   = cv2.FONT_HERSHEY_SIMPLEX


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
    Draw bounding box + name label.
    Blue = just confirmed, green = recognised, red = unknown.
    """
    x, y, w, h = box
    color = _BLUE if confirmed else (_GREEN if known else _RED)
    text  = f"{label}  {confidence:.1f}%" if confidence is not None else label
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    _text_with_bg(frame, text, (x, max(y - 8, 14)), fg=_YELLOW, bg=_BLACK)


def draw_progress_bar(
    frame:    np.ndarray,
    box:      tuple[int, int, int, int],
    progress: float,
    maximum:  float,
) -> None:
    """Draw a green fill bar just below the bounding box."""
    bx        = box[0]
    by        = box[1] + box[3] + 4
    bar_width = box[2]
    filled    = int(bar_width * min(progress, maximum) / maximum)
    cv2.rectangle(frame, (bx, by), (bx + bar_width, by + 6),
                  (60, 60, 60), cv2.FILLED)
    cv2.rectangle(frame, (bx, by), (bx + filled, by + 6),
                  _GREEN, cv2.FILLED)


def draw_hud(frame: np.ndarray, lines: list[str]) -> None:
    """Render status lines in the top-left corner."""
    for i, line in enumerate(lines):
        _text_with_bg(frame, line, (10, 24 + i * 26),
                      scale=0.6, fg=_WHITE, bg=(30, 30, 30))


def draw_status(frame: np.ndarray, message: str, ok: bool = True) -> None:
    """Render a status message at the bottom of the frame."""
    if not message:
        return
    h = frame.shape[0]
    color = _GREEN if ok else _RED
    cv2.putText(frame, message, (10, h - 15),
                _FONT, 0.55, color, 1, cv2.LINE_AA)


def draw_registration_overlay(
    frame:   np.ndarray,
    state:   str,
    name:    str = "",
    message: str = "",
    **kwargs,
) -> None:
    h, w = frame.shape[:2]

    # Small panel in bottom-left corner instead of centre
    px, py = 10, h - 160
    pw, ph = 340, 145

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph),
                  (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    tx = px + 12   # text left margin inside panel
    
    if state == "typing":
        _text_with_bg(frame, "Register new person",
                      (tx, py + 25), scale=0.6,
                      fg=_WHITE, bg=(20, 20, 20))
        _text_with_bg(frame, f"Name: {name}_",
                      (tx, py + 60), scale=0.65,
                      fg=_YELLOW, bg=(20, 20, 20))
        _text_with_bg(frame, "Enter to confirm   ESC to cancel",
                      (tx, py + 95), scale=0.45,
                      fg=(180, 180, 180), bg=(20, 20, 20))

    elif state == "capturing":
        if kwargs.get("ready", False):
            _text_with_bg(frame, "Face detected",
                          (tx, py + 30), scale=0.6,
                          fg=_GREEN, bg=(20, 20, 20))
            _text_with_bg(frame, "Press SPACE to capture",
                          (tx, py + 65), scale=0.55,
                          fg=_GREEN, bg=(20, 20, 20))
            _text_with_bg(frame, "ESC to cancel",
                          (tx, py + 100), scale=0.45,
                          fg=(180, 180, 180), bg=(20, 20, 20))
        else:
            _text_with_bg(frame, "Look at the camera",
                          (tx, py + 40), scale=0.6,
                          fg=_YELLOW, bg=(20, 20, 20))
            _text_with_bg(frame, "Waiting for face...",
                          (tx, py + 80), scale=0.5,
                          fg=(180, 180, 180), bg=(20, 20, 20))

    elif state == "processing":
        _text_with_bg(frame, "Processing...",
                      (tx, py + 30), scale=0.6,
                      fg=_YELLOW, bg=(20, 20, 20))
        _text_with_bg(frame, message or "",
                      (tx, py + 70), scale=0.5,
                      fg=_WHITE, bg=(20, 20, 20))

    elif state == "done":
        _text_with_bg(frame, f"{name} registered!",
                      (tx, py + 40), scale=0.65,
                      fg=_GREEN, bg=(20, 20, 20))
        _text_with_bg(frame, "Resuming recognition...",
                      (tx, py + 80), scale=0.5,
                      fg=_WHITE, bg=(20, 20, 20))

    elif state == "error":
        _text_with_bg(frame, "Registration failed",
                      (tx, py + 30), scale=0.6,
                      fg=_RED, bg=(20, 20, 20))
        _text_with_bg(frame, (message or "")[:40],
                      (tx, py + 65), scale=0.45,
                      fg=(180, 180, 180), bg=(20, 20, 20))
        _text_with_bg(frame, "Press any key to continue",
                      (tx, py + 100), scale=0.45,
                      fg=_WHITE, bg=(20, 20, 20))   