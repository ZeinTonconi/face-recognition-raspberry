"""
capture_faces.py — Guided 5-angle face registration via webcam.

Usage:
    python capture_faces.py

The script walks you through five poses one at a time:
    CENTER  →  LEFT  →  RIGHT  →  UP  →  DOWN

For each pose:
  1. A prompt on screen tells you where to look.
  2. As soon as a face is detected, a live preview with bounding box appears.
  3. Press SPACE (or 'c') to capture that pose.  Press ESC to abort.

Captured crops are saved as JPEG images under:
    data/raw/<name>/<name>_<pose>.jpg

After all five poses are done (or if you also uploaded photos into the
same folder), run build_dataset.py to compute embeddings, then
train_model.py to retrain the classifier.
"""

import os
import sys
import cv2

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
RAW_DIR = os.path.join("data", "raw")

POSES = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]

POSE_INSTRUCTIONS: dict[str, list[str]] = {
    "CENTER": ["Look straight at the camera.", "Press SPACE to capture."],
    "LEFT":   ["Turn your head slightly LEFT.", "Press SPACE to capture."],
    "RIGHT":  ["Turn your head slightly RIGHT.", "Press SPACE to capture."],
    "UP":     ["Tilt your head slightly UP.", "Press SPACE to capture."],
    "DOWN":   ["Tilt your head slightly DOWN.", "Press SPACE to capture."],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def capture_pose(
    cap: cv2.VideoCapture,
    pose: str,
    save_path: str,
) -> bool:
    """
    Show a live preview and wait for the user to press SPACE to capture.

    Returns True on successful capture, False if the user pressed ESC.
    """
    instructions = POSE_INSTRUCTIONS[pose]
    tracked_box  = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read error.")
            return False

        boxes       = detect_faces(frame)
        tracked_box = select_tracked_box(boxes, tracked_box)

        if tracked_box is not None:
            draw_face_box(frame, tracked_box, pose, known=True)

        hud_lines = [f"Pose {POSES.index(pose)+1}/{len(POSES)}: {pose}"] + instructions
        if tracked_box is None:
            hud_lines.append("No face detected — reposition yourself.")
        draw_hud(frame, hud_lines)

        cv2.imshow("Capture — face registration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:                       # ESC → abort
            return False
        if key in (32, ord("c")) and tracked_box is not None:   # SPACE or 'c'
            crop = crop_face(frame, tracked_box)
            cv2.imwrite(save_path, crop)
            print(f"  [{pose}] saved → {save_path}")
            return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(name) -> None:
    
    person_dir = os.path.join(RAW_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    camera_idx = pick_camera()
    cap        = cv2.VideoCapture(camera_idx)

    print(f"\nRegistering: {name}")
    print("You will be guided through 5 poses.  Press SPACE to capture each one.")
    print("Press ESC at any time to abort.\n")

    captured = []
    for pose in POSES:
        filename  = f"{name}_{pose.lower()}.jpg"
        save_path = os.path.join(person_dir, filename)

        print(f"Pose: {pose}")
        ok = capture_pose(cap, pose, save_path)

        if not ok:
            print("Aborted.")
            break

        captured.append(pose)
        # Brief green flash to confirm capture
        ret, flash = cap.read()
        if ret:
            overlay = flash.copy()
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1], overlay.shape[0]),
                          (0, 200, 0), cv2.FILLED)
            cv2.addWeighted(overlay, 0.25, flash, 0.75, 0, flash)
            draw_hud(flash, [f"Captured: {pose}  ✓"])
            cv2.imshow("Capture — face registration", flash)
            cv2.waitKey(400)

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nDone. Captured {len(captured)}/{len(POSES)} poses: {captured}")
    print(f"Images saved to:  {person_dir}/")
    print("\nNext steps:")
    print("  1. (Optional) Copy any extra photos into the same folder.")
    print("  2. Run  build_dataset.py  to compute embeddings.")
    print("  3. Run  train_model.py    to retrain the classifier.")

def capture_person(name: str):
    if not name:
        raise ValueError("Name cannot be empty")
    main(name)

def capture_streamlit(name: str):
    import streamlit as st
    import cv2
    import os

    from face_utils import detect_faces, crop_face, draw_face_box

    person_dir = os.path.join("data", "raw", name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    captured = 0

    while captured < 5:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detect_faces(frame)

        if boxes:
            draw_face_box(frame, boxes[0], f"{captured+1}/5")

        frame_placeholder.image(frame, channels="BGR")

        # ⚠️ Streamlit buttons are tricky inside loops
        if st.button("Capture"):
            if boxes:
                face = crop_face(frame, boxes[0])
                path = os.path.join(person_dir, f"{name}_{captured}.jpg")
                cv2.imwrite(path, face)
                captured += 1

    cap.release()

if __name__ == "__main__":
    name = input("Enter the person's name (no spaces): ").strip()
    if not name:
        print("Name cannot be empty.")
        sys.exit(1)

    capture_person(name)