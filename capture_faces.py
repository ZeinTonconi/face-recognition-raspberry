"""
capture_faces.py — Guided 5-angle face registration via webcam.

Usage:
    python capture_faces.py

Guides you through CENTER → LEFT → RIGHT → UP → DOWN.
Press SPACE to capture each pose. Press ESC to abort.

Saved to data/raw/<n>/<n>_<pose>.jpg
After this run pipeline.bat to augment, embed and train.
"""

import os
import sys
import platform
import cv2

from config import RAW_DIR, FACE_SIZE
from core.detector import detect_faces, crop_face, select_tracked_box, box_distance
from core.drawing  import draw_face_box, draw_hud

POSES = ["center", "left", "right", "up", "down"]

POSE_INSTRUCTIONS = {
    "center": ["Look straight at the camera.", "Press SPACE to capture."],
    "left":   ["Turn your head slightly LEFT.", "Press SPACE to capture."],
    "right":  ["Turn your head slightly RIGHT.", "Press SPACE to capture."],
    "up":     ["Tilt your head slightly UP.", "Press SPACE to capture."],
    "down":   ["Tilt your head slightly DOWN.", "Press SPACE to capture."],
}


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


def capture_pose(cap, pose: str, save_path: str) -> bool:
    instructions = POSE_INSTRUCTIONS[pose]
    tracked_box  = None
    prev_box     = None

    while True:
        ret, frame = cap.read()
        if not ret:
            return False

        boxes       = detect_faces(frame)
        tracked_box = select_tracked_box(boxes, tracked_box)

        if tracked_box is not None:
            draw_face_box(frame, tracked_box, pose.upper())

        idx = POSES.index(pose)
        draw_hud(frame, [f"Pose {idx+1}/{len(POSES)}: {pose.upper()}"] + instructions
                 + (["No face detected — reposition."] if tracked_box is None else []))

        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            return False
        if key in (32, ord("c")) and tracked_box is not None:
            face = crop_face(frame, tracked_box)
            cv2.imwrite(save_path, face)
            print(f"  [{pose}] saved → {save_path}")
            return True


def main() -> None:
    name = input("Enter the person's name (no spaces): ").strip()
    if not name:
        print("Name cannot be empty.")
        sys.exit(1)

    person_dir = os.path.join(RAW_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    camera_idx, backend = pick_camera()
    cap = cv2.VideoCapture(camera_idx, backend)

    print(f"\nRegistering: {name}")
    print("Press SPACE to capture each pose. ESC to abort.\n")

    captured = []
    for pose in POSES:
        save_path = os.path.join(person_dir, f"{name}_{pose}.jpg")
        print(f"Pose: {pose.upper()}")
        if not capture_pose(cap, pose, save_path):
            print("Aborted.")
            break
        captured.append(pose)

        ret, flash = cap.read()
        if ret:
            draw_hud(flash, [f"Captured: {pose.upper()} ✓"])
            cv2.imshow("Capture", flash)
            cv2.waitKey(400)

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nDone. {len(captured)}/{len(POSES)} poses captured.")
    print(f"Images saved to: {person_dir}/")
    print("\nNext step:  pipeline.bat")


if __name__ == "__main__":
    main()
