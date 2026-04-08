import os
import cv2
from pathlib import Path
from collections import defaultdict

from face_utils import detect_faces, crop_face

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_DIR = "bulk_input"
OUTPUT_DIR = os.path.join("data", "raw")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_name(filename: str) -> str:
    """
    Extract person name from filename.
    Example: 'juan_1.jpg' → 'juan'
    """
    return Path(filename).stem.split("_")[0].lower()


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"Input folder not found: {INPUT_DIR}")
        return

    counters = defaultdict(int)

    files = [f for f in os.listdir(INPUT_DIR)
             if Path(f).suffix.lower() in IMAGE_EXTS]

    if not files:
        print("No images found.")
        return

    total_saved = 0
    total_skipped = 0

    for fname in files:
        path = os.path.join(INPUT_DIR, fname)
        name = extract_name(fname)

        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Cannot read: {fname}")
            continue

        boxes = detect_faces(img)

        if not boxes:
            print(f"[SKIP] No face: {fname}")
            total_skipped += 1
            continue

        # Take largest face (same as your system)
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = boxes[0]

        # Filter small faces (important)
        if w < 80 or h < 80:
            print(f"[SKIP] Face too small: {fname}")
            total_skipped += 1
            continue

        face = crop_face(img, boxes[0])

        # Save
        person_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        counters[name] += 1
        out_name = f"{name}_auto_{counters[name]:04d}.jpg"
        out_path = os.path.join(person_dir, out_name)

        cv2.imwrite(out_path, face)
        total_saved += 1

        print(f"[OK] {fname} → {out_path}")

    print("\n── Summary ─────────────────────────────")
    print(f"Saved  : {total_saved}")
    print(f"Skipped: {total_skipped}")
    print("────────────────────────────────────────")

    print("\nNext steps:")
    print("  python augment_dataset.py")
    print("  python build_dataset.py")
    print("  python train_model.py")


if __name__ == "__main__":
    main()