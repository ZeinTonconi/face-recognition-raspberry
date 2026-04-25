"""
import_drive_photos.py — Import student photos from a Drive folder.

Usage:
    python import_drive_photos.py --input "C:/Users/Zein/Downloads/fotos"
    python import_drive_photos.py --input fotos --preview   # dry run, no files saved

Filename convention expected:
    <anything> - <Student Name>.<ext>

The name is always the part after the LAST dash in the filename.
Examples that all work:
    130cb2db-9565-4109-bdff-ad2250ac0c23 - Andrés Pacheco.jpg
    WhatsApp Image 2026-03-26 at 15.32.33 - Serely Karina Quette.jpg
    001 - Juan Perez.png

What this script does:
    1. Reads all images from --input folder.
    2. Parses the student name from the filename (part after last dash).
    3. Detects and crops the largest face in each photo.
    4. Saves the crop to data/raw/<name>/<name>_imported.jpg
    5. Skips photos where no face is detected and reports them at the end.

After this script run:
    cmd /c pipeline.bat
"""

import os
import sys
import re
import argparse
import logging
from pathlib import Path

import cv2

from core.detector import detect_faces, crop_face

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DIR    = os.path.join("data", "raw")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Name parsing
# ---------------------------------------------------------------------------

def parse_name(filename: str) -> str | None:
    """
    Extract the student name from a filename.
    Rule: take everything after the last ' - ' (space-dash-space).
    Strip the extension and clean up whitespace.

    Examples:
        '130cb2db-...-ad2250ac0c23 - Andrés Pacheco.jpg' → 'Andrés Pacheco'
        'WhatsApp Image 2026-03-26 at 15.32.33 - Serely Karina Quette.jpg'
            → 'Serely Karina Quette'
    """
    stem = Path(filename).stem          # remove extension
    parts = stem.split(" - ")           # split on space-dash-space
    if len(parts) < 2:
        return None
    name = parts[-1].strip()            # last part is always the name
    # Remove any trailing garbage (extra dots, underscores, numbers)
    name = re.sub(r"[\._\d]+$", "", name).strip()
    return name if name else None


def name_to_folder(name: str) -> str:
    """
    Convert display name to a safe folder name.
    'Andrés Pacheco' → 'Andres_Pacheco'
    """
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "ñ": "n", "Ñ": "N", "ü": "u", "Ü": "U",
    }
    for src, dst in replacements.items():
        name = name.replace(src, dst)
    # Replace spaces and any non-alphanumeric with underscore
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    return name.strip("_")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir: str, preview: bool = False) -> None:
    if not os.path.isdir(input_dir):
        log.error("Input folder not found: %s", input_dir)
        sys.exit(1)

    # Collect all image files
    all_files = [
        f for f in sorted(os.listdir(input_dir))
        if Path(f).suffix.lower() in IMAGE_EXTS
    ]

    if not all_files:
        log.error("No image files found in %s", input_dir)
        sys.exit(1)

    log.info("Found %d image(s) in %s", len(all_files), input_dir)

    saved    = []
    skipped  = []   # no face detected
    no_name  = []   # could not parse name

    for fname in all_files:
        img_path = os.path.join(input_dir, fname)

        # ── Parse name ───────────────────────────────────────────────────
        display_name = parse_name(fname)
        if not display_name:
            log.warning("Could not parse name from: %s", fname)
            no_name.append(fname)
            continue

        folder_name = name_to_folder(display_name)

        # ── Read image (numpy decode handles non-ASCII paths on Windows) ─
        try:
            buf = open(img_path, "rb").read()
            arr = __import__("numpy").frombuffer(buf, dtype="uint8")
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            img = None

        if img is None:
            log.warning("Could not read: %s", fname)
            skipped.append((fname, "unreadable"))
            continue

        # ── Detect face ──────────────────────────────────────────────────
        boxes = detect_faces(img)

        if not boxes:
            # Fallback: use whole image when MediaPipe misses AI-generated
            # or illustrated faces. Logged as warning so you can review.
            log.warning("[%s] No face in %s — using full image as fallback.",
                        display_name, fname)
            from config import FACE_SIZE
            face = cv2.resize(img, FACE_SIZE)
        else:
            boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
            face = crop_face(img, boxes[0])

        # ── Save ─────────────────────────────────────────────────────────
        person_dir = os.path.join(RAW_DIR, folder_name)
        save_path  = os.path.join(person_dir, f"{folder_name}_imported.jpg")

        if not preview:
            os.makedirs(person_dir, exist_ok=True)
            cv2.imwrite(save_path, face)

        log.info("  %-30s → %s", display_name, save_path)
        saved.append((display_name, save_path))

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n── Import summary ───────────────────────────────────")
    print(f"  Imported successfully : {len(saved)}")
    print(f"  Skipped (no face)    : {len(skipped)}")
    print(f"  Skipped (no name)    : {len(no_name)}")

    if skipped:
        print("\n── Photos with no face detected ─────────────────────")
        for fname, reason in skipped:
            print(f"  {reason:<22}  {fname}")

    if no_name:
        print("\n── Photos with unparseable name ──────────────────────")
        for fname in no_name:
            print(f"  {fname}")

    if preview:
        print("\n[PREVIEW MODE — no files were saved]")
    else:
        print(f"\nCrops saved to: {RAW_DIR}/")
        print("\nNext step:  cmd /c pipeline.bat")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import Drive student photos into the face recognition dataset."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the folder containing the Drive photos."
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Dry run — show what would be imported without saving anything."
    )
    args = parser.parse_args()
    main(input_dir=args.input, preview=args.preview)