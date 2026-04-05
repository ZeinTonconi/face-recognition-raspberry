"""
augment_dataset.py — Expand the raw face photos using image augmentation.

Usage:
    python augment_dataset.py                  # augment everyone in data/raw/
    python augment_dataset.py --name Zein      # augment one person only
    python augment_dataset.py --preview Zein   # show augmentations without saving

For each original photo this script generates a set of augmented variants:
rotations, flips, brightness/contrast shifts, blur, noise, saturation shifts,
perspective warps, and combinations of the above.

Starting from 5 photos you can realistically get 80-120 augmented images per
person, which gives the KNN classifier much more to work with.

Augmented images are saved alongside the originals in data/raw/<name>/ so
build_dataset.py picks them up automatically on the next run.

Run order:
    capture_faces.py  →  augment_dataset.py  →  build_dataset.py  →  train_model.py
"""

import os
import argparse
import logging
import random
import itertools
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DIR    = os.path.join("data", "raw")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SEED       = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)
random.seed(SEED)
np.random.seed(SEED)


# ===========================================================================
# Individual augmentation functions
# Each takes a BGR image and returns a BGR image of the same size.
# ===========================================================================

def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate by *angle* degrees around the image centre."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_REFLECT_101)


def flip_h(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)


def brightness(img: np.ndarray, delta: int) -> np.ndarray:
    """Shift brightness by *delta* (-100 … +100)."""
    return cv2.convertScaleAbs(img, alpha=1.0, beta=delta)


def contrast(img: np.ndarray, alpha: float) -> np.ndarray:
    """Scale contrast by *alpha* (0.6 = lower, 1.4 = higher)."""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


def saturation(img: np.ndarray, scale: float) -> np.ndarray:
    """Multiply the S channel in HSV by *scale* (0 = greyscale, 2 = vivid)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
    return cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)


def hue_shift(img: np.ndarray, delta: int) -> np.ndarray:
    """Shift hue by *delta* degrees (-30 … +30)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("int32")
    hsv[:, :, 0] = (hsv[:, :, 0] + delta) % 180
    return cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)


def gaussian_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Mild Gaussian blur to simulate slight defocus."""
    k = ksize if ksize % 2 == 1 else ksize + 1   # must be odd
    return cv2.GaussianBlur(img, (k, k), 0)


def gaussian_noise(img: np.ndarray, std: float = 10.0) -> np.ndarray:
    """Add random Gaussian noise."""
    noise = np.random.normal(0, std, img.shape).astype("float32")
    noisy = np.clip(img.astype("float32") + noise, 0, 255)
    return noisy.astype("uint8")


def perspective_warp(img: np.ndarray, strength: float = 0.06) -> np.ndarray:
    """
    Apply a slight random perspective warp to simulate a different camera angle.
    *strength* controls how far the corners are shifted (fraction of image size).
    """
    h, w = img.shape[:2]
    d = strength
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    rng = np.random.uniform(-d, d, (4, 2)) * np.array([w, h])
    dst = np.clip(src + rng, 0, [w, h]).astype("float32")
    M   = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h),
                               borderMode=cv2.BORDER_REFLECT_101)


def gamma(img: np.ndarray, g: float) -> np.ndarray:
    """Apply gamma correction — g < 1 brightens, g > 1 darkens."""
    inv  = 1.0 / g
    lut  = np.array([((i / 255.0) ** inv) * 255
                     for i in range(256)], dtype="uint8")
    return cv2.LUT(img, lut)


def sharpen(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype="float32")
    return cv2.filter2D(img, -1, kernel)


def grayscale_rgb(img: np.ndarray) -> np.ndarray:
    """Convert to greyscale but keep 3 channels (simulates B&W camera)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ===========================================================================
# Augmentation plan
# ===========================================================================
# Each entry is (suffix, function).
# "Combo" entries apply two transforms in sequence.

def build_augmentation_plan() -> list[tuple[str, callable]]:
    plan = [
        # ── Geometry ──────────────────────────────────────────────────────
        # ("rot_p5",   lambda i: rotate(i,  5)),
        # ("rot_p10",  lambda i: rotate(i, 10)),
        # ("rot_p15",  lambda i: rotate(i, 15)),
        # ("rot_m5",   lambda i: rotate(i,  -5)),
        # ("rot_m10",  lambda i: rotate(i, -10)),
        # ("rot_m15",  lambda i: rotate(i, -15)),
        # ("flip",     flip_h),
        # ("persp_a",  lambda i: perspective_warp(i, 0.04)),
        # ("persp_b",  lambda i: perspective_warp(i, 0.07)),

        # ── Brightness / contrast ─────────────────────────────────────────
        # ("bright_p30",  lambda i: brightness(i,  30)),
        ("bright_p60",  lambda i: brightness(i,  60)),
        # ("bright_m30",  lambda i: brightness(i, -30)),
        ("bright_m60",  lambda i: brightness(i, -60)),
        ("contrast_lo", lambda i: contrast(i, 0.7)),
        ("contrast_hi", lambda i: contrast(i, 1.4)),
        # ("gamma_lo",    lambda i: gamma(i, 0.6)),
        ("gamma_hi",    lambda i: gamma(i, 1.6)),

        # ── Colour ────────────────────────────────────────────────────────
        # ("sat_lo",   lambda i: saturation(i, 0.3)),
        # ("sat_hi",   lambda i: saturation(i, 1.8)),
        # ("hue_p15",  lambda i: hue_shift(i,  15)),
        # ("hue_m15",  lambda i: hue_shift(i, -15)),
        # ("gray",     grayscale_rgb),

        # ── Blur / noise / sharpness ──────────────────────────────────────
        # ("blur3",    lambda i: gaussian_blur(i, 3)),
        # ("blur5",    lambda i: gaussian_blur(i, 5)),
        # ("noise_lo", lambda i: gaussian_noise(i, 8)),
        # ("noise_hi", lambda i: gaussian_noise(i, 18)),
        # ("sharp",    sharpen),

        # ── Combos (geometry + lighting) ──────────────────────────────────
        # ("flip_bright",    lambda i: brightness(flip_h(i),  30)),
        # ("flip_dark",      lambda i: brightness(flip_h(i), -30)),
        # ("rot10_bright",   lambda i: brightness(rotate(i, 10),  25)),
        # ("rot_m10_dark",   lambda i: brightness(rotate(i,-10), -25)),
        # ("persp_noise",    lambda i: gaussian_noise(perspective_warp(i, 0.05), 10)),
        # ("flip_sat",       lambda i: saturation(flip_h(i), 1.6)),
        # ("rot10_blur",     lambda i: gaussian_blur(rotate(i, 10), 3)),
        # ("bright_sharp",   lambda i: sharpen(brightness(i, 20))),
        # ("gray_blur",      lambda i: gaussian_blur(grayscale_rgb(i), 3)),
        # ("gamma_lo_noise", lambda i: gaussian_noise(gamma(i, 0.5), 8)),
    ]
    return plan


# ===========================================================================
# Per-person augmentation
# ===========================================================================

def is_original(filename: str) -> bool:
    """
    Return True if this file is an original capture (not already augmented).
    Augmented files have a known suffix pattern like _rot_p10, _flip, etc.
    """
    stem = Path(filename).stem
    # Augmented files contain a double underscore separator before the suffix
    return "__aug__" not in stem


def augment_person(person_dir: str, dry_run: bool = False) -> int:
    """
    Augment all original images in *person_dir*.

    Returns the number of new images written.
    """
    person_dir = Path(person_dir)
    originals  = [
        f for f in person_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTS and is_original(f.name)
    ]

    if not originals:
        log.warning("No original images found in %s", person_dir)
        return 0

    plan    = build_augmentation_plan()
    written = 0

    for src_path in sorted(originals):
        img = cv2.imread(str(src_path))
        if img is None:
            log.warning("Could not read %s — skipping.", src_path.name)
            continue

        for suffix, fn in plan:
            out_name = f"{src_path.stem}__aug__{suffix}{src_path.suffix}"
            out_path = person_dir / out_name

            if out_path.exists():
                continue   # already generated — skip (incremental)

            try:
                aug = fn(img)
            except Exception as exc:
                log.warning("Augmentation '%s' failed on %s: %s",
                            suffix, src_path.name, exc)
                continue

            if not dry_run:
                cv2.imwrite(str(out_path), aug)
            written += 1

    return written


# ===========================================================================
# Preview (show augmentations in a grid window)
# ===========================================================================

def preview_person(person_dir: str) -> None:
    """
    Show all augmentations for the first original image in a scrollable grid.
    Press any key to advance, ESC to quit.
    """
    person_dir = Path(person_dir)
    originals  = [
        f for f in sorted(person_dir.iterdir())
        if f.suffix.lower() in IMAGE_EXTS and is_original(f.name)
    ]
    if not originals:
        log.error("No original images in %s", person_dir)
        return

    src_path = originals[0]
    img      = cv2.imread(str(src_path))
    plan     = build_augmentation_plan()

    THUMB = 160   # thumbnail size for the grid
    COLS  = 6

    thumbs = [cv2.resize(img, (THUMB, THUMB))]   # original first
    labels = ["original"]

    for suffix, fn in plan:
        try:
            aug = fn(img)
            thumbs.append(cv2.resize(aug, (THUMB, THUMB)))
            labels.append(suffix)
        except Exception:
            pass

    # Pad to fill the last row
    while len(thumbs) % COLS != 0:
        thumbs.append(np.zeros((THUMB, THUMB, 3), dtype="uint8"))
        labels.append("")

    rows = []
    for r in range(len(thumbs) // COLS):
        row_imgs = []
        for c in range(COLS):
            idx   = r * COLS + c
            thumb = thumbs[idx].copy()
            label = labels[idx]
            cv2.putText(thumb, label, (2, THUMB - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 180), 1)
            row_imgs.append(thumb)
        rows.append(np.hstack(row_imgs))

    grid = np.vstack(rows)
    cv2.imshow(f"Augmentation preview — {src_path.name}", grid)
    log.info("Showing %d augmentations.  Press any key to close.", len(plan))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment raw face photos to expand the training dataset."
    )
    parser.add_argument("--name",    type=str, default=None,
                        help="Augment only this person (folder name in data/raw/).")
    parser.add_argument("--preview", type=str, default=None, metavar="NAME",
                        help="Show augmentation grid for NAME without saving.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count augmentations without writing files.")
    args = parser.parse_args()

    # ── Preview mode ──────────────────────────────────────────────────────
    if args.preview:
        person_dir = os.path.join(RAW_DIR, args.preview)
        if not os.path.isdir(person_dir):
            log.error("Folder not found: %s", person_dir)
            return
        preview_person(person_dir)
        return

    # ── Augment mode ──────────────────────────────────────────────────────
    if not os.path.isdir(RAW_DIR):
        log.error("Raw data folder not found: %s", RAW_DIR)
        log.error("Run capture_faces.py first.")
        return

    if args.name:
        people = [args.name]
    else:
        people = [
            d for d in os.listdir(RAW_DIR)
            if os.path.isdir(os.path.join(RAW_DIR, d))
        ]

    if not people:
        log.error("No person folders found in %s", RAW_DIR)
        return

    total_written = 0
    for name in sorted(people):
        person_dir = os.path.join(RAW_DIR, name)
        if not os.path.isdir(person_dir):
            log.warning("Folder not found for '%s' — skipping.", name)
            continue

        originals = [
            f for f in os.listdir(person_dir)
            if Path(f).suffix.lower() in IMAGE_EXTS and is_original(f)
        ]
        log.info("%-20s  %d original(s)", name, len(originals))

        written = augment_person(person_dir, dry_run=args.dry_run)
        total_written += written

        plan_size = len(build_augmentation_plan())
        total_possible = len(originals) * plan_size
        log.info("  → %d new augmentation(s) written  "
                 "(total per person: %d original + %d augmented = %d)",
                 written,
                 len(originals),
                 total_possible,
                 len(originals) + total_possible)

    action = "Would write" if args.dry_run else "Wrote"
    print(f"\n{action} {total_written} augmented image(s) across {len(people)} person(s).")
    if not args.dry_run and total_written > 0:
        print("\nNext steps:")
        print("  python build_dataset.py   # recompute embeddings (incremental)")
        print("  python train_model.py     # retrain the classifier")


if __name__ == "__main__":
    main()
