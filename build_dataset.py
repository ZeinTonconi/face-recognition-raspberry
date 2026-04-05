"""
build_dataset.py — Build (or rebuild) the embedding cache from raw face images.

Usage:
    python build_dataset.py              # process everything in data/raw/
    python build_dataset.py --force      # recompute even cached embeddings

Folder layout expected:
    data/
      raw/
        alice/          ← one sub-folder per person, named exactly as you want
          alice_center.jpg    ← any .jpg / .jpeg / .png files
          alice_left.jpg
          ... (from capture_faces.py or dropped in manually)
        bob/
          bob_center.jpg
          extra_photo.png     ← uploaded manually — treated identically

What this script does:
  1. Walks data/raw/ and finds every image file.
  2. Runs face detection + DeepFace Facenet512 embedding on each image.
  3. Appends the (embedding, label) pair to a single pickle file at
     data/embeddings.pkl.

Incremental mode (default):
  Already-processed image paths are stored inside embeddings.pkl.
  Re-running the script only processes NEW images — useful when you add
  one person without wanting to reprocess everyone.

After this script, run train_model.py.
"""

import os
import sys
import pickle
import argparse
import logging
import numpy as np
import cv2

from face_utils import detect_faces, crop_face, get_embedding

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DIR        = os.path.join("data", "raw")
EMBEDDINGS_PKL = os.path.join("data", "embeddings.pkl")
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_cache() -> dict:
    """
    Load existing embeddings cache from disk.

    Returns a dict with keys:
        "embeddings"  : np.ndarray  shape (N, 512), float32
        "labels"      : list[str]   length N
        "processed"   : set[str]    absolute image paths already embedded
    """
    if os.path.exists(EMBEDDINGS_PKL):
        with open(EMBEDDINGS_PKL, "rb") as f:
            data = pickle.load(f)
        # Back-compat: older files may not have "processed"
        if "processed" not in data:
            data["processed"] = set()
        log.info("Loaded cache: %d embeddings, %d unique people",
                 len(data["labels"]),
                 len(set(data["labels"])))
        return data

    return {
        "embeddings": np.empty((0, 512), dtype="float32"),
        "labels":     [],
        "processed":  set(),
    }


def save_cache(data: dict) -> None:
    os.makedirs(os.path.dirname(EMBEDDINGS_PKL), exist_ok=True)
    with open(EMBEDDINGS_PKL, "wb") as f:
        pickle.dump(data, f)
    log.info("Cache saved → %s  (%d embeddings total)",
             EMBEDDINGS_PKL, len(data["labels"]))


# ---------------------------------------------------------------------------
# Image scanning
# ---------------------------------------------------------------------------

def find_images(root: str) -> list[tuple[str, str]]:
    """
    Walk *root* and return (abs_path, label) for every image file found.

    The label is the name of the immediate sub-folder (e.g. "alice").
    Images directly inside *root* (not in a sub-folder) are skipped.
    """
    pairs = []
    for person_name in sorted(os.listdir(root)):
        person_dir = os.path.join(root, person_name)
        if not os.path.isdir(person_dir):
            continue
        for fname in sorted(os.listdir(person_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            pairs.append((os.path.abspath(os.path.join(person_dir, fname)),
                          person_name))
    return pairs


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_image(img_path: str, label: str) -> np.ndarray | None:
    """
    Load one image, detect + crop the largest face, return its embedding.

    Returns None if no face is detected or the embedding fails.
    """
    frame = cv2.imread(img_path)
    if frame is None:
        log.warning("Could not read image: %s", img_path)
        return None

    boxes = detect_faces(frame)
    if not boxes:
        log.warning("[%s] No face detected in %s — skipping.",
                    label, os.path.basename(img_path))
        return None

    # Use the largest detected face (most prominent in the photo)
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    face = crop_face(frame, boxes[0])

    embedding = get_embedding(face)
    if embedding is None:
        log.warning("[%s] Embedding failed for %s — skipping.",
                    label, os.path.basename(img_path))
    return embedding


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(force: bool = False) -> None:
    if not os.path.isdir(RAW_DIR):
        log.error("Raw data folder not found: %s", RAW_DIR)
        log.error("Run capture_faces.py first, or create the folder and drop images in.")
        sys.exit(1)

    all_images = find_images(RAW_DIR)
    if not all_images:
        log.error("No images found under %s", RAW_DIR)
        sys.exit(1)

    log.info("Found %d images across %d people.",
             len(all_images),
             len({label for _, label in all_images}))

    cache = load_cache() if not force else {
        "embeddings": np.empty((0, 512), dtype="float32"),
        "labels":     [],
        "processed":  set(),
    }

    new_embeddings: list[np.ndarray] = []
    new_labels:     list[str]        = []
    skipped  = 0
    failed   = 0

    for img_path, label in all_images:
        if img_path in cache["processed"]:
            skipped += 1
            continue

        log.info("  Processing [%s] %s", label, os.path.basename(img_path))
        emb = process_image(img_path, label)

        if emb is not None:
            new_embeddings.append(emb)
            new_labels.append(label)
            cache["processed"].add(img_path)
        else:
            failed += 1

    if new_embeddings:
        new_arr = np.vstack(new_embeddings).astype("float32")
        cache["embeddings"] = np.vstack(
            [cache["embeddings"], new_arr]
        ) if cache["embeddings"].shape[0] > 0 else new_arr
        cache["labels"].extend(new_labels)
        save_cache(cache)
    else:
        log.info("No new embeddings to add.")

    # Summary
    people = sorted(set(cache["labels"]))
    print("\n── Dataset summary ──────────────────────────────")
    for p in people:
        count = cache["labels"].count(p)
        print(f"  {p:<20}  {count:>3} embedding(s)")
    print(f"  {'TOTAL':<20}  {len(cache['labels']):>3}")
    print(f"\n  Skipped (already cached): {skipped}")
    print(f"  Failed (no face / error): {failed}")
    print("─────────────────────────────────────────────────")
    print("\nNext step:  python train_model.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build embedding cache from raw images.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute all embeddings, ignoring the cache.")
    args = parser.parse_args()
    main(force=args.force)
