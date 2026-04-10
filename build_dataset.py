"""
build_dataset.py — Build (or rebuild) the embedding cache from raw face images.

Usage:
    python build_dataset.py              # process everything in data/raw/
    python build_dataset.py --force      # recompute all embeddings from scratch

Folder layout expected:
    data/raw/<name>/
        <name>_center.jpg
        <name>_left.jpg
        ... (any .jpg/.jpeg/.png files)

Incremental by default — already-processed images are skipped.
After this script, run train_model.py.
"""

import os
import sys
import pickle
import argparse
import logging
import numpy as np
import cv2

from config import RAW_DIR, EMBEDDINGS_PKL
from core.detector import detect_faces, crop_face
from core.embedder import get_embedding

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_cache() -> dict:
    if os.path.exists(EMBEDDINGS_PKL):
        with open(EMBEDDINGS_PKL, "rb") as f:
            data = pickle.load(f)
        if "processed" not in data:
            data["processed"] = set()
        log.info("Loaded cache: %d embeddings, %d people",
                 len(data["labels"]), len(set(data["labels"])))
        return data
    return {
        "embeddings": np.empty((0, 128), dtype="float32"),
        "labels":     [],
        "processed":  set(),
    }


def save_cache(data: dict) -> None:
    os.makedirs(os.path.dirname(EMBEDDINGS_PKL), exist_ok=True)
    with open(EMBEDDINGS_PKL, "wb") as f:
        pickle.dump(data, f)
    log.info("Cache saved → %s  (%d embeddings)", EMBEDDINGS_PKL, len(data["labels"]))


def find_images(root: str) -> list[tuple[str, str]]:
    pairs = []
    for person_name in sorted(os.listdir(root)):
        person_dir = os.path.join(root, person_name)
        if not os.path.isdir(person_dir):
            continue
        for fname in sorted(os.listdir(person_dir)):
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                pairs.append((os.path.abspath(os.path.join(person_dir, fname)),
                               person_name))
    return pairs


def process_image(img_path: str, label: str) -> np.ndarray | None:
    frame = cv2.imread(img_path)
    if frame is None:
        log.warning("Could not read: %s", img_path)
        return None
    boxes = detect_faces(frame)
    if not boxes:
        log.warning("[%s] No face in %s — skipping.", label, os.path.basename(img_path))
        return None
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return get_embedding(crop_face(frame, boxes[0]))


def main(force: bool = False) -> None:
    if not os.path.isdir(RAW_DIR):
        log.error("Raw data folder not found: %s", RAW_DIR)
        sys.exit(1)

    all_images = find_images(RAW_DIR)
    if not all_images:
        log.error("No images found under %s", RAW_DIR)
        sys.exit(1)

    log.info("Found %d images across %d people.",
             len(all_images), len({l for _, l in all_images}))

    cache = load_cache() if not force else {
        "embeddings": np.empty((0, 128), dtype="float32"),
        "labels": [], "processed": set(),
    }

    new_embeddings, new_labels = [], []
    skipped = failed = 0

    for img_path, label in all_images:
        if img_path in cache["processed"]:
            skipped += 1
            continue
        log.info("  [%s] %s", label, os.path.basename(img_path))
        emb = process_image(img_path, label)
        if emb is not None:
            new_embeddings.append(emb)
            new_labels.append(label)
            cache["processed"].add(img_path)
        else:
            failed += 1

    if new_embeddings:
        new_arr = np.vstack(new_embeddings).astype("float32")
        cache["embeddings"] = (np.vstack([cache["embeddings"], new_arr])
                               if cache["embeddings"].shape[0] > 0 else new_arr)
        cache["labels"].extend(new_labels)
        save_cache(cache)
    else:
        log.info("No new embeddings to add.")

    people = sorted(set(cache["labels"]))
    print("\n── Dataset summary ──────────────────────")
    for p in people:
        print(f"  {p:<20}  {cache['labels'].count(p):>3} embedding(s)")
    print(f"  {'TOTAL':<20}  {len(cache['labels']):>3}")
    print(f"\n  Skipped (cached): {skipped}  |  Failed: {failed}")
    print("─────────────────────────────────────────")
    print("\nNext step:  python train_model.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
