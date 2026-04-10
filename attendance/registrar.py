"""
attendance/registrar.py — Live registration pipeline.

Handles the full flow of registering a new person without leaving
the recognition screen:
  1. Capture one frontal photo from the webcam
  2. Generate QUICK_AUG_COUNT augmented variants
  3. Compute embeddings for all variants
  4. Add them to the embeddings cache on disk
  5. Retrain the KNN model
  6. Return the new model so recognize_live can hot-swap it

Designed to run in a background thread so the camera preview stays live.
"""

import os
import pickle
import logging
import random
from pathlib import Path

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from config import (
    RAW_DIR, EMBEDDINGS_PKL, MODEL_FILE, MODELS_DIR,
    FACE_SIZE, PARAM_GRID, CV_FOLDS, QUICK_AUG_COUNT,
)
from core.detector import detect_faces, crop_face
from core.embedder import get_embedding

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ===========================================================================
# Augmentation (quick subset — 15 variants from QUICK_AUG_COUNT)
# ===========================================================================

def _rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def _quick_augment(face_bgr: np.ndarray, count: int) -> list[np.ndarray]:
    """
    Generate *count* augmented variants from a single face crop.
    Always includes the original as the first entry.
    """
    face = cv2.resize(face_bgr, FACE_SIZE)

    # Full pool of cheap augmentations — pick count-1 from these
    pool = [
        lambda i: _rotate(i,  8),
        lambda i: _rotate(i, -8),
        lambda i: _rotate(i, 15),
        lambda i: _rotate(i,-15),
        lambda i: cv2.flip(i, 1),
        lambda i: cv2.convertScaleAbs(i, alpha=1.0, beta=30),
        lambda i: cv2.convertScaleAbs(i, alpha=1.0, beta=-30),
        lambda i: cv2.convertScaleAbs(i, alpha=1.0, beta=60),
        lambda i: cv2.convertScaleAbs(i, alpha=1.0, beta=-60),
        lambda i: cv2.convertScaleAbs(i, alpha=1.3, beta=0),
        lambda i: cv2.convertScaleAbs(i, alpha=0.7, beta=0),
        lambda i: cv2.GaussianBlur(i, (3, 3), 0),
        lambda i: cv2.GaussianBlur(i, (5, 5), 0),
        lambda i: (lambda n: np.clip(i.astype("float32") + n, 0, 255).astype("uint8"))(
            np.random.normal(0, 10, i.shape).astype("float32")),
        lambda i: (lambda hsv: cv2.cvtColor(
            (hsv * [1, 1.6, 1]).clip(0, 255).astype("uint8"), cv2.COLOR_HSV2BGR))(
            cv2.cvtColor(i, cv2.COLOR_BGR2HSV).astype("float32")),
        lambda i: (lambda hsv: cv2.cvtColor(
            (hsv * [1, 0.4, 1]).clip(0, 255).astype("uint8"), cv2.COLOR_HSV2BGR))(
            cv2.cvtColor(i, cv2.COLOR_BGR2HSV).astype("float32")),
        lambda i: cv2.flip(_rotate(i, 8), 1),
        lambda i: cv2.convertScaleAbs(_rotate(i, -10), alpha=1.0, beta=25),
        lambda i: cv2.GaussianBlur(cv2.flip(i, 1), (3, 3), 0),
    ]

    selected = random.sample(pool, min(count - 1, len(pool)))
    result   = [face]
    for fn in selected:
        try:
            result.append(fn(face))
        except Exception:
            pass
    return result


# ===========================================================================
# Embedding cache helpers
# ===========================================================================

def _load_cache() -> dict:
    if os.path.exists(EMBEDDINGS_PKL):
        with open(EMBEDDINGS_PKL, "rb") as f:
            data = pickle.load(f)
        if "processed" not in data:
            data["processed"] = set()
        return data
    return {
        "embeddings": np.empty((0, 128), dtype="float32"),
        "labels":     [],
        "processed":  set(),
    }


def _save_cache(data: dict) -> None:
    os.makedirs(os.path.dirname(EMBEDDINGS_PKL), exist_ok=True)
    with open(EMBEDDINGS_PKL, "wb") as f:
        pickle.dump(data, f)


# ===========================================================================
# KNN retraining
# ===========================================================================

def _retrain(X: np.ndarray, y: np.ndarray):
    """
    Retrain KNN with GridSearchCV. Returns the best estimator.
    Clamps CV folds to avoid errors with small datasets.
    """
    unique, counts = np.unique(y, return_counts=True)
    cv_folds = min(CV_FOLDS, int(min(counts)) - 1)
    cv_folds = max(cv_folds, 2)

    # Clamp n_neighbors to training set size
    max_k = len(X)
    param_grid = {
        k: [v for v in vals if not (k == "n_neighbors" and v > max_k)]
        for k, vals in PARAM_GRID.items()
    }

    search = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=cv_folds,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(X, y)
    log.info("Retrain done — best params: %s  cv acc: %.3f",
             search.best_params_, search.best_score_)
    return search.best_estimator_


# ===========================================================================
# Save raw photo
# ===========================================================================

def _save_raw_photo(face_bgr: np.ndarray, name: str) -> str:
    """Save the original captured face to data/raw/<name>/."""
    person_dir = os.path.join(RAW_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    # Find next available index
    existing = list(Path(person_dir).glob("*.jpg"))
    idx      = len(existing)
    path     = os.path.join(person_dir, f"{name}_live_{idx:03d}.jpg")
    cv2.imwrite(path, face_bgr)
    return path


# ===========================================================================
# Main entry point
# ===========================================================================

def register_person(
    name: str,
    face_bgr: np.ndarray,
    on_progress=None,    # callable(message: str) for UI updates
) -> object:
    """
    Full registration pipeline for one person.

    Args:
        name:        Person's name (used as label and folder name).
        face_bgr:    Captured face crop (BGR, any size).
        on_progress: Optional callback for status messages shown in the UI.

    Returns:
        Newly trained KNN model (ready to hot-swap into recognize_live).

    Raises:
        RuntimeError if embedding fails or dataset has < 2 people.
    """

    def progress(msg: str) -> None:
        log.info(msg)
        if on_progress:
            on_progress(msg)

    # ── 1. Save raw photo ────────────────────────────────────────────────
    progress("Saving photo...")
    raw_path = _save_raw_photo(face_bgr, name)

    # ── 2. Generate augmentations ────────────────────────────────────────
    progress(f"Generating {QUICK_AUG_COUNT} augmentations...")
    variants = _quick_augment(face_bgr, QUICK_AUG_COUNT)

    # ── 3. Compute embeddings ────────────────────────────────────────────
    progress("Computing embeddings...")
    new_embeddings = []
    for variant in variants:
        emb = get_embedding(variant)
        if emb is not None:
            new_embeddings.append(emb)

    if not new_embeddings:
        raise RuntimeError(
            f"Could not compute any embeddings for {name}. "
            "Check that the face is clearly visible."
        )

    # ── 4. Update cache ──────────────────────────────────────────────────
    progress("Updating dataset...")
    cache = _load_cache()

    new_arr = np.vstack(new_embeddings).astype("float32")
    if cache["embeddings"].shape[0] > 0:
        cache["embeddings"] = np.vstack([cache["embeddings"], new_arr])
    else:
        cache["embeddings"] = new_arr

    cache["labels"].extend([name] * len(new_embeddings))
    cache["processed"].add(os.path.abspath(raw_path))
    _save_cache(cache)

    progress(f"Added {len(new_embeddings)} embeddings for {name}.")

    # ── 5. Retrain KNN ───────────────────────────────────────────────────
    X = cache["embeddings"]
    y = np.array(cache["labels"])

    unique = np.unique(y)
    if len(unique) < 2:
        raise RuntimeError(
            "Need at least 2 people registered to train the classifier. "
            f"Currently only have: {list(unique)}"
        )

    progress("Retraining model...")
    model = _retrain(X, y)

    # ── 6. Save model to disk ────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    progress(f"Done! {name} registered successfully.")
    return model
