"""
train_model.py — Train the KNN classifier from the cached embeddings.

Usage:
    python train_model.py

Reads data/embeddings.pkl, runs GridSearchCV, prints metrics,
saves best model to models/knn_model.pkl.
"""

import os
import sys
import pickle
import logging
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

from config import EMBEDDINGS_PKL, MODEL_FILE, MODELS_DIR, PARAM_GRID, CV_FOLDS, TEST_SIZE, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_embeddings() -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(EMBEDDINGS_PKL):
        log.error("Embeddings not found: %s — run build_dataset.py first.", EMBEDDINGS_PKL)
        sys.exit(1)
    with open(EMBEDDINGS_PKL, "rb") as f:
        data = pickle.load(f)
    X = np.array(data["embeddings"], dtype="float32")
    y = np.array(data["labels"])
    if len(X) == 0:
        log.error("Embeddings file is empty.")
        sys.exit(1)
    unique, counts = np.unique(y, return_counts=True)
    log.info("Loaded %d embeddings — %d people:", len(y), len(unique))
    for name, cnt in zip(unique, counts):
        log.info("    %-20s  %d", name, cnt)
    return X, y


def main() -> None:
    X, y = load_embeddings()

    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        log.error("Need at least 2 people to train.")
        sys.exit(1)

    can_stratify = all(c >= int(1 / TEST_SIZE) for c in counts)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE,
        stratify=y if can_stratify else None,
        random_state=RANDOM_STATE,
    )

    cv_folds = max(2, min(CV_FOLDS, int(min(counts)) - 1))
    max_k    = len(X_tr)
    param_grid = {
        k: [v for v in vals if not (k == "n_neighbors" and v > max_k)]
        for k, vals in PARAM_GRID.items()
    }

    log.info("GridSearchCV with %d-fold CV ...", cv_folds)
    search = GridSearchCV(KNeighborsClassifier(), param_grid,
                          cv=cv_folds, scoring="accuracy", n_jobs=-1)
    search.fit(X_tr, y_tr)
    best = search.best_estimator_

    log.info("Best params : %s", search.best_params_)
    log.info("CV accuracy : %.3f", search.best_score_)

    y_pred = best.predict(X_te)
    print(f"\n── Hold-out metrics ─────────────────────")
    print(f"  Accuracy : {accuracy_score(y_te, y_pred):.3f}")
    print(f"  Precision: {precision_score(y_te, y_pred, average='macro', zero_division=0):.3f}")
    print(f"  Recall   : {recall_score(y_te, y_pred, average='macro', zero_division=0):.3f}")
    print(f"  F1       : {f1_score(y_te, y_pred, average='macro', zero_division=0):.3f}")
    print(f"\n── Per-class report ──────────────────────")
    print(classification_report(y_te, y_pred, target_names=list(unique), zero_division=0))

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(best, f)
    print(f"Model saved → {MODEL_FILE}")
    print("Run:  python recognize_live.py")


if __name__ == "__main__":
    main()
