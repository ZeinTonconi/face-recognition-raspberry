"""
train_model.py — Train the KNN classifier from the cached embeddings.

Usage:
    python train_model.py

Reads data/embeddings.pkl (built by build_dataset.py), runs a cross-validated
hyperparameter search, prints evaluation metrics, and saves the best model to
models/knn_model.pkl.

Re-run this script any time you add new people or new photos.
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDINGS_PKL = os.path.join("data", "embeddings.pkl")
MODELS_DIR     = "models"
MODEL_FILE     = os.path.join(MODELS_DIR, "knn_model.pkl")

PARAM_GRID = {
    "n_neighbors": [3, 5, 7],
    "weights":     ["uniform", "distance"],
    "metric":      ["euclidean", "cosine"],
}
CV_FOLDS     = 5
TEST_SIZE    = 0.2
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Load (X, y) from the embeddings cache."""
    if not os.path.exists(EMBEDDINGS_PKL):
        log.error("Embeddings file not found: %s", EMBEDDINGS_PKL)
        log.error("Run build_dataset.py first.")
        sys.exit(1)

    with open(EMBEDDINGS_PKL, "rb") as f:
        data = pickle.load(f)

    X = np.array(data["embeddings"], dtype="float32")
    y = np.array(data["labels"])

    if len(X) == 0:
        log.error("Embeddings file is empty. Run build_dataset.py.")
        sys.exit(1)

    unique, counts = np.unique(y, return_counts=True)
    log.info("Loaded %d embeddings — %d people:", len(y), len(unique))
    for name, cnt in zip(unique, counts):
        log.info("    %-20s  %d sample(s)", name, cnt)

    return X, y


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                  classes: list[str]) -> None:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred,    average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred,        average="macro", zero_division=0)

    print(f"\n── Hold-out metrics ({int(TEST_SIZE*100)}% test split) ───────────")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Precision: {prec:.3f}  (macro)")
    print(f"  Recall   : {rec:.3f}  (macro)")
    print(f"  F1       : {f1:.3f}  (macro)")
    print("\n── Per-class report ──────────────────────────────")
    print(classification_report(y_true, y_pred,
                                target_names=classes, zero_division=0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    X, y = load_embeddings()

    # ── Minimum sample guard ──────────────────────────────────────────────
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        log.error("Need at least 2 people in the dataset to train a classifier.")
        sys.exit(1)

    # Stratified split requires enough samples per class.
    # If any class has fewer than 1/TEST_SIZE samples, skip stratification.
    min_samples_for_strat = int(1 / TEST_SIZE)
    can_stratify = all(c >= min_samples_for_strat for c in counts)
    if not can_stratify:
        log.warning("Some classes have very few samples — "
                    "using non-stratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y if can_stratify else None,
        random_state=RANDOM_STATE,
    )

    # ── GridSearchCV ──────────────────────────────────────────────────────
    # Clamp n_neighbors to never exceed training set size
    max_k = len(X_train)
    param_grid = {
        k: [v for v in vals if not (k == "n_neighbors" and v > max_k)]
        for k, vals in PARAM_GRID.items()
    }

    cv_folds = min(CV_FOLDS, min(counts))   # folds ≤ smallest class size
    log.info("GridSearchCV with %d-fold CV ...", cv_folds)

    search = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=cv_folds,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    log.info("Best params : %s", search.best_params_)
    log.info("CV accuracy : %.3f", search.best_score_)

    # ── Evaluate on hold-out ──────────────────────────────────────────────
    y_pred = best.predict(X_test)
    print_metrics(y_test, y_pred, classes=list(unique))

    # ── Save model ────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(best, f)

    print(f"\nModel saved → {MODEL_FILE}")
    print("Run  recognize_live.py  to start live recognition.")


if __name__ == "__main__":
    main()
