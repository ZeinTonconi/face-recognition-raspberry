"""
core/classifier.py — KNN model loading and prediction.
"""

import os
import sys
import pickle
import numpy as np

from config import MODEL_FILE


def load_model():
    """Load the trained KNN model from disk. Exits if not found."""
    if not os.path.exists(MODEL_FILE):
        print(f"Model not found: {MODEL_FILE}")
        print("Run pipeline.bat first.")
        sys.exit(1)
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded — classes: {model.classes_.tolist()}")
    return model


def predict(model, embedding: np.ndarray) -> tuple[str, float]:
    """
    Run KNN on a single embedding.
    Returns (name, confidence_percent).
    """
    vec = embedding.reshape(1, -1)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        idx   = int(np.argmax(probs))
        return str(model.classes_[idx]), float(probs[idx]) * 100.0
    return str(model.predict(vec)[0]), 100.0
