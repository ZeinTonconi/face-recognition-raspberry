"""
config.py — Central configuration for the face recognition attendance system.

Every tunable constant lives here. No magic numbers anywhere else.
"""

import os
import platform

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR       = "data"
RAW_DIR        = os.path.join(DATA_DIR, "raw")
EMBEDDINGS_PKL = os.path.join(DATA_DIR, "embeddings.pkl")
MODELS_DIR     = "models"
MODEL_FILE     = os.path.join(MODELS_DIR, "knn_model.pkl")

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
FACE_SIZE             = (160, 160)   # crop size before embedding
IOU_THRESHOLD         = 0.45        # min IoU to continue tracking a box
MIN_DETECT_CONFIDENCE = 0.5         # mediapipe detection threshold

# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------
UNKNOWN_THRESHOLD  = 55.0   # confidence % below this → shown as "Unknown"
CONFIRM_THRESHOLD  = 75.0   # confidence % required to start the 3s timer
EMBEDDING_MODEL    = "small" # "small" = faster, "large" = more accurate

# ---------------------------------------------------------------------------
# Attendance
# ---------------------------------------------------------------------------
CONFIRM_SECONDS  = 3.0    # seconds same person must stay in frame
COOLDOWN_MINUTES = 20
     # minutes before same person can be recorded again

# ---------------------------------------------------------------------------
# Camera / display
# ---------------------------------------------------------------------------
BOX_JUMP_THRESHOLD = 80   # pixels — jump larger than this = new person
BOX_MOVE_THRESHOLD = 40   # pixels — movement needed to submit new embedding
CAP_BACKEND = (
    "dshow" if platform.system() == "Windows" else "v4l2"
)

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
SERVER_URL   = "http://98.88.188.212:3000/api/v1/access-logs/register"
DEVICE_ID    = "windows-dev-1"
HTTP_TIMEOUT = 5   # seconds before a POST is considered failed

# ---------------------------------------------------------------------------
# Live registration
# ---------------------------------------------------------------------------
QUICK_AUG_COUNT = 15   # augmentations generated per photo during live register
                        # raise for production, keep low for fast demo

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "n_neighbors": [3, 5, 7],
    "weights":     ["uniform", "distance"],
    "metric":      ["euclidean", "cosine"],
}
CV_FOLDS     = 5
TEST_SIZE    = 0.2
RANDOM_STATE = 42
