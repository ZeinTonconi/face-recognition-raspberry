# Face Recognition — Model Pipeline

## Project structure

```
project/
│
├── face_utils.py          # Shared helpers (detection, embedding, drawing, camera)
├── capture_faces.py       # Guided 5-angle webcam registration
├── build_dataset.py       # Scan raw images → compute embeddings → cache
├── train_model.py         # Load embeddings → GridSearchCV KNN → save model
├── recognize_live.py      # Webcam → live recognition with confidence score
│
├── data/
│   ├── raw/
│   │   ├── alice/         # One folder per person — exact name used as label
│   │   │   ├── alice_center.jpg
│   │   │   ├── alice_left.jpg
│   │   │   ├── alice_right.jpg
│   │   │   ├── alice_up.jpg
│   │   │   ├── alice_down.jpg
│   │   │   └── extra_photo.jpg   # Any extra uploaded images go here too
│   │   └── bob/
│   │       └── ...
│   └── embeddings.pkl     # Auto-generated — do not edit manually
│
└── models/
    └── knn_model.pkl      # Auto-generated — do not edit manually
```

## Workflow

### Step 1 — Register a person via webcam

```bash
python capture_faces.py
```

You will be guided through 5 poses: CENTER → LEFT → RIGHT → UP → DOWN.  
Press **SPACE** to capture each pose. Press **ESC** to abort.

Images are saved to `data/raw/<name>/`.

### Step 2 — (Optional) Add extra photos by upload

Drop any `.jpg`, `.jpeg`, or `.png` files directly into `data/raw/<name>/`.  
No special naming is required — the folder name is the label.

### Step 3 — Build the embedding cache

```bash
python build_dataset.py
```

Scans `data/raw/`, detects faces, computes Facenet512 embeddings, and saves
everything to `data/embeddings.pkl`.

This step is **incremental by default** — only new images are processed.  
To recompute everything from scratch:

```bash
python build_dataset.py --force
```

### Step 4 — Train the classifier

```bash
python train_model.py
```

Loads the embedding cache, runs a cross-validated hyperparameter search,
prints accuracy/precision/recall/F1, and saves the best KNN to
`models/knn_model.pkl`.

Re-run this step any time you add new people or photos.

### Step 5 — Run live recognition

```bash
python recognize_live.py
```

Opens the webcam and displays a bounding box with the person's name and
confidence percentage.  Press **ESC** to quit.

---

## Tuning

| Parameter | File | Default | Effect |
|---|---|---|---|
| `UNKNOWN_THRESHOLD` | `recognize_live.py` | `55.0` | Min confidence (%) to show a name instead of "Unknown". Raise to reduce false positives. |
| `EMBED_EVERY_N` | `recognize_live.py` | `5` | Run DeepFace every N frames. Lower = more responsive, higher = smoother video. |
| `PARAM_GRID` | `train_model.py` | k=3/5/7, euclidean/cosine | KNN hyperparameter search space. |

## Dependencies

```
opencv-python
mediapipe
deepface
scikit-learn
numpy
```
