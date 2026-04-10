#!/bin/bash
set -e

source venv/bin/activate

echo "============================================"
echo " Step 1/3 — Augmenting photos..."
echo "============================================"
python augment_dataset.py "$@"

echo ""
echo "============================================"
echo " Step 2/3 — Building embedding cache..."
echo "============================================"
python build_dataset.py "$@"

echo ""
echo "============================================"
echo " Step 3/3 — Training model..."
echo "============================================"
python train_model.py

echo ""
echo "============================================"
echo " Pipeline complete. Run recognize.sh next."
echo "============================================"
