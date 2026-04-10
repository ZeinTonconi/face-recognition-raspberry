#!/bin/bash
source venv/bin/activate
echo "============================================"
echo " Face Registration"
echo " You will capture 5 poses: CENTER, LEFT,"
echo " RIGHT, UP, DOWN."
echo "============================================"
python capture_faces.py
