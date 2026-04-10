@echo off
echo ============================================
echo  Creating virtual environment...
echo ============================================
python -m venv venv
call venv\Scripts\activate.bat

echo.
echo ============================================
echo  Installing dependencies...
echo ============================================
pip install --upgrade pip
pip install cmake
pip install dlib
pip install face_recognition
pip install mediapipe
pip install opencv-python
pip install scikit-learn
pip install numpy
pip install requests

echo.
echo ============================================
echo  Done. You can now run register.bat
echo ============================================
pause
