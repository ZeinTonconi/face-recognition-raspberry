@echo off
call venv\Scripts\activate.bat
echo ============================================
echo  Face Registration
echo  Follow the on-screen prompts.
echo  You will capture 5 poses: CENTER, LEFT,
echo  RIGHT, UP, DOWN.
echo ============================================
python capture_faces.py
pause
