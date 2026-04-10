@echo off
call venv\Scripts\activate.bat
echo ============================================
echo  Starting live face recognition...
echo  Press ESC or Q to quit.
echo ============================================
python recognize_live.py
pause
