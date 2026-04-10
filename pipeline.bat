@echo off
call venv\Scripts\activate.bat
echo ============================================
echo  Step 1/3 — Augmenting photos...
echo ============================================
python augment_dataset.py %*
if %errorlevel% neq 0 ( echo ERROR in augmentation & pause & exit /b 1 )

echo.
echo ============================================
echo  Step 2/3 — Building embedding cache...
echo ============================================
python build_dataset.py %*
if %errorlevel% neq 0 ( echo ERROR in build & pause & exit /b 1 )

echo.
echo ============================================
echo  Step 3/3 — Training model...
echo ============================================
python train_model.py
if %errorlevel% neq 0 ( echo ERROR in training & pause & exit /b 1 )

echo.
echo ============================================
echo  Pipeline complete. Run recognize.bat next.
echo ============================================
pause
