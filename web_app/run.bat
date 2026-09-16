@echo off
setlocal EnableDelayedExpansion
title PulmoScan AI Workstation
echo ============================================
echo    PulmoScan AI - Lung Cancer Workstation
echo ============================================
echo.
REM ---- STEP 1: Find Python ----
echo [1/5] Finding Python...
set PYTHON_CMD=
python --version >nul 2>&1
if %errorlevel%==0 (
    set PYTHON_CMD=python
    goto :found_python
)
py --version >nul 2>&1
if %errorlevel%==0 (
    set PYTHON_CMD=py
    goto :found_python
)
echo ERROR: Python not found. Please install from https://python.org
pause
exit /b 1
:found_python
echo     OK - Using: %PYTHON_CMD%
echo.
REM ---- STEP 2: Remove old venv and recreate ----
echo [2/5] Preparing virtual environment...
if exist "venv" (
    echo     Removing old venv...
    rmdir /s /q venv
)
echo     Creating fresh virtual environment...
%PYTHON_CMD% -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Could not create virtual environment.
    pause
    exit /b 1
)
echo     Done.
echo.
REM ---- STEP 3: Activate ----
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
echo     Activated.
echo.
REM ---- STEP 4: Install dependencies ----
echo [4/5] Installing dependencies...
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Dependency installation failed.
    pause
    exit /b 1
)
echo     All dependencies ready.
echo.
REM ---- STEP 5: Check trained models ----
echo [5/5] Checking for trained AI models...
if exist "backend\lung_cancer_classifier.keras" (
    echo     [FOUND] lung_cancer_classifier.keras
) else (
    echo     [MISSING] lung_cancer_classifier.keras - CV fallback will be used
)
if exist "backend\lung_segmentation_model.keras" (
    echo     [FOUND] lung_segmentation_model.keras
) else (
    echo     [MISSING] lung_segmentation_model.keras - CV fallback will be used
)
echo.
REM ---- Launch Flask ----
echo ============================================
echo  Server starting at http://127.0.0.1:5000
echo  Press Ctrl+C to stop.
echo ============================================
echo.
python backend\app.py
pause