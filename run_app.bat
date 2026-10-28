@echo off
title Alzheimer's Risk Predictor
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: venv folder not found. Put this file in the main project folder.
    pause
    exit /b
)

call venv\Scripts\activate.bat

echo Starting the app... your browser will open in a few seconds.
echo Close this window (or press Ctrl + C) to stop the app.
echo.

streamlit run app.py

pause