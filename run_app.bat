@echo off
cd /d "%~dp0"

echo Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install it from python.org
    echo and tick "Add Python to PATH" during installation.
    pause
    exit /b
)

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing required libraries. Please wait...
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn streamlit matplotlib seaborn joblib

echo Starting the app...
streamlit run app.py
pause