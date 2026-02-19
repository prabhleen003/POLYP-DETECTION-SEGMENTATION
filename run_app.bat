@echo off
REM SD-DeepLab Streamlit App Launcher (Windows)
REM This script automatically activates the virtual environment and runs the app

echo ==========================================
echo SD-DeepLab Polyp Segmentation App
echo ==========================================
echo.

REM Check if venv exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if requirements are installed
pip show streamlit > nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Run the app
echo.
echo Starting SD-DeepLab Streamlit App...
echo The app will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py

pause
