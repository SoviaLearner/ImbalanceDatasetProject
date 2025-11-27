@echo off
REM Run Streamlit application on Windows

echo Starting Klasifikasi IPM Application...
echo Opening browser at http://localhost:8501

REM Check if virtual environment exists
if exist "venv" (
    call venv\Scripts\activate.bat
    echo Virtual environment activated
)

REM Run streamlit
streamlit run app.py
pause
