#!/bin/bash
# Run Streamlit application on macOS/Linux

echo "Starting Klasifikasi IPM Application..."
echo "Opening browser at http://localhost:8501"

# Check if virtual environment exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
fi

# Run streamlit
streamlit run app.py
