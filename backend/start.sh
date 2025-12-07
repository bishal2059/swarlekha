#!/bin/bash

echo "Starting Swarlekha TTS Backend..."

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install requirements if needed
pip install -r requirements.txt

# Run the server
python main.py
