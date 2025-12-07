# Swarlekha TTS Backend

FastAPI backend for Swarlekha Text-to-Speech model with voice cloning capabilities.

## Features

- Text-to-speech generation with default voice
- Voice cloning from reference audio
- Automatic output organization by voice name
- REST API with CORS support
- Automatic device detection (CUDA/MPS/CPU)

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Server

```bash
# Method 1: Using the start script
chmod +x start.sh
./start.sh

# Method 2: Direct Python (from backend directory)
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
python main.py

# Method 3: Using uvicorn directly
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server will start at `http://localhost:8000`

## API Endpoints

### Health Check

```
GET /api/health
```

### Generate Speech

```
POST /api/generate
Content-Type: multipart/form-data

Parameters:
- text: string (required) - Text to convert to speech
- reference_audio: file (optional) - Audio file for voice cloning
- voice_name: string (optional) - Name for organizing outputs (default: "default")
```

### List Voices

```
GET /api/voices
```

## Example Usage

### Using cURL

```bash
# Generate with default voice
curl -X POST "http://localhost:8000/api/generate" \
  -F "text=Hello, this is a test" \
  -F "voice_name=test" \
  --output output.wav

# Generate with voice cloning
curl -X POST "http://localhost:8000/api/generate" \
  -F "text=Hello, this is a test" \
  -F "reference_audio=@path/to/audio.wav" \
  -F "voice_name=custom" \
  --output output.wav
```

### Using Python

```python
import requests

url = "http://localhost:8000/api/generate"

# With voice cloning
with open("reference.wav", "rb") as audio_file:
    files = {"reference_audio": audio_file}
    data = {
        "text": "Hello, this is a test",
        "voice_name": "custom"
    }
    response = requests.post(url, data=data, files=files)

    with open("output.wav", "wb") as f:
        f.write(response.content)
```

## Directory Structure

```
backend/
├── main.py              # FastAPI application
├── models.py            # Pydantic models
├── requirements.txt     # Python dependencies
├── start.sh            # Start script
└── README.md           # This file
```

## Output Structure

Generated audio files are saved in:

```
examples/output/{voice_name}/generated_{timestamp}_{session_id}.wav
```

## Environment Variables

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)

## License

Same as the main Swarlekha project
