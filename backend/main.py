from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torchaudio as ta
import torch
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
import tempfile
from typing import Optional

# Add parent directory to path to import swarlekha_model
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarlekha_model.tts import SwarlekhaTTS

app = FastAPI(title="Swarlekha TTS API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Load model at startup
model = None

@app.on_event("startup")
async def startup_event():
    global model
    model = SwarlekhaTTS.from_pretrained(device=device)
    print("Model loaded successfully!")

@app.get("/")
async def root():
    return {
        "message": "Swarlekha TTS API",
        "version": "1.0.0",
        "device": device,
        "endpoints": {
            "generate": "/api/generate",
            "health": "/api/health"
        }
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "device": device,
        "model_loaded": model is not None
    }

@app.post("/api/generate")
async def generate_voice(
    text: str = Form(...),
    reference_audio: Optional[UploadFile] = File(None),
    voice_name: Optional[str] = Form("default")
):
    """
    Generate speech from text with optional voice cloning.
    
    Parameters:
    - text: The text to convert to speech
    - reference_audio: Optional audio file for voice cloning
    - voice_name: Name for organizing output files (default: "default")
    """
    if not text or text.strip() == "":
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="Text is too long (max 5000 characters)")
    
    try:
        # Create unique session ID
        session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        output_dir = Path(f"examples/output/{voice_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        output_filename = f"generated_{timestamp}_{session_id}.wav"
        output_path = output_dir / output_filename
        
        # Handle reference audio if provided
        audio_prompt_path = None
        temp_audio_path = None
        
        if reference_audio:
            # Save uploaded file temporarily
            temp_audio_path = Path(tempfile.gettempdir()) / f"ref_audio_{session_id}.wav"
            with open(temp_audio_path, "wb") as buffer:
                content = await reference_audio.read()
                buffer.write(content)
            audio_prompt_path = str(temp_audio_path)
        
        # Generate speech
        if audio_prompt_path:
            wav = model.generate(text, audio_prompt_path=audio_prompt_path)
        else:
            wav = model.generate(text)
        
        # Save output
        ta.save(str(output_path), wav, model.sr)
        
        # Clean up temporary file
        if temp_audio_path and temp_audio_path.exists():
            temp_audio_path.unlink()
        
        # Return the generated audio file
        return FileResponse(
            path=str(output_path),
            media_type="audio/wav",
            filename=output_filename,
            headers={
                "X-Session-ID": session_id,
                "X-Voice-Name": voice_name,
                "X-Output-Path": str(output_path)
            }
        )
    
    except Exception as e:
        # Clean up on error
        if temp_audio_path and temp_audio_path.exists():
            temp_audio_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error generating voice: {str(e)}")

@app.get("/api/voices")
async def list_voices():
    """List all available voice outputs"""
    output_base = Path("examples/output")
    if not output_base.exists():
        return {"voices": []}
    
    voices = []
    for voice_dir in output_base.iterdir():
        if voice_dir.is_dir():
            files = list(voice_dir.glob("*.wav"))
            voices.append({
                "name": voice_dir.name,
                "count": len(files),
                "files": [f.name for f in files]
            })
    
    return {"voices": voices}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
