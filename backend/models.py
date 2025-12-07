from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to convert to speech")
    voice_name: Optional[str] = Field(default="default", description="Name for organizing output files")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "The only way to do great work is to love what you do.",
                "voice_name": "demo"
            }
        }

class GenerateResponse(BaseModel):
    session_id: str
    voice_name: str
    output_path: str
    filename: str
    audio_url: str
    created_at: datetime

class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool

class VoiceInfo(BaseModel):
    name: str
    count: int
    files: list[str]

class VoicesResponse(BaseModel):
    voices: list[VoiceInfo]
