# Swarlekha: Advanced Text-to-Speech and Voice Conversion System

**Swarlekha** is a state-of-the-art AI-powered text-to-speech (TTS) and voice conversion (VC) system with a complete full-stack web application. Built on advanced neural architectures, it supports zero-shot voice cloning—generate speech in any voice with just a short audio sample.

## 🚀 Features

### Core ML Features
- **High-Quality Text-to-Speech**: Generate natural, expressive speech from text
- **Zero-Shot Voice Cloning**: Clone any voice with just a few seconds of audio
- **Voice Conversion**: Transform the voice characteristics of existing audio
- **Multi-Device Support**: Automatic detection and optimization for CUDA, MPS, and CPU

### Web Application Features
- **FastAPI Backend**: REST API with automatic documentation
- **React Frontend**: Modern UI with glassmorphism design
- **Drag & Drop Audio Upload**: Easy reference audio uploading
- **Voice Recording**: Record directly in browser for voice cloning
- **Real-time Audio Playback**: Play and download generated audio
- **Fully Responsive**: Mobile-friendly design

## 📁 Project Structure

```
swarlekha/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── inference_tts.py             # TTS inference script
├── inference_vc.py              # Voice conversion inference script
├── start_all.sh                 # Start both backend and frontend
├── stop_all.sh                  # Stop all services
├── backend/                     # FastAPI REST API
│   ├── main.py                  # API server
│   ├── models.py                # Pydantic data models
│   └── start.sh                 # Backend start script
├── frontend/                    # React TypeScript UI
│   ├── src/
│   │   ├── components/          # UI components
│   │   ├── services/api.ts      # API client
│   │   └── App.tsx              # Main app
│   └── public/demo/             # Demo audio files
├── examples/                    # Input/output audio examples
└── swarlekha_model/             # Core model implementation
    ├── tts.py                   # TTS model wrapper
    ├── vc.py                    # Voice conversion wrapper
    └── models/                  # Neural network architectures
        ├── s3gen/               # Speech generation model
        ├── s3tokenizer/         # Speech tokenization
        ├── t3/                  # Text processing model
        ├── tokenizers/          # Text tokenization
        └── voice_encoder/       # Voice encoding for cloning
```

## 🛠 Installation & Setup

### Prerequisites
- Python 3.10
- Node.js 18+
- CUDA-compatible GPU (recommended) or CPU
- At least 8GB RAM (16GB+ recommended for GPU usage)

### Quick Start (Full Stack)

```bash
# 1. Clone the repository
git clone https://github.com/bishal2059/swarlekha.git
cd swarlekha

# 2. Backend setup
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Frontend setup
cd frontend
npm install
cd ..

# 4. Start everything
chmod +x start_all.sh stop_all.sh
./start_all.sh
```

**Application URLs:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Python-Only Setup (Without Web App)

```bash
# Clone and setup
git clone https://github.com/bishal2059/swarlekha.git
cd swarlekha

# Create conda environment
conda create -n swarlekha python=3.10 -y
conda activate swarlekha

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Dependencies
- `torch>=2.6.0` - PyTorch deep learning framework
- `torchaudio>=2.6.0` - Audio processing
- `transformers>=4.46.3` - Hugging Face transformers
- `librosa>=0.11.0` - Audio analysis
- `diffusers>=0.29.0` - Diffusion models
- `safetensors>=0.5.3` - Safe tensor serialization
- Additional: `numpy`, `s3tokenizer`, `resemble-perth`, `conformer`

## 🔄 How the Project is Initialized

### Model Loading
The ML models are **automatically downloaded** from Hugging Face ([ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox)) on first run and cached locally in the Hugging Face cache directory (`~/.cache/huggingface/`).

### Backend Initialization
When `python main.py` runs in the backend:
1. FastAPI server starts on port 8000
2. The `SwarlekhaTTS` model is loaded (triggers model download on first run)
3. Device auto-detection selects CUDA > MPS > CPU
4. API endpoints become available at `/api/*`

### Frontend Initialization
When `npm run dev` runs in the frontend:
1. Vite development server starts on port 3000
2. React app loads with environment variables from `.env`
3. API client connects to backend at `VITE_API_URL` (default: http://localhost:8000)

### Startup Scripts
- **`start_all.sh`**: Launches both backend and frontend in background processes
- **`stop_all.sh`**: Terminates all running services

## 🎯 Usage

### Option 1: Web Application

1. Start the application: `./start_all.sh`
2. Open browser: http://localhost:3000
3. Enter text in the text area
4. Choose voice type:
   - **Default Voice**: Click "Default Voice" button
   - **Clone Voice**: Upload audio or record directly
5. Click "Generate Voice"
6. Play or download the generated audio

To stop: `./stop_all.sh`

### Option 2: Python Scripts

**Text-to-Speech (TTS):**

```python
import torchaudio as ta
from swarlekha_model.tts import SwarlekhaTTS

# Initialize model (auto-detects best device)
model = SwarlekhaTTS.from_pretrained()

# Generate speech with default voice
text = "Hello, this is Swarlekha speaking with natural voice synthesis."
wav = model.generate(text)
ta.save("output.wav", wav, model.sr)

# Clone voice from audio sample
audio_prompt = "path/to/reference_voice.wav"
wav_cloned = model.generate(text, audio_prompt_path=audio_prompt)
ta.save("output_cloned.wav", wav_cloned, model.sr)
```

**Voice Conversion (VC):**

```python
import torchaudio as ta
from swarlekha_model.vc import SwarlekhaVC

# Initialize voice conversion model
model = SwarlekhaVC.from_pretrained()

# Convert voice
source_audio = "path/to/source_audio.wav"
target_voice = "path/to/target_voice.wav"

converted_wav = model.generate(
    audio=source_audio,
    target_voice_path=target_voice
)
ta.save("converted_output.wav", converted_wav, model.sr)
```

### Running Inference Scripts

**Text-to-Speech:**
```bash
python inference_tts.py
```

**Voice Conversion:**
```bash
python inference_vc.py
```

Make sure to place your reference audio files in the `examples/input/` directory and check the `examples/output/` directory for generated results.

## 🔧 Configuration

The system automatically detects and uses the best available device:
- **CUDA**: For NVIDIA GPUs (recommended for fastest inference)
- **MPS**: For Apple Silicon Macs
- **CPU**: Fallback option (slower but works on any system)

## 📊 Model Architecture

Swarlekha leverages several advanced neural network architectures:

- **T3 Model**: Text processing and linguistic feature extraction
- **S3Gen**: High-quality speech generation using flow matching
- **S3Tokenizer**: Efficient speech tokenization
- **Voice Encoder**: Speaker embedding extraction for voice cloning
- **HiFi-GAN**: Neural vocoder for audio synthesis
- **Transformer Modules**: Attention mechanisms for sequence modeling

## 🎵 Audio Requirements

- **Input Audio**: WAV format, 16kHz+ sample rate recommended
- **Reference Voice**: 3-10 seconds of clear speech (longer is better)
- **Output**: High-quality WAV files at model's native sample rate

## 🚧 Future Features

### Planned Enhancements
- [ ] **Nepali Language Support**: Complete integration for Nepali text-to-speech voice cloning
- [ ] **Real-time Inference**: Optimizations for live speech generation
- [ ] **Fine-tuning Framework**: Custom model training on domain-specific data

## 🌐 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/generate` | Generate speech from text |
| GET | `/api/voices` | List available voices |

### Generate Speech

```bash
# Default voice
curl -X POST "http://localhost:8000/api/generate" \
  -F "text=Hello, this is a test" \
  -F "voice_name=test" \
  --output output.wav

# With voice cloning
curl -X POST "http://localhost:8000/api/generate" \
  -F "text=Hello, this is a test" \
  -F "reference_audio=@path/to/audio.wav" \
  -F "voice_name=custom" \
  --output output.wav
```


## 🙏 Acknowledgments

We extend our heartfelt gratitude to **Resemble AI** for open-sourcing the Chatterbox model, which forms the foundation of Swarlekha:

- **Resemble AI Chatterbox**: This project is built upon the excellent [Chatterbox](https://github.com/resemble-ai/chatterbox) architecture by Resemble AI
- **Pre-trained Models**: We utilize the publicly available weights from the [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) repository on Hugging Face
- **Research Paper**: Based on the groundbreaking work described in their research publications
- **Open Source Community**: Special thanks to Resemble AI for contributing to the open-source AI community and making advanced TTS technology accessible

### Additional Acknowledgments
- [Hugging Face](https://huggingface.co/) for the transformers ecosystem and model hosting
- [PyTorch](https://pytorch.org/) team for the deep learning framework
- The broader speech synthesis research community for advancing the field

---

**🌟 Star this repository if you find Swarlekha useful!**

**📧 Contact**: For questions, issues, or collaboration opportunities, please open an issue on GitHub.

**📚 Usage Note**: This project is intended for research and educational purposes only. Please ensure compliance with applicable laws and ethical guidelines when using voice cloning technology.



