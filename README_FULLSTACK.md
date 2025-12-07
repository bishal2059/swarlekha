# 🎤 Swarlekha TTS - Complete Full-Stack Application

**Swarlekha** is a state-of-the-art AI-powered text-to-speech (TTS) system with advanced voice cloning capabilities. This repository includes both the ML model and a complete production-ready web application.

## ✨ What's New: Full-Stack Web Application

This project now includes a **complete full-stack application** with:

- 🔴 **FastAPI Backend** - REST API for voice generation
- 🔵 **React Frontend** - Beautiful, modern web interface
- 🎨 **Professional UI** - Glassmorphism design with animations
- 🚀 **Production Ready** - Docker support and complete documentation

👉 **[Quick Start Guide](QUICKSTART.md)** | **[Complete Documentation](PROJECT_README.md)** | **[Setup Instructions](SETUP_COMPLETE.md)**

---

## 🚀 Features

### Core ML Features

- ✅ High-Quality Text-to-Speech generation
- ✅ Zero-Shot Voice Cloning from audio samples
- ✅ Multi-Device Support (CUDA/MPS/CPU)
- ✅ Professional audio quality
- ✅ Advanced neural architectures

### Web Application Features

- ✅ Beautiful modern UI with glassmorphism design
- ✅ Text-to-speech with default voice
- ✅ Voice cloning with audio upload
- ✅ Direct voice recording in browser
- ✅ Real-time audio playback
- ✅ Download generated audio
- ✅ Fully responsive design
- ✅ REST API with automatic documentation
- ✅ Docker deployment ready

---

## 🎯 Quick Start

### Option 1: Web Application (Recommended)

```bash
# 1. Install backend dependencies
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..
pip install -r requirements.txt

# 2. Install frontend dependencies
cd frontend
npm install
cd ..

# 3. Start everything
./start_all.sh

# 4. Open browser
open http://localhost:3000
```

### Option 2: Python Script (Original Method)

```python
import torchaudio as ta
from swarlekha_model.tts import SwarlekhaTTS

# Initialize model
model = SwarlekhaTTS.from_pretrained(device="cuda")  # or "cpu", "mps"

# Generate with default voice
text = "Hello, this is a test"
wav = model.generate(text)
ta.save("output.wav", wav, model.sr)

# Generate with voice cloning
wav = model.generate(text, audio_prompt_path="reference.wav")
ta.save("output_cloned.wav", wav, model.sr)
```

---

## 📱 Web Application

### URLs

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Features

#### 🎨 Modern UI

- Glassmorphism design with frosted glass effects
- Smooth animations powered by Framer Motion
- Beautiful gradient backgrounds
- Fully responsive (mobile, tablet, desktop)

#### 🎤 Voice Generation

- Enter text (up to 5000 characters)
- Choose default voice or clone from audio
- Upload reference audio or record directly
- Real-time generation with progress indicators

#### 📦 Audio Management

- Automatic organization by voice name
- Playback controls
- Download functionality
- Demo section with examples

### API Endpoints

```bash
# Health check
GET /api/health

# Generate voice
POST /api/generate
  - text: string (required)
  - reference_audio: file (optional)
  - voice_name: string (optional)

# List voices
GET /api/voices
```

---

## 📁 Project Structure

```
swarlekha/
├── backend/                    # FastAPI REST API
│   ├── main.py                # API server
│   ├── models.py              # Data models
│   └── requirements.txt       # Backend dependencies
│
├── frontend/                   # React TypeScript UI
│   ├── src/
│   │   ├── components/        # UI components
│   │   ├── services/          # API client
│   │   └── App.tsx            # Main app
│   ├── public/                # Static assets
│   └── package.json           # Frontend dependencies
│
├── swarlekha_model/           # ML model code
│   ├── tts.py                 # TTS interface
│   ├── vc.py                  # Voice conversion
│   ├── models/                # Neural architectures
│   └── weights/               # Model weights
│
├── examples/
│   ├── input/                 # Reference audio
│   └── output/                # Generated outputs
│
├── inference_tts.py           # Python inference script
├── inference_vc.py            # Voice conversion script
├── start_all.sh              # Start both services
├── docker-compose.yml        # Docker orchestration
│
└── Documentation
    ├── QUICKSTART.md          # Quick start guide
    ├── PROJECT_README.md      # Full documentation
    ├── SETUP_COMPLETE.md      # Setup instructions
    └── PROJECT_SUMMARY.md     # Complete summary
```

---

## 🛠 Installation

### Prerequisites

- Python 3.8+
- Node.js 18+
- CUDA-compatible GPU (optional, for acceleration)
- 8GB+ RAM

### Full Installation

```bash
# Clone repository
git clone https://github.com/bishal2059/swarlekha.git
cd swarlekha

# Install main model dependencies
pip install -r requirements.txt

# Install backend dependencies
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..

# Install frontend dependencies
cd frontend
npm install
cd ..
```

---

## 🚀 Usage

### Web Interface

1. **Start the application:**

   ```bash
   ./start_all.sh
   ```

2. **Open browser:**
   - Navigate to http://localhost:3000
   - Enter your text
   - Choose voice type (default or cloned)
   - Generate and download!

### Python API

```python
import torchaudio as ta
from swarlekha_model.tts import SwarlekhaTTS

# Initialize
model = SwarlekhaTTS.from_pretrained(device="cuda")

# Generate speech
text = "Your text here"
wav = model.generate(text)
ta.save("output.wav", wav, model.sr)

# With voice cloning
wav = model.generate(text, audio_prompt_path="reference.wav")
ta.save("cloned.wav", wav, model.sr)
```

### REST API

```bash
# Generate with default voice
curl -X POST "http://localhost:8000/api/generate" \
  -F "text=Hello world" \
  -F "voice_name=test" \
  --output output.wav

# Generate with voice cloning
curl -X POST "http://localhost:8000/api/generate" \
  -F "text=Hello world" \
  -F "reference_audio=@reference.wav" \
  -F "voice_name=custom" \
  --output output.wav
```

---

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# Stop services
docker-compose down
```

---

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 3 steps
- **[PROJECT_README.md](PROJECT_README.md)** - Complete documentation
- **[SETUP_COMPLETE.md](SETUP_COMPLETE.md)** - Detailed setup guide
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Project overview
- **[VISUAL_GUIDE.txt](VISUAL_GUIDE.txt)** - Visual project guide
- **[backend/README.md](backend/README.md)** - Backend documentation
- **[frontend/README.md](frontend/README.md)** - Frontend documentation

---

## 🎨 Technology Stack

### Backend

- FastAPI - Modern Python web framework
- Uvicorn - ASGI server
- PyTorch - Deep learning framework
- Torchaudio - Audio processing

### Frontend

- React 18 - UI library
- TypeScript - Type safety
- Vite - Build tool
- Tailwind CSS - Styling
- Framer Motion - Animations
- Axios - HTTP client

### ML Model

- Flow Matching - Speech synthesis
- Transformers - Text processing
- Voice Encoder - Voice embedding
- HiFi-GAN - Audio generation

---

## 🎯 Use Cases

- **Content Creation**: Generate voiceovers for videos
- **Accessibility**: Text-to-speech for visually impaired
- **Language Learning**: Pronunciation examples
- **Gaming**: Dynamic NPC voices
- **Audiobooks**: Automated narration
- **Virtual Assistants**: Custom voice personalities
- **Voice Acting**: Quick prototyping
- **Podcasting**: Intro/outro generation

---

## 📊 Model Architecture

```
Text Input
    ↓
Text Tokenizer → Text Encoder (T3)
    ↓
Semantic Tokens → Speech Tokenizer (S3Tokenizer)
    ↓
Acoustic Features → Flow Matching Model
    ↓
Mel Spectrogram → HiFi-GAN Vocoder
    ↓
Audio Output (WAV)

[Optional: Voice Encoder for cloning]
Reference Audio → Voice Embeddings → Conditioning
```

---

## 🔧 Configuration

### Backend Settings

Edit `backend/main.py`:

```python
# Change host/port
uvicorn.run(app, host="0.0.0.0", port=8000)

# CORS settings
allow_origins=["*"]  # Configure for production
```

### Frontend Settings

Edit `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000
```

### Model Settings

```python
# Device selection
device = "cuda"  # or "mps", "cpu"

# Generation parameters
model.generate(
    text="Your text",
    audio_prompt_path="reference.wav",  # Optional
    temperature=0.7,  # Adjust creativity
    max_length=1000   # Max output length
)
```

---

## 🐛 Troubleshooting

### Backend Issues

```bash
# Check if port 8000 is available
lsof -i :8000

# Verify Python dependencies
pip list | grep -E "fastapi|torch|torchaudio"

# Check model weights
ls -lh swarlekha_model/weights/
```

### Frontend Issues

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install

# Check Node version (should be 18+)
node --version
```

### Model Issues

```bash
# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Verify model files
python -c "from swarlekha_model.tts import SwarlekhaTTS"
```

---

## 📝 Examples

### Example 1: Basic Generation

```python
from swarlekha_model.tts import SwarlekhaTTS
import torchaudio as ta

model = SwarlekhaTTS.from_pretrained()
text = "Welcome to Swarlekha TTS"
wav = model.generate(text)
ta.save("welcome.wav", wav, model.sr)
```

### Example 2: Voice Cloning

```python
text = "This is a cloned voice"
reference = "examples/input/indra.wav"
wav = model.generate(text, audio_prompt_path=reference)
ta.save("cloned.wav", wav, model.sr)
```

### Example 3: Batch Processing

```python
texts = [
    "First sentence",
    "Second sentence",
    "Third sentence"
]

for i, text in enumerate(texts):
    wav = model.generate(text)
    ta.save(f"output_{i}.wav", wav, model.sr)
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

---

## 🙏 Acknowledgments

- Original Swarlekha model by [@bishal2059](https://github.com/bishal2059)
- FastAPI framework
- React and Vite communities
- PyTorch team
- All contributors and users

---

## 📧 Support

- **Documentation**: Check the docs folder
- **Issues**: Open a GitHub issue
- **API Docs**: http://localhost:8000/docs (when running)

---

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Made with ❤️ using Swarlekha TTS**

🎤 **Happy Voice Generation!** ✨
