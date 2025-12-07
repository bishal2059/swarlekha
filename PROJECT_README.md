# Swarlekha TTS - Full Stack Application

A complete full-stack application for Text-to-Speech generation with advanced voice cloning capabilities, powered by the Swarlekha model.

## 🚀 Features

### Backend

- **FastAPI REST API** with automatic documentation
- **Voice Generation** from text with default voice
- **Voice Cloning** from reference audio
- **Automatic Output Management** organized by voice name
- **CORS Support** for frontend integration
- **Device Auto-detection** (CUDA/MPS/CPU)

### Frontend

- **Modern React UI** with TypeScript
- **Beautiful Glassmorphism Design** with animations
- **Drag & Drop Audio Upload**
- **Voice Recording** directly in browser
- **Real-time Audio Playback**
- **Download Generated Audio**
- **Demo Section** with examples
- **Fully Responsive** mobile-friendly design

## 📋 Prerequisites

- Python 3.8+
- Node.js 18+
- npm or yarn
- CUDA (optional, for GPU acceleration)

## 🛠️ Installation

### 1. Clone and Setup Backend

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Also install the main project requirements
cd ..
pip install -r requirements.txt
```

### 2. Setup Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
```

## 🚀 Running the Application

### Method 1: Using Separate Terminals

**Terminal 1 - Backend:**

```bash
cd backend
source venv/bin/activate  # If not already activated
python main.py
```

Backend will run at: `http://localhost:8000`

**Terminal 2 - Frontend:**

```bash
cd frontend
npm run dev
```

Frontend will run at: `http://localhost:3000`

### Method 2: Using the Start Script

```bash
# Make the script executable
chmod +x start_all.sh

# Run both services
./start_all.sh
```

## 📁 Project Structure

```
swarlekha/
├── backend/                    # FastAPI backend
│   ├── main.py                # API server
│   ├── models.py              # Pydantic models
│   ├── requirements.txt       # Python dependencies
│   ├── start.sh              # Backend start script
│   └── README.md             # Backend documentation
│
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── services/         # API services
│   │   ├── App.tsx           # Main app
│   │   └── index.css         # Styles
│   ├── public/
│   │   └── demo/            # Demo audio files
│   ├── package.json         # Node dependencies
│   └── README.md            # Frontend documentation
│
├── swarlekha_model/           # ML model code
│   ├── tts.py                # TTS interface
│   ├── vc.py                 # Voice conversion
│   ├── models/               # Model implementations
│   └── weights/              # Model weights
│
├── examples/
│   ├── input/                # Reference audio files
│   └── output/               # Generated audio outputs
│       ├── ashish/
│       └── indra/
│
├── inference_tts.py          # Original inference script
├── inference_vc.py           # Voice conversion script
├── requirements.txt          # Main project dependencies
└── README.md                 # This file
```

## 🔧 Configuration

### Backend Configuration

Edit `backend/main.py` to configure:

- Host and port
- CORS origins
- Model settings

### Frontend Configuration

Edit `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000
```

## 📡 API Documentation

Once the backend is running, visit:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### API Endpoints

**Health Check**

```
GET /api/health
```

**Generate Voice**

```
POST /api/generate
Content-Type: multipart/form-data

Parameters:
- text: string (required)
- reference_audio: file (optional)
- voice_name: string (optional, default: "default")
```

**List Voices**

```
GET /api/voices
```

## 🎨 Frontend Features

### Voice Generation Interface

1. Enter text (up to 5000 characters)
2. Choose voice type:
   - **Default Voice**: Use built-in voice
   - **Clone Voice**: Upload reference audio or record
3. Generate and play audio
4. Download generated audio

### Audio Upload Options

- Drag & drop files
- Click to browse
- Record directly in browser
- Supported formats: WAV, MP3, M4A, OGG

### Demo Section

- Pre-configured examples
- Compare default vs cloned voices
- Sample audio playback

## 🎯 Usage Examples

### Generate with Default Voice

```bash
curl -X POST "http://localhost:8000/api/generate" \
  -F "text=Hello, this is a test" \
  -F "voice_name=test" \
  --output output.wav
```

### Generate with Voice Cloning

```bash
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

## 🚀 Deployment

### Backend Deployment

**Using Docker:**

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "backend/main.py"]
```

**Using Gunicorn:**

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.main:app
```

### Frontend Deployment

**Build for production:**

```bash
cd frontend
npm run build
```

Deploy the `frontend/dist/` directory to:

- Netlify
- Vercel
- AWS S3 + CloudFront
- Any static hosting service

## 🔍 Troubleshooting

### Backend Issues

**Model not loading:**

- Ensure weights are in `swarlekha_model/weights/`
- Check Python version compatibility
- Verify CUDA/PyTorch installation

**Port already in use:**

```bash
# Change port in backend/main.py
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Frontend Issues

**API connection failed:**

- Verify backend is running
- Check CORS configuration
- Update `VITE_API_URL` in `.env`

**Build errors:**

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

## 🎨 Customization

### Styling

- Edit `frontend/tailwind.config.js` for colors
- Modify `frontend/src/index.css` for global styles
- Update component styles in individual `.tsx` files

### Functionality

- Add new API endpoints in `backend/main.py`
- Create new components in `frontend/src/components/`
- Extend API services in `frontend/src/services/api.ts`

## 📝 Demo Data

Replace demo audio files in `frontend/public/demo/`:

1. Generate audio using the model
2. Copy files to the demo directory
3. Name them as specified in the README

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

Same as the main Swarlekha project

## 🙏 Acknowledgments

- Swarlekha TTS Model
- FastAPI
- React & Vite
- Tailwind CSS
- Framer Motion

## 📧 Support

For issues and questions:

- Check the documentation
- Review existing issues
- Create a new issue with details

---

**Made with ❤️ using Swarlekha TTS**
