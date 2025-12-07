# Quick Start Guide - Swarlekha TTS

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..
pip install -r requirements.txt  # Install main project dependencies
cd ..

# Frontend
cd frontend
npm install
cd ..
```

### Step 2: Start the Application

**Option A: Automatic (Recommended)**

```bash
chmod +x start_all.sh
./start_all.sh
```

**Option B: Manual**

Terminal 1 (Backend):

```bash
cd backend
source venv/bin/activate
python main.py
```

Terminal 2 (Frontend):

```bash
cd frontend
npm run dev
```

### Step 3: Use the Application

1. Open browser: `http://localhost:3000`
2. Enter text in the text area
3. Choose voice type:
   - **Default Voice**: Click "Default Voice" button
   - **Clone Voice**: Click "Clone Voice", upload audio or record
4. Click "Generate Voice"
5. Play or download the generated audio

## 📍 Important URLs

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## 🎯 Quick Test

### Test Backend Only

```bash
cd backend
source venv/bin/activate
python main.py

# In another terminal:
curl http://localhost:8000/api/health
```

### Test Full Stack

1. Start both services (see Step 2)
2. Open http://localhost:3000
3. Enter text: "Hello, this is a test"
4. Click "Generate Voice"
5. Play the generated audio

## 🛑 Stop the Application

**If using automatic start:**

```bash
./stop_all.sh
```

**If using manual start:**
Press `Ctrl+C` in both terminal windows

## 📝 Example: Generate Voice with cURL

```bash
# Default voice
curl -X POST "http://localhost:8000/api/generate" \
  -F "text=Hello world" \
  -F "voice_name=test" \
  --output output.wav

# With voice cloning
curl -X POST "http://localhost:8000/api/generate" \
  -F "text=Hello world" \
  -F "reference_audio=@examples/input/indra.wav" \
  -F "voice_name=indra" \
  --output output.wav
```

## 🔧 Troubleshooting

**Backend won't start:**

```bash
# Check Python version
python --version  # Should be 3.8+

# Check if port 8000 is in use
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Try different port (edit backend/main.py)
uvicorn.run(app, host="0.0.0.0", port=8001)
```

**Frontend won't start:**

```bash
# Check Node version
node --version  # Should be 18+

# Clear cache
rm -rf node_modules package-lock.json
npm install

# Try different port
# Edit vite.config.ts, change port: 3000 to another port
```

**Cannot connect to API:**

- Check backend is running: `curl http://localhost:8000/api/health`
- Check `.env` file in frontend directory
- Verify VITE_API_URL is correct

**Model not loading:**

- Ensure weights are in `swarlekha_model/weights/`
- Check main requirements.txt is installed
- Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`

## 📂 Directory Structure Check

Before starting, ensure you have:

```
swarlekha/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── ...
├── frontend/
│   ├── src/
│   ├── package.json
│   └── ...
├── swarlekha_model/
│   ├── weights/
│   │   ├── s3gen.safetensors
│   │   ├── t3_cfg.safetensors
│   │   └── ...
│   └── ...
├── examples/
│   ├── input/
│   └── output/
└── requirements.txt
```

## 🎬 Next Steps

1. **Add Demo Audio**: Generate sample audio and place in `frontend/public/demo/`
2. **Customize UI**: Edit `frontend/tailwind.config.js` for colors
3. **Add Features**: Extend API endpoints in `backend/main.py`
4. **Deploy**: Follow deployment guide in PROJECT_README.md

## 📚 More Information

- Full documentation: `PROJECT_README.md`
- Backend details: `backend/README.md`
- Frontend details: `frontend/README.md`

## 💡 Tips

- Keep both backend and frontend running for full functionality
- Check logs if something goes wrong:
  - Backend: `tail -f backend.log`
  - Frontend: `tail -f frontend.log`
- Use API docs at `/docs` for API testing
- Browser console (F12) shows frontend errors

---

Need help? Check the main README or create an issue!
