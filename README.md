# Swarlekha: Advanced Text-to-Speech and Voice Conversion System

**Swarlekha** is a state-of-the-art AI-powered text-to-speech (TTS) and voice conversion (VC) system that delivers high-quality, natural-sounding speech synthesis. Built on advanced neural architectures, Swarlekha supports zero-shot voice cloning, enabling users to generate speech in any voice with just a short audio sample.

## 🚀 Features

- **High-Quality Text-to-Speech**: Generate natural, expressive speech from text
- **Zero-Shot Voice Cloning**: Clone any voice with just a few seconds of audio
- **Voice Conversion**: Transform the voice characteristics of existing audio
- **Multi-Device Support**: Automatic detection and optimization for CUDA, MPS, and CPU
- **Easy-to-Use API**: Simple Python interface for quick integration
- **Professional Audio Quality**: Built on advanced neural architectures including flow matching and transformer models

## 📁 Project Structure

```
swarlekha/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── inference_tts.py             # Text-to-speech inference script
├── inference_vc.py              # Voice conversion inference script
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
- Python 3.9+
- CUDA-compatible GPU (recommended) or CPU
- At least 8GB RAM (16GB+ recommended for GPU usage)

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bishal2059/swarlekha.git
   cd swarlekha
   ```

2. **Create and activate conda environment:**
   ```bash
   conda create -n swarlekha python=3.9 -y
   conda activate swarlekha
   ```

3. **Install dependencies:**
   ```bash
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
- Additional utilities: `numpy`, `s3tokenizer`, `resemble-perth`, `conformer`

## 🎯 Usage

### Text-to-Speech (TTS)

Generate speech from text with default voice or clone a specific voice:

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

### Voice Conversion (VC)

Transform voice characteristics of existing audio:

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
- [ ] **Web Interface**: User-friendly web application for non-technical users
- [ ] **API Server**: RESTful API for integration into applications


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



