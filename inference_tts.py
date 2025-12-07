import torchaudio as ta
import torch
from swarlekha_model.tts import SwarlekhaTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = SwarlekhaTTS.from_pretrained(device=device)

text = "The only way to do great work is to love what you do. If you haven't found it yet, keep looking. Don't settle. As with all matters of the heart, you'll know when you find it."
wav = model.generate(text) #synthesize with default voice
ta.save("examples/output/indra/test-1.wav", wav, model.sr)

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "examples/input/indra.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH) #clone the voice from this audio
ta.save("examples/output/indra/test-2.wav", wav, model.sr)