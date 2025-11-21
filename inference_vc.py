import torch
import torchaudio as ta

from swarlekha_model.vc import SwarlekhaVC

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

AUDIO_PATH = "examples/input/mine.wav" # the output follows the tone and text of this audio
TARGET_VOICE_PATH = "examples/input/TARGET_VOICE.wav" # the output mimics the voice in this audio

model = SwarlekhaVC.from_pretrained(device)
wav = model.generate(
    audio=AUDIO_PATH,
    target_voice_path=TARGET_VOICE_PATH,
)
ta.save("examples/output/test-vc.wav", wav, model.sr)
