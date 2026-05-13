import argparse
import torch
import torchaudio
from pathlib import Path
from swarlekha_model.tts_nepali import SwarlekhaNepaliTTS
from safetensors.torch import load_file

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = SwarlekhaNepaliTTS.from_pretrained(device=device)

# nepali_model_checkpoint = "swarlekha_model/weights/t3_nepali_checkpoint.pt"

# resume_state = torch.load(
#     nepali_model_checkpoint,
#     map_location="cpu",
#     weights_only=True
# )

# cleaned_state = {
#     k.replace("patched_model.", "").replace("model.", ""): v
#     for k, v in resume_state.items()
# }

# missing, unexpected = model.t3.load_state_dict(
#     cleaned_state,
#     strict=False
# )

# model.t3.to(device).eval()
# model.s3gen.tokenizer.to(device)
# model.ve.to(device)



text = "नमस्ते! आज हामी एकदमै सजिलो र सबैको मनपर्ने नेपाली खाना—दाल, भात र आलु-तमा (वा आलु-कोपी) को तरकारी—बनाउन सिक्नेछौँ।"
AUDIO_PROMPT_PATH = "examples/input/indra.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH) #clone the voice from this audio
import soundfile as sf
import numpy as np
sf.write("examples/output/indra/test-nepali.wav", wav.squeeze().cpu().numpy(), model.sr)