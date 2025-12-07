# Demo Audio Files

This directory contains demo audio files for the Swarlekha TTS application.

## Files

- `default_voice.wav` - Example of default voice generation
- `cloned_voice_1.wav` - Example of voice cloning (sample 1)
- `cloned_voice_2.wav` - Example of voice cloning (sample 2)

## Replacement Instructions

Replace these placeholder files with actual generated audio from your Swarlekha TTS model:

1. Generate audio using the inference_tts.py script
2. Copy the generated files to this directory
3. Rename them to match the expected filenames above

## Sample Text

All demo files should use the same text for consistency:

```
The only way to do great work is to love what you do. If you haven't found it yet, keep looking. Don't settle. As with all matters of the heart, you'll know when you find it.
```

## Format Specifications

- Format: WAV
- Sample Rate: As configured in your model (typically 22050 or 24000 Hz)
- Channels: Mono or Stereo as per model output
- Bit Depth: 16-bit or higher
