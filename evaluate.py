import argparse
import os
import sys
import numpy as np
import librosa
import torch
import soundfile as sf

# Patch pkg_resources for webrtcvad on Python 3.14+
import types
if 'pkg_resources' not in sys.modules:
    mock_pkg_resources = types.ModuleType('pkg_resources')
    class MockDistribution:
        def __init__(self, version):
            self.version = version
    def mock_get_distribution(name):
        return MockDistribution('2.0.10')
    mock_pkg_resources.get_distribution = mock_get_distribution
    sys.modules['pkg_resources'] = mock_pkg_resources

# Evaluation metrics imports
import traceback
try:
    from pesq import pesq
    import pystoi
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    from scipy.stats import pearsonr
    import whisper
    from resemblyzer import VoiceEncoder, preprocess_wav
except ImportError as e:
    print(f"Missing required dependency for evaluation: {e}")
    traceback.print_exc()
    print("\nPlease install the required packages:")
    print("pip install pesq pystoi fastdtw scipy openai-whisper resemblyzer")
    exit(1)


def calculate_mcd(ref_wav, gen_wav, sr=16000):
    """Calculate Mel-Cepstral Distortion (MCD) between two audio signals."""
    # Extract MFCCs
    mfcc_ref = librosa.feature.mfcc(y=ref_wav, sr=sr, n_mfcc=13)
    mfcc_gen = librosa.feature.mfcc(y=gen_wav, sr=sr, n_mfcc=13)
    
    # DTW alignment
    distance, path = fastdtw(mfcc_ref.T, mfcc_gen.T, dist=euclidean)
    
    # Calculate MCD
    mcd = (10.0 / np.log(10)) * np.sqrt(2) * (distance / len(path))
    return mcd

def calculate_pesq(ref_wav, gen_wav, sr=16000):
    """Calculate PESQ score (target > 3.5)."""
    # PESQ only supports 8000 or 16000 Hz
    if sr != 16000:
        ref_wav = librosa.resample(ref_wav, orig_sr=sr, target_sr=16000)
        gen_wav = librosa.resample(gen_wav, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    try:
        score = pesq(sr, ref_wav, gen_wav, 'wb')
        return score
    except Exception as e:
        print(f"PESQ calculation failed: {e}")
        return None

def calculate_stoi(ref_wav, gen_wav, sr=16000):
    """Calculate STOI score (target > 0.90)."""
    # Ensure same length for STOI
    min_len = min(len(ref_wav), len(gen_wav))
    score = pystoi.stoi(ref_wav[:min_len], gen_wav[:min_len], sr, extended=False)
    return score

def calculate_wer_cer(gen_wav_path, reference_text, language=None, model_size="base"):
    """Calculate Word Error Rate (WER) and Character Error Rate (CER) using Whisper."""
    print(f"Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)
    
    # Transcribe with optional language hint and initial prompt to force Devanagari script
    options = {}
    if language:
        options["language"] = language
        if language in ["ne", "nepali", "hi", "hindi"]:
            options["initial_prompt"] = "यो नेपाली भाषाको वाक्य हो। कृपया देवनागरी लिपिमा लेख्नुहोस्।"
        
    result = model.transcribe(gen_wav_path, **options)
    transcription = result["text"].strip()
    
    # Calculate WER and CER using jiwer if installed
    try:
        from jiwer import wer, cer
        w_score = wer(reference_text.lower(), transcription.lower())
        c_score = cer(reference_text.lower(), transcription.lower())
    except ImportError:
        # Basic Levenshtein distance for WER and CER fallback
        def levenshtein(ref, hyp):
            d = np.zeros((len(ref) + 1, len(hyp) + 1))
            for i in range(len(ref) + 1): d[i][0] = i
            for j in range(len(hyp) + 1): d[0][j] = j
            for i in range(1, len(ref) + 1):
                for j in range(1, len(hyp) + 1):
                    if ref[i-1] == hyp[j-1]:
                        d[i][j] = d[i-1][j-1]
                    else:
                        d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + 1)
            return d[len(ref)][len(hyp)]
            
        ref_words = reference_text.lower().split()
        hyp_words = transcription.lower().split()
        w_score = levenshtein(ref_words, hyp_words) / max(len(ref_words), 1)
        
        # CER
        c_score = levenshtein(list(reference_text.lower()), list(transcription.lower())) / max(len(reference_text), 1)
        
    return w_score, c_score, transcription

def calculate_secs(ref_wav_path, gen_wav_path):
    """Calculate Speaker Embedding Cosine Similarity (SECS) (target > 0.80)."""
    encoder = VoiceEncoder()
    
    # Preprocess and embed
    ref_wav = preprocess_wav(ref_wav_path)
    gen_wav = preprocess_wav(gen_wav_path)
    
    embed_ref = encoder.embed_utterance(ref_wav)
    embed_gen = encoder.embed_utterance(gen_wav)
    
    # Cosine similarity
    secs = np.dot(embed_ref, embed_gen) / (np.linalg.norm(embed_ref) * np.linalg.norm(embed_gen))
    return secs

def calculate_f0_metrics(ref_wav, gen_wav, sr=16000):
    """Calculate F0 RMSE and Pearson F0 correlation."""
    # Extract F0 using librosa.pyin
    f0_ref, _, _ = librosa.pyin(ref_wav, fmin=50, fmax=500, sr=sr)
    f0_gen, _, _ = librosa.pyin(gen_wav, fmin=50, fmax=500, sr=sr)
    
    # Replace NaNs with 0 (unvoiced)
    f0_ref = np.nan_to_num(f0_ref)
    f0_gen = np.nan_to_num(f0_gen)
    
    # Align using DTW
    distance, path = fastdtw(f0_ref.reshape(-1, 1), f0_gen.reshape(-1, 1), dist=euclidean)
    
    aligned_ref = np.array([f0_ref[p[0]] for p in path])
    aligned_gen = np.array([f0_gen[p[1]] for p in path])
    
    # Only calculate on voiced frames in both
    voiced_indices = (aligned_ref > 0) & (aligned_gen > 0)
    
    if np.sum(voiced_indices) < 2:
        return None, None
        
    aligned_ref_voiced = aligned_ref[voiced_indices]
    aligned_gen_voiced = aligned_gen[voiced_indices]
    
    rmse = np.sqrt(np.mean((aligned_ref_voiced - aligned_gen_voiced) ** 2))
    corr, _ = pearsonr(aligned_ref_voiced, aligned_gen_voiced)
    
    return rmse, corr

def calculate_utmos(gen_wav_path):
    """Calculate UTMOS score using torch.hub (target > 4.0)."""
    try:
        predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        wav, sr = librosa.load(gen_wav_path, sr=16000)
        score = predictor(torch.from_numpy(wav).unsqueeze(0), sr)
        return score.item()
    except Exception as e:
        print(f"UTMOS calculation failed: {e}")
        return None

def evaluate(ref_audio_path, gen_audio_path, text=None, language=None, model_size="base"):
    """Run all evaluation metrics."""
    print(f"Evaluating:\nReference: {ref_audio_path}\nGenerated: {gen_audio_path}")
    if language:
        print(f"Language hint: {language}")
    
    if not os.path.exists(ref_audio_path) or not os.path.exists(gen_audio_path):
        print("Error: Audio files not found.")
        return
        
    # Load audio
    ref_wav, ref_sr = librosa.load(ref_audio_path, sr=16000)
    gen_wav, gen_sr = librosa.load(gen_audio_path, sr=16000)
    
    results = {}
    
    # 1. MCD
    print("\nCalculating MCD (Target < 6dB)...")
    mcd = calculate_mcd(ref_wav, gen_wav)
    results['MCD'] = mcd
    print(f"MCD: {mcd:.2f} dB")
    
    # 2. PESQ
    print("Calculating PESQ (Target > 3.5)...")
    pesq_score = calculate_pesq(ref_wav, gen_wav)
    results['PESQ'] = pesq_score
    if pesq_score:
        print(f"PESQ: {pesq_score:.2f}")
        
    # 3. STOI
    print("Calculating STOI (Target > 0.90)...")
    stoi = calculate_stoi(ref_wav, gen_wav)
    results['STOI'] = stoi
    print(f"STOI: {stoi:.2f}")
    
    # 4. SECS (Speaker Similarity)
    print("Calculating SECS (Target > 0.80)...")
    secs = calculate_secs(ref_audio_path, gen_audio_path)
    results['SECS'] = secs
    print(f"SECS: {secs:.2f}")
    
    # 5. F0 Prosody Metrics
    print("Calculating F0 RMSE and Correlation...")
    rmse, corr = calculate_f0_metrics(ref_wav, gen_wav)
    if rmse is not None and corr is not None:
        results['F0_RMSE'] = rmse
        results['F0_Corr'] = corr
        print(f"F0 RMSE: {rmse:.2f} Hz (Target < 30Hz)")
        print(f"F0 Correlation: {corr:.2f} (Target > 0.85)")
    else:
        print("F0 Metrics: Failed to calculate (not enough voiced frames)")
        
    # 6. UTMOS
    print("Calculating UTMOS using SpeechMOS (Target > 4.0)...")
    utmos = calculate_utmos(gen_audio_path)
    if utmos is not None:
        results['UTMOS'] = utmos
        print(f"UTMOS: {utmos:.2f}")
    
    # 7. WER & CER
    if text:
        print("\nCalculating WER and CER...")
        wer_score, cer_score, transcription = calculate_wer_cer(gen_audio_path, text, language, model_size)
        results['WER'] = wer_score * 100
        results['CER'] = cer_score * 100
        print(f"WER: {wer_score*100:.2f}% (Target < 5%)")
        print(f"CER: {cer_score*100:.2f}% (Target < 5% for Nepali)")
        print(f"Transcription: {transcription}")
        print(f"Reference Text: {text}")
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Swarlekha TTS output")
    parser.add_argument("--ref", type=str, required=True, help="Path to reference audio")
    parser.add_argument("--gen", type=str, required=True, help="Path to generated audio")
    parser.add_argument("--text", type=str, default=None, help="Ground truth text")
    parser.add_argument("--lang", type=str, default=None, help="Language code for ASR (e.g., 'ne' for Nepali, 'en' for English)")
    parser.add_argument("--model", type=str, default="base", help="Whisper model size (base, small, medium, large)")
    
    args = parser.parse_args()
    evaluate(args.ref, args.gen, args.text, args.lang, args.model)
