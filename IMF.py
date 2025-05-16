import os
import soundfile as sf
import numpy as np
import librosa
from PyEMD import EMD

# ─────── CONFIG ───────
BASE_PATH = r"C:\Users\chitt\Downloads\ICSV\Git\ICSV-Challenge\data"
TRAIN_FOLDER = "train"  # Change this if your train data is in a different folder
SAMPLE_RATE = 16000

# ─────── HELPERS ───────

def load_mono(path):
    y, sr = sf.read(path)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    return y

def to_pcm16(signal):
    signal = signal / np.max(np.abs(signal))  # Normalize to [-1, 1]
    return np.clip(signal, -1.0, 1.0)

# ─────── MAIN PROCESSING LOOP ───────

train_path = os.path.join(BASE_PATH, TRAIN_FOLDER)

# Walk through all files in the train directory
for root, dirs, files in os.walk(train_path):
    for file in files:
        if file.lower().endswith(".wav"):
            wav_path = os.path.join(root, file)
            print(f"🔧 Processing: {os.path.basename(wav_path)}")

            try:
                # Load and preprocess
                y = load_mono(wav_path)
                y = librosa.effects.preemphasis(y, coef=0.97)

                # Apply EMD
                emd = EMD()
                imfs = emd(y)

                if imfs.shape[0] < 9:
                    print(f"⚠️  Not enough IMFs to reconstruct low-frequency part: {wav_path}")
                    continue

                # Reconstruct low-frequency part (IMFs 8+)
                y_low = np.sum(imfs[8:, :], axis=0)
                y_out = to_pcm16(y_low)

                # Overwrite original file
                sf.write(wav_path, y_out, SAMPLE_RATE, subtype='PCM_16')
                print(f"✅ Overwritten: {wav_path}")

            except Exception as e:
                print(f"❌ Error processing {wav_path}: {e}")

print("\n🎉 Done! All train audio files processed and replaced with low-frequency reconstructions.")