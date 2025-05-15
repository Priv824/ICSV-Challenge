import os
import soundfile as sf
import numpy as np
import librosa
from PyEMD import EMD

# ─────── CONFIG ───────
BASE_PATH = os.path.expanduser("~/ICSV-Surya/ICSV-Challenge")
DRONE_TYPES = ["A", "B", "C"]
MOVEMENTS = ["Front", "Back", "Left", "Right", "Clockwise", "CounterClockwise"]
SAMPLE_RATE = 16000

# ─────── HELPERS ───────

def find_wav(base, subset, drone, move, flag):
    folder = os.path.join(base, subset)
    if not os.path.isdir(folder):
        return None
    for fn in os.listdir(folder):
        if fn.lower().endswith(".wav") and f"_{drone}_{move}_{flag}" in fn:
            return os.path.join(folder, fn)
    return None

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

# ─────── PROCESSING LOOP ───────

for drone in DRONE_TYPES:
    for mv in MOVEMENTS:
        wav_path = find_wav(BASE_PATH, "train", drone, mv, "normal")
        if not wav_path:
            print(f"⚠️  Missing: {drone}_{mv}")
            continue

        print(f"🔧 Processing: {os.path.basename(wav_path)}")

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

        # Overwrite original file with reconstructed version
        sf.write(wav_path, y_out, SAMPLE_RATE, subtype='PCM_16')
        print(f"✅ Overwritten: {wav_path}")

print("\n🎉 Done! All train audio files processed and replaced with low-frequency reconstructions.")
