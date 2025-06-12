import os
import soundfile as sf
import numpy as np
import librosa
import matplotlib.pyplot as plt

# ─────── CONFIG ────────────────────────────────────────────────────────────
BASE_PATH       = r"C:\Users\chitt\Downloads\ICSV\data"
OUTPUT_DIR      = r"C:\Users\chitt\Downloads\ICSV\comparison_plots"
DRONE_TYPES     = ["A", "B", "C"]
MOVEMENTS       = ["Front", "Back", "Left", "Right", "Clockwise", "CounterClockwise"]

# Spectrogram parameters
N_FFT           = 2048     # “Analysis window length”
HOP_LENGTH      = 128     # typically N_FFT/4
DB_RANGE        = 95      # dynamic range in dB (0 down to -DB_RANGE)
GAMMA           = 0.6      # gamma correction for better contrast
# ──────────────────────────────────────────────────────────────────────────

def load_mono(path):
    y, sr = sf.read(path)
    if y.ndim == 2:
        y = y.mean(axis=1)
    return y, sr

def find_wav(base, subset, drone, move, flag):
    folder = os.path.join(base, subset)
    if not os.path.isdir(folder):
        return None
    for fn in os.listdir(folder):
        if fn.lower().endswith(".wav") and f"_{drone}_{move}_{flag}" in fn:
            return os.path.join(folder, fn)
    return None

def plot_comparison(y_norm, y_anom, sr, out_png):
    # 1) Spectrograms → dB
    S_n  = np.abs(librosa.stft(y_norm, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann"))
    S_a  = np.abs(librosa.stft(y_anom, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann"))
    db_n = librosa.amplitude_to_db(S_n, ref=np.max)
    db_a = librosa.amplitude_to_db(S_a, ref=np.max)
    db_n = np.clip(db_n, -DB_RANGE, 0)
    db_a = np.clip(db_a, -DB_RANGE, 0)

    # ★ Apply gamma correction
    db_n = ((db_n + DB_RANGE) / DB_RANGE) ** GAMMA
    db_a = ((db_a + DB_RANGE) / DB_RANGE) ** GAMMA

    dur = len(y_norm)/sr
    fmax = sr/2/1000  # kHz

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    (ax_spec_n, ax_spec_a), (ax_wav_n, ax_wav_a) = axes

    # ── Normal Spectrogram ────────────────────────────────────────────────
    ax_spec_n.imshow(
        db_n, origin="lower", aspect="auto", cmap="gray_r",
        extent=[0, dur, 0, fmax]
    )
    ax_spec_n.set_title("Normal  – Spectrogram")
    ax_spec_n.set_ylabel("kHz")
    ax_spec_n.set_xticks([])

    # ── Anomaly Spectrogram ───────────────────────────────────────────────
    ax_spec_a.imshow(
        db_a, origin="lower", aspect="auto", cmap="gray_r",
        extent=[0, dur, 0, fmax]
    )
    ax_spec_a.set_title("Anomaly – Spectrogram")
    ax_spec_a.set_yticks([])

    # ── Normal Waveform ───────────────────────────────────────────────────
    times = np.linspace(0, dur, len(y_norm))
    ax_wav_n.plot(times, y_norm, color="black", linewidth=0.5)
    ax_wav_n.set_title("Normal  – Waveform")
    ax_wav_n.set_ylabel("Amplitude")
    ax_wav_n.set_xlim(0, dur)

    # ── Anomaly Waveform ──────────────────────────────────────────────────
    times2 = np.linspace(0, len(y_anom)/sr, len(y_anom))
    ax_wav_a.plot(times2, y_anom, color="black", linewidth=0.5)
    ax_wav_a.set_title("Anomaly – Waveform")
    ax_wav_a.set_yticks([])
    ax_wav_a.set_xlim(0, dur)

    # common x-axis label
    for ax in (ax_wav_n, ax_wav_a):
        ax.set_xlabel("Time (s)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close(fig)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for drone in DRONE_TYPES:
        for mv in MOVEMENTS:
            wav_n = find_wav(BASE_PATH, "train", drone, mv, "normal")
            wav_a = find_wav(BASE_PATH, "eval",  drone, mv, "anomaly")
            if not wav_n or not wav_a:
                print(f"⚠️  Missing: {drone}_{mv}")
                continue

            # ★ Add this:
            print(f"🔹 {drone}_{mv}:")
            print(f"    Normal file : {os.path.basename(wav_n)}")
            print(f"    Anomaly file: {os.path.basename(wav_a)}")

            y_n, sr = load_mono(wav_n)
            y_n = librosa.effects.preemphasis(y_n, coef=0.97)

            y_a, _  = load_mono(wav_a)
            y_a = librosa.effects.preemphasis(y_a, coef=0.97)
            out_png = os.path.join(OUTPUT_DIR, f"{drone}_{mv}.png")
            plot_comparison(y_n, y_a, sr, out_png)

    print("✅ Done — plots stored in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
