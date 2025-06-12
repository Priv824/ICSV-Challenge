import os
import soundfile as sf
import numpy as np
import librosa
import matplotlib.pyplot as plt
from PyEMD import EMD

# ─────── CONFIG ────────────────────────────────────────────────────────────
BASE_PATH = r"C:\Users\chitt\Downloads\ICSV\data"
OUTPUT_DIR = r"C:\Users\chitt\Downloads\ICSV\emd_plots"
DRONE_TYPES = ["A", "B", "C"]
MOVEMENTS = ["Front", "Back", "Left", "Right", "Clockwise", "CounterClockwise"]

# EMD parameters
MAX_IMFS = 16  # Maximum number of IMFs to display
PLOT_HEIGHT_PER_IMF = 0.8  # Height in inches for each IMF subplot
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

def plot_emd_comparison(y_norm, y_anom, sr, drone_type, movement, out_png):
    # Initialize EMD
    emd = EMD()
    
    # Compute IMFs
    imfs_n = emd(y_norm)
    imfs_a = emd(y_anom)

    print(imfs_n.shape, imfs_a.shape)
    
    # Determine actual number of IMFs to plot
    num_imfs = min(max(imfs_n.shape[0], imfs_a.shape[0]), MAX_IMFS)
    
    # Create figure with appropriate height
    fig_height = num_imfs * PLOT_HEIGHT_PER_IMF * 2  # Double height for both signals
    fig, axes = plt.subplots(num_imfs, 2, figsize=(12, fig_height), sharex=True)
    
    if num_imfs == 1:
        axes = axes.reshape(1, -1)  # Ensure axes is 2D even for single IMF
    
    time = np.arange(len(y_norm)) / sr
    
    # Plot IMFs for normal signal (left column)
    for i in range(num_imfs):
        if i < imfs_n.shape[0]:
            axes[i, 0].plot(time, imfs_n[i], 'b', linewidth=0.8)

        axes[i, 0].set_ylabel(f"IMF {i+1}", rotation=0, ha='right', va='center')
        axes[i, 0].grid(True, alpha=0.3)
        if i != num_imfs-1:
            axes[i, 0].set_xticks([])
    
    # Plot IMFs for anomaly signal (right column)
    for i in range(num_imfs):
        if i < imfs_a.shape[0]:
            axes[i, 1].plot(time, imfs_a[i], 'r', linewidth=0.8)
        axes[i, 1].grid(True, alpha=0.3)
        if i != num_imfs-1:
            axes[i, 1].set_xticks([])
    
    # Set titles and labels
    axes[0, 0].set_title(f"Normal - {drone_type} {movement}", pad=20)
    axes[0, 1].set_title(f"Anomaly - {drone_type} {movement}", pad=20)
    
    for i in range(num_imfs):
        axes[i, 0].set_xlim(0, time[-1])
        axes[i, 1].set_xlim(0, time[-1])
    
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for drone in DRONE_TYPES:
        for mv in MOVEMENTS:
            wav_n = find_wav(BASE_PATH, "train", drone, mv, "normal")
            wav_a = find_wav(BASE_PATH, "eval", drone, mv, "anomaly")
            if not wav_n or not wav_a:
                print(f"⚠️  Missing: {drone}_{mv}")
                continue

            print(f"🔹 Processing {drone}_{mv}:")
            print(f"    Normal file: {os.path.basename(wav_n)}")
            print(f"    Anomaly file: {os.path.basename(wav_a)}")

            # Load and preprocess audio
            y_n, sr = load_mono(wav_n)
            y_n = librosa.effects.preemphasis(y_n, coef=0.97)
            
            y_a, _ = load_mono(wav_a)
            y_a = librosa.effects.preemphasis(y_a, coef=0.97)
            
            # Make sure both signals have the same length (truncate to shorter one)
            min_len = min(len(y_n), len(y_a))
            y_n = y_n[:min_len]
            y_a = y_a[:min_len]
            
            # Generate output path and plot
            out_png = os.path.join(OUTPUT_DIR, f"EMD_{drone}_{mv}.png")
            plot_emd_comparison(y_n, y_a, sr, drone, mv, out_png)

    print("✅ EMD analysis complete. Plots stored in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()