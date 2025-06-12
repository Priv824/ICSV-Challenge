import os
import soundfile as sf
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Config (same as before)
BASE_PATH = r"C:\Users\chitt\Downloads\ICSV\data"
OUTPUT_DIR = r"C:\Users\chitt\Downloads\ICSV\tsne_time_series_plots"
DRONE_TYPES = ["A", "B", "C"]
MOVEMENTS = ["Front", "Back", "Left", "Right", "Clockwise", "CounterClockwise"]
N_FFT = 2048
HOP_LENGTH = 512

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

def extract_time_series_features(y, sr):
    """Extract all time points of spectral features"""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    times = librosa.times_like(S, sr=sr, hop_length=HOP_LENGTH)
    
    features = {
        'time': times,
        'flatness': librosa.feature.spectral_flatness(S=S).flatten()/np.mean(S),
        'rolloff': librosa.feature.spectral_rolloff(S=S, sr=sr).flatten()/np.mean(S),
        'bandwidth': librosa.feature.spectral_bandwidth(S=S, sr=sr).flatten()/np.mean(S),
        'centroid': librosa.feature.spectral_centroid(S=S, sr=sr).flatten()/np.mean(S)
    }
    return features

def plot_tsne_time_series(features_n, features_a, drone_type, movement, out_png):
    """Plot t-SNE of all time points with blue/red coloring for normal/anomaly"""
    # Combine features into array
    X_n = np.column_stack([features_n['flatness'], features_n['rolloff'], 
                          features_n['bandwidth'], features_n['centroid']])
    X_a = np.column_stack([features_a['flatness'], features_a['rolloff'],
                         features_a['bandwidth'], features_a['centroid']])
    
    # Standardize and combine
    scaler = StandardScaler()
    X = scaler.fit_transform(np.vstack([X_n, X_a]))
    
    # t-SNE (adjust perplexity for your dataset size)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # Split back into normal/anomaly
    n_points = len(X_n)
    tsne_n = X_tsne[:n_points]
    tsne_a = X_tsne[n_points:]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot with fixed colors
    ax.scatter(tsne_n[:, 0], tsne_n[:, 1], color='blue', alpha=0.6, label='Normal')
    ax.scatter(tsne_a[:, 0], tsne_a[:, 1], color='red', alpha=0.6, label='Anomaly')
    
    # Formatting
    ax.set_title(f"{drone_type} {movement} - t-SNE (Normal vs Anomaly)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for drone in DRONE_TYPES:
        for mv in MOVEMENTS:
            # File handling (same as before)
            wav_n = find_wav(BASE_PATH, "train", drone, mv, "normal")
            wav_a = find_wav(BASE_PATH, "eval", drone, mv, "anomaly")
            
            if not wav_n or not wav_a:
                print(f"Missing pair: {drone}_{mv}")
                continue
                
            print(f"\nProcessing {drone} {mv}...")
            print(f"Normal file: {wav_n}")
            print(f"Anomaly file: {wav_a}")
            
            # Load audio
            y_n, sr = load_mono(wav_n)
            y_a, _ = load_mono(wav_a)
            
            # Extract time-series features
            features_n = extract_time_series_features(y_n, sr)
            features_a = extract_time_series_features(y_a, sr)
            
            # Plot and save
            out_png = os.path.join(OUTPUT_DIR, f"{drone}_{mv}_tsne_series.png")
            plot_tsne_time_series(features_n, features_a, drone, mv, out_png)
            print(f"Saved: {os.path.basename(out_png)}")

if __name__ == "__main__":
    main()