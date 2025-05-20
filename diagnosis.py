import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from dataset import extract_features
import os

def load_features(data_dir, sr, n_fft, hop_length):
    """Load features for all files in a directory"""
    features = []
    for f in os.listdir(data_dir):
        if f.endswith('.wav'):
            wav_path = os.path.join(data_dir, f)
            frames = extract_features(wav_path, sr, n_fft, hop_length)
            features.append(frames.numpy())
    return np.vstack(features)

def plot_tsne(normal_frames, anomaly_frames):
    """Visualize feature separation using t-SNE"""
    all_frames = np.vstack([normal_frames, anomaly_frames])
    labels = np.array([0]*len(normal_frames) + [1]*len(anomaly_frames))
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings = tsne.fit_transform(all_frames)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(embeddings[labels==0, 0], embeddings[labels==0, 1], 
                alpha=0.5, label="Normal", color='blue')
    plt.scatter(embeddings[labels==1, 0], embeddings[labels==1, 1], 
                alpha=0.5, label="Anomaly", color='red')
    plt.title("t-SNE of Normal vs Anomaly Features")
    plt.legend()
    plt.savefig("tsne_plot.png")
    plt.close()

def plot_feature_distributions(normal_frames, anomaly_frames, feature_names):
    """Plot distributions of each feature using matplotlib"""
    for i, name in enumerate(feature_names):
        plt.figure(figsize=(8, 4))
        
        # Calculate histograms
        n_bins = 50
        n_hist, n_bins = np.histogram(normal_frames[:, i], bins=n_bins, density=True)
        a_hist, a_bins = np.histogram(anomaly_frames[:, i], bins=n_bins, density=True)
        
        # Plot lines
        plt.plot(n_bins[:-1], n_hist, label="Normal", color='blue')
        plt.plot(a_bins[:-1], a_hist, label="Anomaly", color='red')
        
        plt.title(f"Distribution of {name}")
        plt.xlabel("Feature Value")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(f"feature_{i}_distribution.png")
        plt.close()

if __name__ == "__main__":
    # Configuration (match your training params)
    SR = 16000
    N_FFT = 2048
    HOP_LENGTH = 512
    
    # Load features
    print("Loading normal samples...")
    normal_frames = load_features("./data/train", SR, N_FFT, HOP_LENGTH)
    print(f"Loaded {len(normal_frames)} normal frames")
    
    print("Loading anomaly samples...")
    anomaly_frames = load_features("./data/eval", SR, N_FFT, HOP_LENGTH)
    print(f"Loaded {len(anomaly_frames)} anomaly frames")
    
    # Feature names for plots
    feature_names = [
        "Spectral Flatness",
        "Spectral Rolloff", 
        "Spectral Bandwidth",
        "Spectral Centroid"
    ]
    
    # Run diagnostics
    print("\nRunning t-SNE visualization...")
    plot_tsne(normal_frames, anomaly_frames)
    
    print("Plotting feature distributions...")
    plot_feature_distributions(normal_frames, anomaly_frames, feature_names)
    
    print("\nDiagnostic plots saved to:")
    print("- tsne_plot.png")
    print("- feature_*_distribution.png")