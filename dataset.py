import os
import torch
import torchaudio
import numpy as np
import librosa
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import utils

def extract_features(
    wav_path: str,
    sr: int,
    n_fft: int,
    hop_length: int
) -> torch.Tensor:
    """Extract raw frame-level features (4D)"""
    # Load and convert to mono
    y, _ = torchaudio.load(wav_path)
    y = y.numpy()[0] if y.ndim == 2 else y.numpy()
    
    # Compute STFT
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    
    # Extract all 4 features per frame
    features = np.vstack([
        librosa.feature.spectral_flatness(S=S)/np.mean(S),
        librosa.feature.spectral_rolloff(S=S, sr=sr)/np.mean(S),
        librosa.feature.spectral_bandwidth(S=S, sr=sr)/np.mean(S),
        librosa.feature.spectral_centroid(S=S, sr=sr)/np.mean(S)
    ])  # Shape: [4, n_frames]
    
    return torch.from_numpy(features.T).float()  # [n_frames, 4]

class FrameDataset(Dataset):
    def __init__(
        self,
        file_list: List[str],
        sr: int,
        n_fft: int,
        hop_length: int
    ):
        self.file_list = file_list
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        wav_path = self.file_list[idx]
        frames = extract_features(wav_path, self.sr, self.n_fft, self.hop_length)
        label = utils.get_anomaly_label(wav_path)
        return frames, label

def collate_fn(batch):
    """Batch frames from multiple audio files"""
    frames = [item[0] for item in batch]  # List of [n_frames, 4] tensors
    labels = torch.tensor([item[1] for item in batch])
    return torch.cat(frames), labels  # [total_frames, 4], [batch_size]

def get_train_loader(args):
    file_list = [os.path.join(args.train_dir, f) for f in os.listdir(args.train_dir)]
    dataset = FrameDataset(file_list, args.sr, args.n_fft, args.hop_length)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )