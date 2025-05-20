import argparse
import os
from typing import List, Tuple, Optional, Union

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from librosa.feature import spectral_flatness, spectral_rolloff, spectral_bandwidth, spectral_centroid

import utils

def normalize_features(features: torch.Tensor, feature_means: Optional[Union[torch.Tensor, np.ndarray]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize features by dividing by their mean.
    If feature_means is None, calculates means from the input.
    Returns normalized features and the means used.
    """
    # Convert feature_means to tensor if it's a numpy array
    if isinstance(feature_means, np.ndarray):
        feature_means = torch.from_numpy(feature_means).float()
    
    # Ensure features are 2D
    if features.dim() == 1:
        features = features.unsqueeze(0)
    
    # Move feature_means to same device as features
    if feature_means is not None:
        feature_means = feature_means.to(features.device)
    
    if feature_means is None:
        feature_means = torch.mean(features, dim=0, keepdim=True)
    
    # Avoid division by zero
    feature_means = torch.where(
        feature_means == 0, 
        torch.tensor(1.0, device=features.device), 
        feature_means
    )
    
    normalized_features = features / feature_means
    
    return normalized_features, feature_means

def extract_features(
    wav_path: str,
    sr: int,
    n_fft: int,
    hop_length: int,
    feature_means: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    wav_data, _ = torchaudio.load(wav_path)
    wav_data = wav_data.numpy()[0]  # Convert to mono numpy array
    
    # Create overlapping frames
    n_frames = 1 + (len(wav_data) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        wav_data,
        shape=(n_frames, n_fft),
        strides=(hop_length * wav_data.itemsize, wav_data.itemsize)
    )
    
    # Apply window to all frames
    window = np.hanning(n_fft)
    frames = frames * window[None, :]
    
    # Compute features for each frame
    features_list = []
    for frame in frames:
        frame_features = np.array([
            spectral_flatness(y=frame, n_fft=n_fft, hop_length=n_fft)[0],
            spectral_rolloff(y=frame, sr=sr, n_fft=n_fft, hop_length=n_fft)[0],
            spectral_bandwidth(y=frame, sr=sr, n_fft=n_fft, hop_length=n_fft)[0],
            spectral_centroid(y=frame, sr=sr, n_fft=n_fft, hop_length=n_fft)[0]
        ])
        features_list.append(frame_features)
    
    # Stack all frame features and compute statistics
    features = np.stack(features_list)  # [n_frames, 4]
    features_mean = np.mean(features, axis=0)  # [4]
    features_std = np.std(features, axis=0)    # [4]
    
    # Concatenate statistics to get final feature vector
    features = np.concatenate([features_mean, features_std])  # [8]
    
    # Convert to tensor and ensure shape [8]
    features = torch.from_numpy(features).float().view(-1)  # Force flattening
    
    # Move to same device as feature_means if provided
    if feature_means is not None and isinstance(feature_means, torch.Tensor):
        features = features.to(feature_means.device)
    
    # Normalize features
    normalized_features, used_means = normalize_features(features.unsqueeze(0), feature_means)
    
    return normalized_features.squeeze(0), used_means.squeeze(0)

class BaselineDataLoader(Dataset):
    def __init__(
        self,
        file_list: List[str],
        sr: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        power: float,
        feature_means: Optional[np.ndarray] = None
    ) -> None:
        self.file_list = file_list
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.feature_means = feature_means

    def __len__(self) -> int:
        return len(self.file_list)

def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, int]:
    wav_path = self.file_list[idx]
    features, _ = extract_features(
        wav_path,
        self.sr,
        self.n_fft,
        self.hop_length,
        self.feature_means
    )
    
    anomaly_label = utils.get_anomaly_label(wav_path)
    drone_label = utils.get_drone_label(wav_path)
    direction_label = utils.get_direction_label(wav_path)

    # Ensure features are 1D (shape [8])
    return features, anomaly_label, drone_label, direction_label

def get_train_loader(
    args: argparse.Namespace,
    feature_means: Optional[np.ndarray] = None
) -> DataLoader:
    file_list = os.listdir(args.train_dir)
    file_list.sort()
    file_list = [os.path.join(args.train_dir, file) for file in file_list]
    
    train_dataloader = BaselineDataLoader(
        file_list, args.sr, args.n_fft, args.win_length, 
        args.hop_length, args.n_mels, args.power, feature_means
    )

    return DataLoader(
        train_dataloader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=collate_fn  # Add this line
    )

def get_eval_loader(
    args: argparse.Namespace,
    feature_means: Optional[np.ndarray] = None
) -> Tuple[DataLoader, List[str]]:
    file_list = os.listdir(args.eval_dir)
    file_list.sort()
    file_list = [os.path.join(args.eval_dir, file) for file in file_list]
    
    eval_dataloader = BaselineDataLoader(
        file_list, args.sr, args.n_fft, args.win_length,
        args.hop_length, args.n_mels, args.power, feature_means
    )

    return DataLoader(
        eval_dataloader, batch_size=1, shuffle=False, num_workers=0
    ), file_list

def get_test_loader(
    args: argparse.Namespace,
    feature_means: Optional[np.ndarray] = None
) -> Tuple[DataLoader, List[str]]:
    file_list = os.listdir(args.test_dir)
    file_list.sort()
    file_list = [os.path.join(args.test_dir, file) for file in file_list]
    
    test_dataloader = BaselineDataLoader(
        file_list, args.sr, args.n_fft, args.win_length,
        args.hop_length, args.n_mels, args.power, feature_means
    )

    return DataLoader(
        test_dataloader, batch_size=1, shuffle=False, num_workers=0
    ), file_list

def collate_fn(batch):
    """Custom collate function to ensure correct feature shapes."""
    features = torch.stack([item[0] for item in batch])  # Shape: [batch_size, 8]
    labels = torch.tensor([item[1] for item in batch])
    drone_labels = torch.tensor([item[2] for item in batch])
    direction_labels = torch.tensor([item[3] for item in batch])
    return features, labels, drone_labels, direction_labels