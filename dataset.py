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
    """Normalize features by feature means with strict shape enforcement"""
    if feature_means is None:
        return features, torch.zeros_like(features)
    
    # Convert and align devices
    if isinstance(feature_means, np.ndarray):
        feature_means = torch.from_numpy(feature_means).float()
    feature_means = feature_means.to(features.device)
    
    # Ensure features are [..., 8]
    if features.dim() == 1:
        features = features.unsqueeze(0)
    features = features[..., :8]  # Force 8 features
    
    # Ensure means are [8]
    feature_means = feature_means.view(-1)[:8]  # Flatten and take first 8
    
    # Normalize
    normalized = features / feature_means
    return normalized.squeeze(), feature_means.squeeze()

def collate_fn(batch):
    """Strict collate function that enforces [batch_size, 8] shape"""
    features = torch.stack([item[0].view(-1)[:8] for item in batch])  # [batch_size, 8]
    labels = torch.tensor([item[1] for item in batch])
    drone_labels = torch.tensor([item[2] for item in batch])
    direction_labels = torch.tensor([item[3] for item in batch])
    return features, labels, drone_labels, direction_labels

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
    
    # Ensure we only take the first 8 features if more exist
    features = np.concatenate([features_mean, features_std])[:8]  # Force 8 features
    
    # Convert to tensor and ensure proper shape
    features = torch.from_numpy(features).float().view(-1)  # Shape: [8]
    
    if feature_means is not None:
        if isinstance(feature_means, np.ndarray):
            feature_means = torch.from_numpy(feature_means).float()
        feature_means = feature_means.to(features.device)
    
    # Normalize only if means are provided
    if feature_means is not None:
        # Ensure shapes match
        if len(feature_means) != len(features):
            feature_means = feature_means[:len(features)]  # Truncate if needed
        normalized_features = features / feature_means
        return normalized_features, feature_means
    
    return features, torch.zeros_like(features)

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
        
        # Ensure features are exactly 8-dimensional
        features = features[:8] if len(features) > 8 else features
        
        anomaly_label = utils.get_anomaly_label(wav_path)
        drone_label = utils.get_drone_label(wav_path)
        direction_label = utils.get_direction_label(wav_path)

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
        collate_fn=collate_fn  # This is critical
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


    """Custom collate function to ensure correct feature shapes."""
    features = torch.stack([item[0] for item in batch])  # Shape: [batch_size, 8]
    labels = torch.tensor([item[1] for item in batch])
    drone_labels = torch.tensor([item[2] for item in batch])
    direction_labels = torch.tensor([item[3] for item in batch])
    return features, labels, drone_labels, direction_labels