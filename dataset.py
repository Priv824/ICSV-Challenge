import argparse
import os
from typing import List, Tuple, Optional

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from librosa.feature import spectral_flatness, spectral_rolloff, spectral_bandwidth, spectral_centroid

import utils

def normalize_features(features: np.ndarray, feature_means: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize features by dividing by their mean.
    If feature_means is None, calculates means from the input.
    Returns normalized features and the means used.
    """
    if feature_means is None:
        feature_means = np.mean(features, axis=0, keepdims=True)
    
    # Avoid division by zero
    feature_means = np.where(feature_means == 0, 1, feature_means)
    normalized_features = features / feature_means
    
    return normalized_features, feature_means

def extract_features(
    wav_path: str,
    sr: int,
    n_fft: int,
    hop_length: int,
    feature_means: Optional[np.ndarray] = None
) -> Tuple[torch.Tensor, np.ndarray]:
    wav_data, _ = torchaudio.load(wav_path)
    wav_data = wav_data.numpy()[0]  # Convert to mono numpy array
    
    # Compute features
    flatness = spectral_flatness(y=wav_data, n_fft=n_fft, hop_length=hop_length)
    rolloff = spectral_rolloff(y=wav_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
    bandwidth = spectral_bandwidth(y=wav_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
    centroid = spectral_centroid(y=wav_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    # Stack features and take mean over time
    features = np.vstack([flatness, rolloff, bandwidth, centroid])
    features = features.mean(axis=1)  # [4]
    
    # Normalize features
    normalized_features, used_means = normalize_features(features[np.newaxis, :], feature_means)
    
    return torch.from_numpy(normalized_features[0]).float(), used_means

class BaselineDataLoader(Dataset):
    def __init__(
        self,
        file_list: List[str],
        feature_means: Optional[np.ndarray] = None
    ) -> None:
        self.file_list = file_list
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
    )

def get_eval_loader(
    args: argparse.Namespace,
    feature_means: Optional[np.ndarray] = None
) -> Tuple[DataLoader, List[str]]:
    file_list = os.listdir(args.eval_dir)
    file_list.sort()
    file_list = [os.path.join(args.eval_dir, file) for file in file_list]
    
    eval_dataloader = BaselineDataLoader(
        file_list, feature_means
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