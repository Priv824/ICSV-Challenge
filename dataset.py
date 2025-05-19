import argparse
import os
from typing import List, Tuple

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

import utils


def wav_to_log_mel(
    wav_path: str,
    sr: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
    power: float,
    fmin: float = 0.0,
    fmax: float = 100.0,
) -> torch.Tensor:
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
        f_min=fmin,
        f_max=fmax,
    )

    wav_data, _ = torchaudio.load(wav_path)
    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    mel_spec = mel_transform(wav_data)
    log_mel_spec = amp_to_db(mel_spec)
    return log_mel_spec


def get_train_loader(
    args: argparse.Namespace,
) -> DataLoader[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    train_dir = args.train_dir
    sr = args.sr
    n_fft = args.n_fft
    win_length = args.win_length
    hop_length = args.hop_length
    n_mels = args.n_mels
    power = args.power

    file_list = os.listdir(train_dir)
    file_list.sort()
    file_list = [os.path.join(train_dir, file) for file in file_list]
    train_dataloader = BaselineDataLoader(
        file_list, sr, n_fft, win_length, hop_length, n_mels, power
    )

    train_loader = DataLoader(
        train_dataloader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
    )
    return train_loader


def get_eval_loader(
    args: argparse.Namespace,
) -> Tuple[
    DataLoader[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], List[str]
]:
    eval_dir = args.eval_dir
    sr = args.sr
    n_fft = args.n_fft
    win_length = args.win_length
    hop_length = args.hop_length
    n_mels = args.n_mels
    power = args.power

    file_list = os.listdir(eval_dir)
    file_list.sort()
    file_list = [os.path.join(eval_dir, file) for file in file_list]
    eval_dataloader = BaselineDataLoader(
        file_list, sr, n_fft, win_length, hop_length, n_mels, power
    )

    eval_loader = DataLoader(
        eval_dataloader, batch_size=1, shuffle=False, num_workers=0
    )
    return eval_loader, file_list


def get_test_loader(
    args: argparse.Namespace,
) -> Tuple[
    DataLoader[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], List[str]
]:
    test_dir = args.test_dir
    sr = args.sr
    n_fft = args.n_fft
    win_length = args.win_length
    hop_length = args.hop_length
    n_mels = args.n_mels
    power = args.power

    file_list = os.listdir(test_dir)
    file_list.sort()
    file_list = [os.path.join(test_dir, file) for file in file_list]
    test_dataloader = BaselineDataLoader(
        file_list, sr, n_fft, win_length, hop_length, n_mels, power
    )

    test_loader = DataLoader(
        test_dataloader, batch_size=1, shuffle=False, num_workers=0
    )
    return test_loader, file_list


class BaselineDataLoader(Dataset):
    def __init__(
        self,
        file_list: list[str],
        sr: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        power: float,
    ) -> None:
        self.file_list = file_list
        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.power = power

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, int]:
        wav_path = self.file_list[idx]
        log_mel_spec = wav_to_log_mel(
            wav_path,
            self.sr,
            self.n_fft,
            self.win_length,
            self.hop_length,
            self.n_mels,
            self.power,
        )

        anomaly_label = utils.get_anomaly_label(wav_path)
        drone_label = utils.get_drone_label(wav_path)
        direction_label = utils.get_direction_label(wav_path)

        return log_mel_spec, anomaly_label, drone_label, direction_label
