import argparse
import os

import numpy as np
import torch
import yaml
from tqdm import tqdm

import dataset
from gmm import GMMAnomalyDetector, save_gmm
import utils


def get_args() -> argparse.Namespace:
    param_path = "./param.yaml"
    with open(param_path) as f:
        param = yaml.safe_load(f)
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-dir", default=param["train_dir"], type=str)
    parser.add_argument("--eval-dir", default=param["eval_dir"], type=str)
    parser.add_argument("--test-dir", default=param["test_dir"], type=str)

    parser.add_argument("--result-dir", default=param["result_dir"], type=str)
    parser.add_argument("--model-dir", default=param["model_dir"], type=str)

    parser.add_argument("--model-path", default=param["model_path"], type=str)

    parser.add_argument("--epochs", default=param["epochs"], type=int)
    parser.add_argument("--batch-size", default=param["batch_size"], type=int)
    parser.add_argument("--lr", default=param["lr"], type=float)

    parser.add_argument("--gpu", default=param["gpu"], type=int)
    parser.add_argument("--n-workers", default=param["n_workers"], type=int)

    parser.add_argument("--sr", default=param["sr"], type=int)
    parser.add_argument("--n-fft", default=param["n_fft"], type=int)
    parser.add_argument("--win-length", default=param["win_length"], type=int)
    parser.add_argument("--hop-length", default=param["hop_length"], type=int)
    parser.add_argument("--n-mels", default=param["n_mels"], type=int)
    parser.add_argument("--power", default=param["power"], type=float)
    parser.add_argument("--fmin", default=param["fmin"], type=float)
    parser.add_argument("--fmax", default=param["fmax"], type=float)

    args = parser.parse_args()
    return args


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(args: argparse.Namespace) -> None:
    print("Training started...")
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # First pass to calculate feature means
    dataloader = dataset.get_train_loader(args)
    all_features = []
    for data in tqdm(dataloader, desc="Calculating feature means"):
        all_features.append(data[0].to(device))
    
    feature_means = torch.mean(torch.cat(all_features), dim=0)
    
    # Convert feature_means to numpy for dataloader compatibility
    feature_means_np = feature_means.cpu().numpy()
    
    # Second pass with normalized features
    dataloader = dataset.get_train_loader(args, feature_means=feature_means_np)
    normalized_features = []
    for data in tqdm(dataloader, desc="Collecting normalized features"):
        # Ensure data is on correct device
        features = data[0].to(device)
        normalized_features.append(features)
    
    normalized_features = torch.cat(normalized_features)
    
    # Train GMM
    gmm = GMMAnomalyDetector(n_components=3, n_features=normalized_features.shape[1], device=device)
    gmm.fit(normalized_features, n_epochs=args.epochs, lr=args.lr)
    
    # Save both GMM and feature means
    save_gmm(gmm, feature_means, os.path.join(args.model_dir, args.model_path))
    print(f"Model saved to {os.path.join(args.model_dir, args.model_path)}")

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(2025)

    train(args)