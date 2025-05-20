import argparse
import os
import torch
import yaml
from tqdm import tqdm
import utils  # Make sure you have this utility module

from dataset import get_train_loader
from gmm import GMMAnomalyDetector

def get_args() -> argparse.Namespace:
    """Load and parse command line arguments"""
    param_path = "./param.yaml"
    with open(param_path) as f:
        param = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument("--train-dir", default=param["train_dir"], type=str)
    
    # Model parameters
    parser.add_argument("--model-dir", default=param["model_dir"], type=str)
    parser.add_argument("--model-path", default=param["model_path"], type=str)
    
    # Training parameters
    parser.add_argument("--epochs", default=param["epochs"], type=int)
    parser.add_argument("--batch-size", default=param["batch_size"], type=int)
    parser.add_argument("--lr", default=param["lr"], type=float)
    parser.add_argument("--n-components", default=param["n_components"], type=int)
    parser.add_argument("--max-iter", default=param["max_iter"], type=int)
    parser.add_argument("--tol", default=param["tol"], type=float)
    parser.add_argument("--reg-covar", default=param["reg_covar"], type=float)
    
    # Hardware parameters
    parser.add_argument("--gpu", default=param["gpu"], type=int)
    parser.add_argument("--n-workers", default=param["n_workers"], type=int)
    
    # Audio processing parameters
    parser.add_argument("--sr", default=param["sr"], type=int)
    parser.add_argument("--n-fft", default=param["n_fft"], type=int)
    parser.add_argument("--hop-length", default=param["hop_length"], type=int)
    args = parser.parse_args()
    return args

def set_seed(seed: int = 2025) -> None:
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(args: argparse.Namespace) -> None:
    """Main training function"""
    print("Training started...")
    os.makedirs(args.model_dir, exist_ok=True)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load all frame-level features
    train_loader = get_train_loader(args)
    all_frames = []
    for frames, _ in tqdm(train_loader, desc="Loading frames"):
        all_frames.append(frames.to(device))
    
    all_frames = torch.cat(all_frames)  # [total_frames, 4]
    print(f"Total frames loaded: {len(all_frames)}")
    print(f"Feature dimension: {all_frames.shape[1]}")
    
    # Train GMM
    gmm = GMMAnomalyDetector(n_components=args.n_components, n_features=4, device=device)
    gmm.fit(all_frames, max_iter=args.max_iter, tol=args.tol)
    
    # Save model
    model_path = os.path.join(args.model_dir, args.model_path)
    torch.save(gmm.gmm.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed()
    train(args)