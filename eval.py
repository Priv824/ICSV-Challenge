import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import sklearn.metrics as metrics
import yaml
import matplotlib.pyplot as plt  # Import Matplotlib for plotting

from dataset import extract_features
from gmm import GMMAnomalyDetector
import utils

def get_args() -> argparse.Namespace:
    """Load and parse command line arguments"""
    param_path = "./param.yaml"
    with open(param_path) as f:
        param = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument("--model_dir", default=param["model_dir"], type=str)
    parser.add_argument("--model_path", default=param["model_path"], type=str)
    parser.add_argument("--result_dir", default=param["result_dir"], type=str)
    
    # Data parameters
    parser.add_argument("--eval_dir", default=param["eval_dir"], type=str)
    
    # Audio processing parameters
    parser.add_argument("--sr", default=param["sr"], type=int)
    parser.add_argument("--n_fft", default=param["n_fft"], type=int)
    parser.add_argument("--hop_length", default=param["hop_length"], type=int)
    parser.add_argument("--n-components", type=int, default=param["n_components"])
    parser.add_argument("--reg-covar", type=float, default=param["reg_covar"])
    
    # Hardware parameters
    parser.add_argument("--gpu", default=param["gpu"], type=int)
    
    args = parser.parse_args()
    return args

def plot_likelihood_histogram(scores, y_true, save_path):
    """Plot histogram of likelihood scores for normal vs anomalies and save it."""
    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]

    plt.figure(figsize=(10, 6))
    plt.hist(normal_scores, bins=30, alpha=0.5, label='Normal', color='blue')
    plt.hist(anomaly_scores, bins=30, alpha=0.5, label='Anomalies', color='red')
    plt.axvline(np.percentile(scores, 5), color='black', linestyle='dashed', linewidth=1, label='Threshold')
    plt.title('Likelihood Histogram for Normal vs Anomalies')
    plt.xlabel('Likelihood Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def eval(args: argparse.Namespace) -> None:
    print("Evaluation started...")
    os.makedirs(args.result_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    
    # Load GMM
    model_path = os.path.join(args.model_dir, args.model_path)
    gmm = GMMAnomalyDetector(n_components=args.n_components, n_features=4, device=device)
    gmm.gmm.load_state_dict(torch.load(model_path))
    
    # Get file list and labels
    file_list = []
    y_true = []
    drone_labels = []
    likelihood_scores = []  # Store likelihood scores for each file

    for f in sorted(os.listdir(args.eval_dir)):
        if f.endswith('.wav'):
            wav_path = os.path.join(args.eval_dir, f)
            file_list.append(wav_path)
            y_true.append(1 if utils.get_anomaly_label(wav_path) > 0 else 0)
            drone_labels.append(utils.get_drone_label(wav_path))
            
            # Extract features and compute likelihood scores for the entire file
            frames = extract_features(wav_path, args.sr, args.n_fft, args.hop_length)
            log_probs = gmm.gmm.score_samples(frames.to(device))
            avg_log_prob = log_probs.mean().item()  # Average likelihood score for the file
            likelihood_scores.append(avg_log_prob)  # Store the average score

    # Convert y_true to a NumPy array for consistency
    y_true = np.array(y_true)

    # Save scores
    score_list = [["File", "Score"]]
    for file_name, score in zip(file_list, likelihood_scores):
        score_list.append([os.path.basename(file_name), score])
    utils.save_csv(score_list, os.path.join(args.result_dir, "eval_scores.csv"))
    
    # Calculate metrics
    auc = metrics.roc_auc_score(y_true, likelihood_scores)
    print(f"\nOverall AUC: {auc:.4f}")
    
    # Save likelihood histogram plot
    hist_save_path = os.path.join(args.result_dir, "likelihood_histogram.png")
    plot_likelihood_histogram(np.array(likelihood_scores), y_true, hist_save_path)
    print(f"Likelihood histogram saved to {hist_save_path}")

    # Per-drone metrics
    drone_types = ["A", "B", "C"]
    for i, drone in enumerate(drone_types):
        indices = [idx for idx, label in enumerate(drone_labels) if label == i]
        if not indices:
            continue
            
        drone_true = [y_true[idx] for idx in indices]
        drone_pred = [likelihood_scores[idx] for idx in indices]
        drone_auc = metrics.roc_auc_score(drone_true, drone_pred)
        print(f"Drone {drone} AUC: {drone_auc:.4f}")

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    eval(args)
