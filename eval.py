import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import sklearn.metrics as metrics
import yaml
import matplotlib.pyplot as plt

from dataset import extract_features
from gmm import GMMAnomalyDetector
import utils

def get_args() -> argparse.Namespace:
    param_path = "./param.yaml"
    with open(param_path) as f:
        param = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_dir", default=param["model_dir"], type=str)
    parser.add_argument("--model_path", default=param["model_path"], type=str)
    parser.add_argument("--result_dir", default=param["result_dir"], type=str)
    parser.add_argument("--eval_dir", default=param["eval_dir"], type=str)
    
    parser.add_argument("--sr", default=param["sr"], type=int)
    parser.add_argument("--n_fft", default=param["n_fft"], type=int)
    parser.add_argument("--hop_length", default=param["hop_length"], type=int)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--reg-covar", type=float, default=1e-6)
    parser.add_argument("--gpu", default=param["gpu"], type=int)
    
    args = parser.parse_args()
    return args

def plot_likelihood_histogram(scores, y_true, save_path):
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

def plot_likelihoods_by_drone_movement(likelihood_scores, y_true, drone_labels, movement_labels, result_dir):
    drone_types = ["A", "B", "C"]
    movement_types = ["Back", "Front", "Left", "Right", "Clockwise", "CounterClockwise"]

    for i, drone in enumerate(drone_types):
        for movement in movement_types:
            idx = np.where((drone_labels == i) & (movement_labels == movement))[0]

            if len(idx) == 0:
                print(f"No data for drone {drone} and movement {movement}, skipping plot.")
                continue

            scores_subset = likelihood_scores[idx]
            y_subset = y_true[idx]

            normal_scores = scores_subset[y_subset == 0]
            anomaly_scores = scores_subset[y_subset == 1]

            plt.figure(figsize=(8, 5))
            plt.hist(normal_scores, bins=30, alpha=0.5, label='Normal', color='blue')
            plt.hist(anomaly_scores, bins=30, alpha=0.5, label='Anomalies', color='red')
            threshold = np.percentile(scores_subset, 5) if len(scores_subset) > 0 else 0
            plt.axvline(threshold, color='black', linestyle='dashed', linewidth=1, label='Threshold')
            plt.title(f'Drone {drone} - {movement} Likelihood Histogram')
            plt.xlabel('Likelihood Score')
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()

            save_path = os.path.join(result_dir, f"{drone}_{movement}_likelihood_histogram.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved histogram for drone {drone}, movement {movement} to {save_path}")

def eval(args: argparse.Namespace) -> None:
    print("Evaluation started...")
    os.makedirs(args.result_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')

    model_path = os.path.join(args.model_dir, args.model_path)
    gmm = GMMAnomalyDetector(n_components=args.n_components, n_features=4, device=device)
    gmm.gmm.load_state_dict(torch.load(model_path))

    file_list = []
    y_true = []
    drone_labels = []
    movement_labels = []
    likelihood_scores = []

    for f in sorted(os.listdir(args.eval_dir)):
        if f.endswith('.wav'):
            wav_path = os.path.join(args.eval_dir, f)
            file_list.append(wav_path)

            y_true.append(1 if utils.get_anomaly_label(wav_path) > 0 else 0)
            drone_labels.append(utils.get_drone_label(wav_path))
            movement_labels.append(utils.get_direction_label(wav_path))  # Use get_direction_label

            frames = extract_features(wav_path, args.sr, args.n_fft, args.hop_length)
            log_probs = gmm.gmm.score_samples(frames.to(device))
            avg_log_prob = log_probs.mean().item()
            likelihood_scores.append(avg_log_prob)

    y_true_np = np.array(y_true)
    drone_labels_np = np.array(drone_labels)
    movement_labels_np = np.array(movement_labels)
    likelihood_scores_np = np.array(likelihood_scores)

    score_list = [["File", "Score"]]
    for file_name, score in zip(file_list, likelihood_scores):
        score_list.append([os.path.basename(file_name), score])
    utils.save_csv(score_list, os.path.join(args.result_dir, "eval_scores.csv"))

    hist_save_path = os.path.join(args.result_dir, "likelihood_histogram.png")
    plot_likelihood_histogram(likelihood_scores_np, y_true_np, hist_save_path)
    print(f"Likelihood histogram saved to {hist_save_path}")

    plot_likelihoods_by_drone_movement(likelihood_scores_np, y_true_np, drone_labels_np, movement_labels_np, args.result_dir)

    auc = metrics.roc_auc_score(y_true_np, likelihood_scores_np)
    print(f"\nOverall AUC: {auc:.4f}")

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
