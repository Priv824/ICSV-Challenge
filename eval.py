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


def plot_likelihood_histogram(scores, y_true, save_path, title="Likelihood Histogram"):
    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]

    plt.figure(figsize=(8, 5))
    plt.hist(normal_scores, bins=30, alpha=0.6, label='Normal', color='dodgerblue')
    plt.hist(anomaly_scores, bins=30, alpha=0.6, label='Anomaly', color='crimson')
    plt.axvline(np.percentile(scores, 5), color='black', linestyle='dashed', linewidth=1.5, label='Threshold')
    plt.title(title, fontsize=14)
    plt.xlabel('Log Likelihood Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def eval(args: argparse.Namespace) -> None:
    print("Evaluation started...")
    os.makedirs(args.result_dir, exist_ok=True)

    device = 'cpu'
    model_path = os.path.join(args.model_dir, args.model_path)
    gmm = GMMAnomalyDetector(n_components=args.n_components, n_features=4, device=device)
    gmm.gmm.load_state_dict(torch.load(model_path))

    file_list, y_true, drone_labels, direction_labels, likelihood_scores = [], [], [], [], []

    for f in sorted(os.listdir(args.eval_dir)):
        if f.endswith('.wav'):
            wav_path = os.path.join(args.eval_dir, f)
            file_list.append(wav_path)
            y_true.append(1 if utils.get_anomaly_label(wav_path) > 0 else 0)
            drone_labels.append(utils.get_drone_label(wav_path))
            direction_labels.append(utils.get_direction_label(wav_path))

            frames = extract_features(wav_path, args.sr, args.n_fft, args.hop_length)
            log_probs = gmm.gmm.score_samples(frames.to(device))
            avg_log_prob = log_probs.mean().item()
            likelihood_scores.append(avg_log_prob)

    y_true = np.array(y_true)

    # Save score table
    score_list = [["File", "Score"]]
    for file_name, score in zip(file_list, likelihood_scores):
        score_list.append([os.path.basename(file_name), score])
    utils.save_csv(score_list, os.path.join(args.result_dir, "eval_scores.csv"))

    # Save global histogram
    plot_likelihood_histogram(np.array(likelihood_scores), y_true,
                              os.path.join(args.result_dir, "likelihood_histogram.png"),
                              title="Likelihood Histogram for Normal vs Anomalies")

        # Overall AUC
    auc = metrics.roc_auc_score(y_true, likelihood_scores)
    print(f"\nOverall AUC: {auc:.4f}")
    # Per drone-type AUC
    drone_types = ["A", "B", "C"]
    for i, drone in enumerate(drone_types):
        indices = [idx for idx, label in enumerate(drone_labels) if label == i]
        if not indices:
            continue
        drone_true = [y_true[idx] for idx in indices]
        drone_pred = [likelihood_scores[idx] for idx in indices]
        drone_auc = metrics.roc_auc_score(drone_true, drone_pred)
        print(f"Drone {drone} AUC: {drone_auc:.4f}")

    # Plot per drone+direction histogram
    label_map = {0: "Back", 1: "Front", 2: "Left", 3: "Right", 4: "Clockwise", 5: "CounterClockwise"}
    for drone_id in [0, 1, 2]:
        for dir_id in range(6):
            indices = [i for i in range(len(y_true)) if drone_labels[i] == drone_id and direction_labels[i] == dir_id]
            if not indices:
                continue
            d_scores = np.array([likelihood_scores[i] for i in indices])
            d_y_true = np.array([y_true[i] for i in indices])
            drone_name = drone_types[drone_id]
            direction_name = label_map[dir_id]
            plot_title = f"Drone {drone_name} - {direction_name}"
            file_name = f"hist_drone{drone_name}_{direction_name}.png"
            save_path = os.path.join(args.result_dir, file_name)
            plot_likelihood_histogram(d_scores, d_y_true, save_path, title=plot_title)
            print(f"Saved: {file_name}")


if __name__ == "__main__":
    args = get_args()
    eval(args)
