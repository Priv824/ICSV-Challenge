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


def plot_likelihood_per_drone_and_direction_separately(
    likelihood_scores, drone_labels, direction_labels, y_true, result_dir
):
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict
    import os

    drone_map = {0: "A", 1: "B", 2: "C"}
    direction_map = {
        0: "Back", 1: "Front", 2: "Left", 3: "Right", 4: "Clockwise", 5: "CounterClockwise"
    }

    data = defaultdict(lambda: {"Normal": [], "Anomaly": []})
    for score, drone, direction, label in zip(likelihood_scores, drone_labels, direction_labels, y_true):
        if direction == -1:
            continue
        drone_name = drone_map.get(drone, "Unknown")
        direction_name = direction_map.get(direction, "Unknown")
        label_str = "Anomaly" if label == 1 else "Normal"
        key = f"{drone_name}_{direction_name}"
        data[key][label_str].append(score)

    for key, value in data.items():
        normal = value["Normal"]
        anomaly = value["Anomaly"]
        if not normal and not anomaly:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        box_data = []
        labels = []
        colors = []

        if normal:
            box_data.append(normal)
            labels.append("Normal")
            colors.append("blue")
        if anomaly:
            box_data.append(anomaly)
            labels.append("Anomaly")
            colors.append("red")

        bp = ax.boxplot(box_data, patch_artist=True, labels=labels)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")

        ax.set_title(f"Likelihood Scores: {key}")
        ax.set_ylabel("Log-Likelihood Score")
        ax.set_ylim([min(min(normal + anomaly) - 1, -200), max(max(normal + anomaly) + 1, 0)])

        save_path = os.path.join(result_dir, f"likelihood_{key}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")


def eval(args: argparse.Namespace) -> None:
    print("Evaluation started...")
    os.makedirs(args.result_dir, exist_ok=True)

    device = 'cpu'
    # Load GMM
    model_path = os.path.join(args.model_dir, args.model_path)
    gmm = GMMAnomalyDetector(n_components=args.n_components, n_features=4, device='cpu')
    gmm.gmm.load_state_dict(torch.load(model_path))

    file_list = []
    y_true = []
    drone_labels = []
    direction_labels = []
    likelihood_scores = []

    for f in sorted(os.listdir(args.eval_dir)):
        if f.endswith('.wav'):
            wav_path = os.path.join(args.eval_dir, f)
            dir_label = utils.get_direction_label(wav_path)
            if dir_label == -1:
                continue

            file_list.append(wav_path)
            y_true.append(1 if utils.get_anomaly_label(wav_path) > 0 else 0)
            drone_labels.append(utils.get_drone_label(wav_path))
            direction_labels.append(dir_label)

            frames = extract_features(wav_path, args.sr, args.n_fft, args.hop_length)
            log_probs = gmm.gmm.score_samples(frames.to(device))
            avg_log_prob = log_probs.mean().item()
            likelihood_scores.append(avg_log_prob)

    y_true = np.array(y_true)

    # Save scores
    score_list = [["File", "Score"]]
    for file_name, score in zip(file_list, likelihood_scores):
        score_list.append([os.path.basename(file_name), score])
    utils.save_csv(score_list, os.path.join(args.result_dir, "eval_scores.csv"))

    # Calculate metrics
    auc = metrics.roc_auc_score(y_true, likelihood_scores)
    print(f"\nOverall AUC: {auc:.4f}")

    # Save histogram
    hist_save_path = os.path.join(args.result_dir, "likelihood_histogram.png")
    plot_likelihood_histogram(np.array(likelihood_scores), y_true, hist_save_path)
    print(f"Likelihood histogram saved to {hist_save_path}")

    # Per-drone AUC
    drone_types = ["A", "B", "C"]
    for i, drone in enumerate(drone_types):
        indices = [idx for idx, label in enumerate(drone_labels) if label == i]
        if not indices:
            continue
        drone_true = [y_true[idx] for idx in indices]
        drone_pred = [likelihood_scores[idx] for idx in indices]
        drone_auc = metrics.roc_auc_score(drone_true, drone_pred)
        print(f"Drone {drone} AUC: {drone_auc:.4f}")

    # Save per drone+direction plots
    plot_likelihood_per_drone_and_direction_separately(
        likelihood_scores,
        drone_labels,
        direction_labels,
        y_true,
        args.result_dir
    )


if __name__ == "__main__":
    args = get_args()
    eval(args)
