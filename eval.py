import argparse
import os

import sklearn.metrics as metrics
import torch
import numpy as np

import dataset
from gmm import load_gmm
import utils
import yaml

def get_args() -> argparse.Namespace:
    # Load parameters from YAML file
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
    parser.add_argument("--win_length", default=param["win_length"], type=int)
    parser.add_argument("--hop_length", default=param["hop_length"], type=int)
    parser.add_argument("--n_mels", default=param["n_mels"], type=int)
    parser.add_argument("--power", default=param["power"], type=float)
    
    # Hardware parameters
    parser.add_argument("--gpu", default=param["gpu"], type=int)
    
    args = parser.parse_args()
    return args
def eval(args: argparse.Namespace) -> None:
    print("Evaluation started...")
    os.makedirs(args.result_dir, exist_ok=True)

    # Load GMM and feature means
    model_path = os.path.join(args.model_dir, args.model_path)
    gmm, feature_means = load_gmm(model_path, device=f'cuda:{args.gpu}')
    
    # Verify all required audio parameters exist
    required_audio_params = ['sr', 'n_fft', 'win_length', 'hop_length', 'n_mels', 'power']
    for param in required_audio_params:
        if not hasattr(args, param):
            raise ValueError(f"Missing required audio parameter: {param}")

    dataloader, file_list = dataset.get_eval_loader(args, feature_means=feature_means)

    score_list = [["File", "Score"]]
    y_true, y_pred = [], []
    drone_label_list = []

    for idx, data in enumerate(dataloader):
        features = data[0].numpy()[np.newaxis, :]  # [1, 4]
        anomaly_label = data[1]
        drone_label = data[2]
        
        score = gmm.score_samples(features)[0]
        
        drone_label_list.append(drone_label.item())
        y_true.append(1 if anomaly_label.item() > 0 else 0)
        y_pred.append(score)
        
        file_name = os.path.splitext(file_list[idx].split("/")[-1])[0]
        score_list.append([file_name, score])

    auc = metrics.roc_auc_score(y_true, y_pred)
    print("Overall AUC: ", auc)
    utils.save_csv(score_list, os.path.join(args.result_dir, "eval_score.csv"))

    # Calculate AUC per drone type
    drone_type_list = ["A", "B", "C"]
    for drone_type in drone_type_list:
        indices = [
            i
            for i, label in enumerate(drone_label_list)
            if label == drone_type_list.index(drone_type)
        ]
        if not indices:
            continue
            
        pred_labels = [y_pred[i] for i in indices]
        true_labels = [y_true[i] for i in indices]
        fault_auc = metrics.roc_auc_score(true_labels, pred_labels)
        print(f"Drone {drone_type} AUC: {fault_auc}")


if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    eval(args)