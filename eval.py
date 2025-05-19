import argparse
import os

import sklearn.metrics as metrics
import torch
import numpy as np

import dataset
from gmm import load_gmm
import utils


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the model")
    parser.add_argument("--model_path", type=str, required=True, help="Name of the model file")
    parser.add_argument("--result_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    return parser.parse_args()

def eval(args: argparse.Namespace) -> None:
    print("Evaluation started...")

    # Load GMM and feature means
    model_path = os.path.join(args.model_dir, args.model_path)
    gmm, feature_means = load_gmm(model_path)
    
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