import argparse
import os

import torch
import numpy as np

import dataset
from gmm import load_gmm
import utils


def test(args: argparse.Namespace) -> None:
    print("Test started...")

    # Load GMM and feature means
    model_path = os.path.join(args.model_dir, args.model_path)
    gmm, feature_means = load_gmm(model_path)
    
    dataloader, file_list = dataset.get_test_loader(args, feature_means=feature_means)

    score_list = [["File", "Score"]]

    for idx, data in enumerate(dataloader):
        features = data[0].numpy()[np.newaxis, :]  # [1, 4]
        score = gmm.score_samples(features)[0]
        
        file_name = os.path.splitext(file_list[idx].split("/")[-1])[0]
        score_list.append([file_name, score])

    utils.save_csv(score_list, os.path.join(args.result_dir, "test_score.csv"))
    print(f"Test scores saved to {os.path.join(args.result_dir, 'test_score.csv')}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--result_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    test(args)