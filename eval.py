import argparse
import os

import sklearn.metrics as metrics
import torch
import numpy as np
from tqdm import tqdm

import dataset
import gmm
from gmm import GMMAnomalyDetector
import utils
from utils import extract_features
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

def evaluate(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    gmm = GMMAnomalyDetector(device=device)
    gmm.gmm.load_state_dict(torch.load(args.model_path))
    
    # Score files
    file_list = [os.path.join(args.eval_dir, f) for f in os.listdir(args.eval_dir)]
    scores = []
    for wav_path in tqdm(file_list):
        frames = extract_features(wav_path, args.sr, args.n_fft, args.hop_length)
        score = gmm.score_file(frames)
        scores.append((wav_path, score))
if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    eval(args)