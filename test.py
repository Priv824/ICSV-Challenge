import argparse
import os

import torch
import numpy as np
from tqdm import tqdm

import dataset
from gmm import load_gmm
import utils
from utils import extract_features


def test(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Load the trained GMM model
    gmm = load_gmm(args.model_path, device)
    
    # Create result directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Process test files
    test_files = [os.path.join(args.test_dir, f) for f in os.listdir(args.test_dir)]
    results = []
    
    for wav_path in tqdm(test_files):
        # Extract features
        frames = extract_features(wav_path, args.sr, args.n_fft, args.hop_length)
        
        # Calculate anomaly scores
        scores = gmm.score_samples(frames)
        anomaly_score = -scores.mean().item()  # Using negative log-likelihood as anomaly score
        
        # Store results
        filename = os.path.basename(wav_path)
        results.append({'file': filename, 'anomaly_score': anomaly_score})
        
    # Save results
    utils.save_results(results, os.path.join(args.result_dir, 'results.csv'))

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