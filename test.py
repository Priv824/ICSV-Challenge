import argparse
import os
import torch
from tqdm import tqdm

from dataset import extract_features
from gmm import GMMAnomalyDetector
import utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing test files")
    parser.add_argument("--result_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    parser.add_argument("--n_fft", type=int, default=2048, help="FFT window size")
    parser.add_argument("--hop_length", type=int, default=512, help="Hop length between frames")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--reg-covar", type=float, default=1e-6)
    return parser.parse_args()

def test(args: argparse.Namespace) -> None:
    print("Test started...")
    os.makedirs(args.result_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    
    # Load GMM
    model_path = os.path.join(args.model_dir, args.model_path)
    gmm = GMMAnomalyDetector(n_components=args.n_components, n_features=4, device=device)
    gmm.gmm.load_state_dict(torch.load(model_path))
    
    # Get test files
    file_list = [os.path.join(args.test_dir, f) for f in sorted(os.listdir(args.test_dir)) 
                if f.endswith('.wav')]
    
    # Score files
    score_list = [["File", "Score"]]
    
    for wav_path in tqdm(file_list, desc="Processing test files"):
        frames = extract_features(wav_path, args.sr, args.n_fft, args.hop_length)
        score = gmm.score_file(frames.to(device))
        
        file_name = os.path.splitext(os.path.basename(wav_path))[0]
        score_list.append([file_name, score])

    # Save results
    utils.save_csv(score_list, os.path.join(args.result_dir, "test_scores.csv"))
    print(f"Test scores saved to {os.path.join(args.result_dir, 'test_scores.csv')}")

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    test(args)