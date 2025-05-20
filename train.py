import torch
import os
from tqdm import tqdm
from dataset import get_train_loader
from gmm import GMMAnomalyDetector

def train(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Load all frame-level features
    train_loader = get_train_loader(args)
    all_frames = []
    for frames, _ in tqdm(train_loader, desc="Loading frames"):
        all_frames.append(frames)
    all_frames = torch.cat(all_frames).to(device)  # [total_frames, 4]
    
    # Train GMM
    gmm = GMMAnomalyDetector(n_components=3, n_features=4, device=device)
    gmm.fit(all_frames, n_epochs=args.epochs, lr=args.lr)
    
    # Save model
    torch.save(gmm.gmm.state_dict(), os.path.join(args.model_dir, args.model_path))