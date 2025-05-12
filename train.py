import argparse
import os

import torch
import yaml
from tqdm import tqdm

import dataset
import net
import utils

import matplotlib.pyplot as plt
import numpy as np

def save_spectrogram_comparison(input_mel, output_mel, epoch, batch_idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy and take first batch item
    input_np = input_mel[0, 0].detach().cpu().numpy()
    output_np = output_mel[0, 0].detach().cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.imshow(input_np, aspect='auto', origin='lower')
    plt.title("Input Spectrogram")
    plt.colorbar()
    
    plt.subplot(2, 1, 2)
    plt.imshow(output_np, aspect='auto', origin='lower')
    plt.title("Output Spectrogram")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"compare_epoch{epoch}_batch{batch_idx}.png"))
    plt.close()

def visualize_activations(model, input_tensor, save_path):
    # Hook to capture activations
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv1d, nn.LayerNorm)):
            hooks.append(layer.register_forward_hook(get_activation(name)))
    
    # Forward pass
    with torch.no_grad():
        model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Visualize
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for name, act in activations.items():
        if len(act.shape) == 3:  # Only visualize conv activations
            plt.figure(figsize=(12, 6))
            plt.imshow(act[0].cpu().numpy(), aspect='auto', cmap='viridis')
            plt.title(f"Activations: {name}")
            plt.colorbar()
            plt.savefig(os.path.join(save_path, f"activations_{name.replace('.', '_')}.png"))
            plt.close()

def get_args() -> argparse.Namespace:
    param_path = "./param.yaml"
    with open(param_path) as f:
        param = yaml.safe_load(f)
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-dir", default=param["train_dir"], type=str)
    parser.add_argument("--eval-dir", default=param["eval_dir"], type=str)
    parser.add_argument("--test-dir", default=param["test_dir"], type=str)

    parser.add_argument("--result-dir", default=param["result_dir"], type=str)
    parser.add_argument("--model-dir", default=param["model_dir"], type=str)

    parser.add_argument("--model-path", default=param["model_path"], type=str)

    parser.add_argument("--epochs", default=param["epochs"], type=int)
    parser.add_argument("--batch-size", default=param["batch_size"], type=int)
    parser.add_argument("--lr", default=param["lr"], type=float)

    parser.add_argument("--gpu", default=param["gpu"], type=int)
    parser.add_argument("--n-workers", default=param["n_workers"], type=int)

    parser.add_argument("--sr", default=param["sr"], type=int)
    parser.add_argument("--n-fft", default=param["n_fft"], type=int)
    parser.add_argument("--win-length", default=param["win_length"], type=int)
    parser.add_argument("--hop-length", default=param["hop_length"], type=int)
    parser.add_argument("--n-mels", default=param["n_mels"], type=int)
    parser.add_argument("--power", default=param["power"], type=float)

    args = parser.parse_args()
    return args


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> None:
    print("Training started...")
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Create directory for analysis
    analysis_dir = os.path.join(args.result_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    model = net.WaveNetModel().cuda()
    dataloader = dataset.get_train_loader(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        model.train()

        p_bar = tqdm(dataloader, total=len(dataloader), desc="Training", ncols=100)
        for batch_idx, data in enumerate(p_bar):
            log_mel = data[0].cuda()
            recon_log_mel = model(log_mel)
            
            # Calculate loss on the valid part
            target = log_mel[..., model.get_receptive_field() :]
            loss = criterion(recon_log_mel, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            p_bar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
            
            # Save comparison every N batches
            if batch_idx % 100 == 0:
                save_spectrogram_comparison(
                    log_mel, 
                    recon_log_mel, 
                    epoch, 
                    batch_idx, 
                    analysis_dir
                )
                
                # Calculate and print frequency band errors
                with torch.no_grad():
                    error_per_band = torch.mean(torch.abs(recon_log_mel - target), dim=(0, 2, 3))
                    print("\nFrequency band errors:")
                    for i, err in enumerate(error_per_band):
                        print(f"Band {i}: {err.item():.4f}")
        # At the end of each epoch, visualize activations
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            # Get a sample batch
            sample_data = next(iter(dataloader))
            sample_input = sample_data[0].cuda()
            
            # Visualize activations
            visualize_activations(
                model,
                sample_input,
                os.path.join(analysis_dir, f"epoch_{epoch}_activations")
            )
    utils.save_model(model, os.path.join(args.model_dir, args.model_path))

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(2025)

    train(args)
