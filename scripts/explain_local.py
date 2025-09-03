#!/usr/bin/env python3
"""
Local attribution analysis for 3D turbulence models.
Supports Integrated Gradients, GradientSHAP, and 3D occlusion.
"""
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.devices import pick_device
from src.dataio.channel_dataset import ChannelDataset
from src.models.unet3d import UNet3D
from src.interp.grad import integrated_gradients, gradient_shap
from src.interp.occlusion3d import occlusion_importance

def load_model(cfg, device):
    """Load trained model from checkpoint."""
    exp_id = cfg['experiment_id']
    results_dir = Path(cfg['paths']['results_dir']) / exp_id
    
    # Find best checkpoint
    ckpts = sorted(results_dir.glob('best_*.pth'))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {results_dir}")
    
    ckpt_path = ckpts[-1]
    
    # Build model
    mcfg = cfg['model']
    model = UNet3D(
        mcfg['in_channels'], 
        mcfg['out_channels'], 
        base_ch=mcfg['base_channels']
    )
    
    # Load weights
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {ckpt_path.name}")
    return model

def compute_attribution(model, x, method, device):
    """Compute attribution using specified method."""
    x = x.to(device)
    
    if method == 'ig':
        return integrated_gradients(model, x, steps=32, baseline='zeros')
    elif method == 'gradshap':
        return gradient_shap(model, x, nsamples=20)
    elif method == 'occlusion':
        return occlusion_importance(model, x, patch=8, stride=8)
    else:
        raise ValueError(f"Unknown method: {method}")

def save_attribution(attribution, sample_idx, method, results_dir):
    """Save attribution as NPZ file."""
    save_path = results_dir / f"sample_{sample_idx:03d}_attrib.npz"
    np.savez_compressed(save_path, attribution=attribution.cpu().numpy())
    return save_path

def plot_attribution_slice(x, attribution, sample_idx, method, figures_dir):
    """Plot central Z slice of input and attribution."""
    # Convert to numpy and get central slice
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
        
    if isinstance(attribution, torch.Tensor):
        attr_np = attribution.cpu().numpy()
    else:
        attr_np = attribution
    
    # Handle different shapes for occlusion vs gradient methods
    if len(attr_np.shape) == 3:  # Occlusion output (D, H, W)
        z_idx = attr_np.shape[0] // 2
        attr_slice = attr_np[z_idx, :, :]
        # For input, take first channel and central slice
        x_slice = x_np[0, 0, x_np.shape[2] // 2, :, :]
    else:  # Gradient methods (1, C, D, H, W)
        z_idx = attr_np.shape[2] // 2
        attr_slice = attr_np[0, 0, z_idx, :, :]  # First channel
        x_slice = x_np[0, 0, z_idx, :, :]
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Input slice
    im1 = axes[0].imshow(x_slice, cmap='viridis')
    axes[0].set_title(f'Input (z={z_idx})')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # Attribution slice
    im2 = axes[1].imshow(attr_slice, cmap='RdBu_r')
    axes[1].set_title(f'Attribution ({method})')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    # Save plot
    save_path = figures_dir / f"sample_{sample_idx:03d}_slice.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path

def main():
    parser = argparse.ArgumentParser(description='Local attribution analysis')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--split', choices=['val', 'test'], default='val', help='Dataset split')
    parser.add_argument('--method', choices=['ig', 'gradshap', 'occlusion'], required=True, help='Attribution method')
    parser.add_argument('--n', type=int, default=4, help='Number of samples to analyze')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config and setup paths
    cfg = load_config(args.config)
    exp_id = cfg['experiment_id']
    device = pick_device(args.cuda)
    
    results_dir = Path(cfg['paths']['results_dir']) / exp_id / 'interp' / args.method
    figures_dir = Path('figures') / exp_id / 'interp' / args.method
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = ChannelDataset(cfg, args.split, eval_mode=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load model
    model = load_model(cfg, device)
    
    # Process samples
    attribution_stats = []
    
    for i, (x, y) in enumerate(loader):
        if i >= args.n:
            break
            
        print(f"Processing sample {i+1}/{args.n}...")
        
        # Compute attribution
        attribution = compute_attribution(model, x, args.method, device)
        
        # Save attribution
        save_path = save_attribution(attribution, i, args.method, results_dir)
        print(f"  Saved attribution: {save_path}")
        
        # Plot slice
        plot_path = plot_attribution_slice(x, attribution, i, args.method, figures_dir)
        print(f"  Saved plot: {plot_path}")
        
        # Compute stats
        attr_norm = torch.norm(attribution).item()
        attr_mean = torch.mean(torch.abs(attribution)).item()
        attr_max = torch.max(torch.abs(attribution)).item()
        
        attribution_stats.append({
            'sample': i,
            'norm': attr_norm,
            'mean_abs': attr_mean,
            'max_abs': attr_max
        })
    
    # Print summary table
    print(f"\nAttribution Statistics ({args.method}):")
    print("Sample | L2 Norm  | Mean |Attr| | Max |Attr|")
    print("-------|----------|-----------|----------")
    for stats in attribution_stats:
        print(f"{stats['sample']:6d} | {stats['norm']:8.3f} | {stats['mean_abs']:9.6f} | {stats['max_abs']:8.6f}")
    
    print(f"\nResults saved to: {results_dir}")
    print(f"Figures saved to: {figures_dir}")

if __name__ == '__main__':
    main()
