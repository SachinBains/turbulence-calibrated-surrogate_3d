#!/usr/bin/env python3
"""
Faithfulness evaluation via top-k ablation curves.
Measures how much model performance degrades when masking top-k% most important features.
"""
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.devices import pick_device
from src.dataio.channel_dataset import ChannelDataset
from src.models.unet3d import UNet3D
from src.interp.grad import integrated_gradients
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
    
    return model

def compute_attribution(model, x, method, device):
    """Compute attribution using specified method."""
    x = x.to(device)
    
    if method == 'ig':
        return integrated_gradients(model, x, steps=32, baseline='zeros')
    elif method == 'occlusion':
        # For occlusion, we need to upsample to match input resolution
        occlusion_map = occlusion_importance(model, x, patch=8, stride=8)
        # Upsample occlusion map to match input size
        B, C, D, H, W = x.shape
        occlusion_upsampled = torch.nn.functional.interpolate(
            occlusion_map.unsqueeze(0).unsqueeze(0),
            size=(D, H, W),
            mode='trilinear',
            align_corners=False
        ).squeeze()
        # Expand to match input channels
        return occlusion_upsampled.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1, -1)
    else:
        raise ValueError(f"Unknown method: {method}")

def mask_top_k_features(x, attribution, k_percent):
    """Mask top-k% most important features by setting them to channel mean."""
    # Flatten attribution to find top-k indices
    attr_flat = attribution.view(-1)
    attr_abs = torch.abs(attr_flat)
    
    # Find top-k% indices
    k_count = int(k_percent * len(attr_abs))
    if k_count == 0:
        return x  # No masking
    
    _, top_indices = torch.topk(attr_abs, k_count)
    
    # Create mask
    mask = torch.zeros_like(attr_flat, dtype=torch.bool)
    mask[top_indices] = True
    mask = mask.view_as(attribution)
    
    # Compute channel-wise means for replacement
    x_masked = x.clone()
    for c in range(x.shape[1]):
        channel_mean = x[:, c].mean()
        x_masked[:, c][mask[:, c]] = channel_mean
    
    return x_masked

def compute_error(model, x, y, device):
    """Compute MSE between model prediction and ground truth."""
    x, y = x.to(device), y.to(device)
    
    with torch.no_grad():
        pred = model(x)
        mse = torch.nn.functional.mse_loss(pred, y)
    
    return mse.item()

def main():
    parser = argparse.ArgumentParser(description='Faithfulness evaluation via top-k ablation')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--split', choices=['val', 'test'], default='val', help='Dataset split')
    parser.add_argument('--method', choices=['ig', 'occlusion'], required=True, help='Attribution method')
    parser.add_argument('--k_list', nargs='+', type=float, default=[0.05, 0.1, 0.2], 
                       help='List of k values (fractions) for top-k ablation')
    parser.add_argument('--n_samples', type=int, default=8, help='Number of samples to evaluate')
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
    
    results_dir = Path(cfg['paths']['results_dir']) / exp_id / 'interp'
    figures_dir = Path('figures') / exp_id / 'interp'
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = ChannelDataset(cfg, args.split, eval_mode=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load model
    model = load_model(cfg, device)
    
    # Prepare results storage
    results = []
    k_values = [0.0] + sorted(args.k_list)  # Include baseline (no masking)
    
    print(f"Evaluating faithfulness for {args.method} on {args.n_samples} samples...")
    print(f"K values: {k_values}")
    
    # Process samples
    sample_count = 0
    for i, (x, y) in enumerate(loader):
        if sample_count >= args.n_samples:
            break
            
        print(f"Processing sample {sample_count + 1}/{args.n_samples}...")
        
        # Compute attribution once per sample
        attribution = compute_attribution(model, x, args.method, device)
        
        # Evaluate for each k value
        for k in k_values:
            if k == 0.0:
                # Baseline: no masking
                x_masked = x
                mask_label = "baseline"
            else:
                # Mask top-k% features
                x_masked = mask_top_k_features(x, attribution, k)
                mask_label = f"top_{k:.0%}"
            
            # Compute error
            error = compute_error(model, x_masked, y, device)
            
            results.append({
                'sample': sample_count,
                'k': k,
                'error': error,
                'method': args.method
            })
        
        sample_count += 1
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save raw results
    csv_path = results_dir / f'faithfulness_{args.method}.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved results to: {csv_path}")
    
    # Compute mean and std across samples
    summary = df.groupby('k')['error'].agg(['mean', 'std']).reset_index()
    
    # Create faithfulness plot
    plt.figure(figsize=(8, 6))
    
    # Plot mean with error bars
    plt.errorbar(summary['k'], summary['mean'], yerr=summary['std'], 
                marker='o', capsize=5, capthick=2, linewidth=2)
    
    plt.xlabel('Fraction of Top Features Masked')
    plt.ylabel('MSE')
    plt.title(f'Faithfulness Curve ({args.method.upper()})')
    plt.grid(True, alpha=0.3)
    
    # Add text annotations
    for _, row in summary.iterrows():
        if row['k'] > 0:  # Skip baseline
            plt.annotate(f'{row["mean"]:.4f}', 
                        xy=(row['k'], row['mean']), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = figures_dir / f'faithfulness_{args.method}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {plot_path}")
    
    # Print summary
    print(f"\nFaithfulness Summary ({args.method}):")
    print("K Value | Mean Error | Std Error")
    print("--------|------------|----------")
    for _, row in summary.iterrows():
        print(f"{row['k']:7.1%} | {row['mean']:10.6f} | {row['std']:9.6f}")
    
    # Compute faithfulness metrics
    baseline_error = summary[summary['k'] == 0.0]['mean'].iloc[0]
    max_k_error = summary[summary['k'] == max(k_values)]['mean'].iloc[0]
    faithfulness_score = (max_k_error - baseline_error) / baseline_error
    
    print(f"\nFaithfulness Score: {faithfulness_score:.3f}")
    print(f"(Higher is better - indicates attribution identifies important features)")

if __name__ == '__main__':
    main()
