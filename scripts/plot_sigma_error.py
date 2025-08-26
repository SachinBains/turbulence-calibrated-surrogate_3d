#!/usr/bin/env python3
"""
Sigma vs |Error| correlation and scatter plot for MC Dropout UQ experiments.
- Loads mc_mean_{split}.npy and mc_var_{split}.npy from results/{exp}/
- Loads ground truth y from HITDataset (with proper normalization)
- Computes per-voxel |error| and sigma, and computes Pearson/Spearman correlations
- Saves JSON stats and scatter/hist plots under results/ and figures/
"""
import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import torch
from src.utils.config import load_config
from src.dataio.hit_dataset import HITDataset
from src.models.unet3d import UNet3D
from src.uq.mc_dropout import mc_predict
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--T', type=int, default=32)
    parser.add_argument('--max_points', type=int, default=50000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_id = cfg['experiment_id']
    results_dir = Path(cfg['paths']['results_dir']) / exp_id
    figures_dir = Path('figures') / exp_id
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    mean_path = results_dir / f"mc_mean_{args.split}.npy"
    var_path = results_dir / f"mc_var_{args.split}.npy"
    # If MC files missing, run prediction
    if not mean_path.exists() or not var_path.exists():
        print("MC mean/var not found, running MC prediction...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ds = HITDataset(cfg, args.split, eval_mode=True)
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        mcfg = cfg['model']
        net = UNet3D(mcfg['in_channels'], mcfg['out_channels'], base_ch=mcfg['base_channels'])
        ckpts = sorted(results_dir.glob('best_*.pth'))
        assert ckpts, f'No checkpoint found in {results_dir}'
        state = torch.load(ckpts[-1], map_location=device)
        net.load_state_dict(state['model'])
        net = net.to(device)
        net.enable_mc_dropout(cfg['uq'].get('dropout_p', 0.2))
        mc_predict(net, loader, device, T=args.T, save_dir=results_dir, cfg=cfg)

    # Load MC artifacts
    mean = np.load(mean_path)
    var = np.load(var_path)
    sigma = np.sqrt(var)

    # Load GT using HITDataset (de-normalized)
    ds = HITDataset(cfg, args.split, eval_mode=True)
    y_true = []
    for _, y in DataLoader(ds, batch_size=1, shuffle=False):
        y_true.append(y.numpy())
    y_true = np.concatenate(y_true, axis=0)
    # Ensure shapes match
    assert mean.shape == y_true.shape, f"Shape mismatch: mean {mean.shape}, y_true {y_true.shape}"

    abs_err = np.abs(mean - y_true)
    sigma_flat = sigma.flatten()
    abs_err_flat = abs_err.flatten()

    # Downsample for scatter plot
    n_points = min(args.max_points, sigma_flat.size)
    idx = np.random.choice(sigma_flat.size, n_points, replace=False)
    sigma_sample = sigma_flat[idx]
    abs_err_sample = abs_err_flat[idx]

    # Correlations
    pearson_r, _ = pearsonr(sigma_flat, abs_err_flat)
    spearman_rho, _ = spearmanr(sigma_flat, abs_err_flat)
    stats = {
        "pearson_r": float(pearson_r),
        "spearman_rho": float(spearman_rho),
        "n": int(sigma_flat.size)
    }
    with open(results_dir / f"sigma_error_stats_{args.split}.json", 'w') as f:
        json.dump(stats, f, indent=2)

    # Scatter plot
    plt.figure(figsize=(6,6))
    plt.scatter(sigma_sample, abs_err_sample, alpha=0.2, s=1)
    plt.xlabel(r"$\sigma$ (MC std)")
    plt.ylabel(r"$|\mu - y|$")
    plt.title(f"Sigma vs |Error| ({exp_id}, {args.split})")
    plt.tight_layout()
    plt.savefig(figures_dir / f"sigma_error_scatter_{args.split}.png", dpi=150)
    plt.close()

    # Error histogram
    plt.figure()
    plt.hist(abs_err_flat, bins=100, alpha=0.7)
    plt.xlabel(r"$|\mu - y|$")
    plt.ylabel("Count")
    plt.title(f"Error Histogram ({exp_id}, {args.split})")
    plt.tight_layout()
    plt.savefig(figures_dir / f"error_hist_{args.split}.png", dpi=150)
    plt.close()

    # Sigma histogram
    plt.figure()
    plt.hist(sigma_flat, bins=100, alpha=0.7)
    plt.xlabel(r"$\sigma$")
    plt.ylabel("Count")
    plt.title(f"Sigma Histogram ({exp_id}, {args.split})")
    plt.tight_layout()
    plt.savefig(figures_dir / f"sigma_hist_{args.split}.png", dpi=150)
    plt.close()

    print(f"Saved stats and plots to {results_dir} and {figures_dir}")

if __name__ == '__main__':
    main()
