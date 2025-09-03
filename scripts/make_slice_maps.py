#!/usr/bin/env python3
"""
Quick slice maps for baseline or MC experiments.
- For baseline: run a deterministic forward pass for prediction μ
- For MC: load mc_mean_{split}.npy and mc_var_{split}.npy for μ and σ
- Save a 2x2 matplotlib figure with truth, pred (μ), |err|, σ (if available)
"""
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from src.utils.devices import pick_device
from src.utils.config import load_config
from src.dataio.channel_dataset import ChannelDataset
from src.models.unet3d import UNet3D
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--axis', choices=['x','y','z'], default='z')
    parser.add_argument('--slice_idx', default='center')
    parser.add_argument('--cuda', action='store_true', help='use CUDA if available')
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_id = cfg['experiment_id']
    results_dir = Path(cfg['paths']['results_dir']) / exp_id
    figures_dir = Path(cfg['paths']['artifacts_root']) / 'figures' / exp_id
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load GT and prediction
    ds = ChannelDataset(cfg, args.split, eval_mode=True)
    y_true = []
    for _, y in DataLoader(ds, batch_size=1, shuffle=False):
        y_true.append(y.numpy())
    y_true = np.concatenate(y_true, axis=0)
    y = y_true[args.index,0] if y_true.shape[1] == 1 else y_true[args.index]

    # Determine if MC or baseline
    is_mc = cfg['uq'].get('method','none').startswith('mc')
    mean_path = results_dir / f"mc_mean_{args.split}.npy"
    var_path = results_dir / f"mc_var_{args.split}.npy"
    if is_mc and mean_path.exists() and var_path.exists():
        mu = np.load(mean_path)[args.index,0] if y_true.shape[1]==1 else np.load(mean_path)[args.index]
        sigma = np.sqrt(np.load(var_path)[args.index,0] if y_true.shape[1]==1 else np.load(var_path)[args.index])
    else:
        # Baseline: run deterministic forward pass
        device = pick_device(args.cuda)
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        mcfg = cfg['model']
        net = UNet3D(mcfg['in_channels'], mcfg['out_channels'], base_ch=mcfg['base_channels'])
        ckpts = sorted(results_dir.glob('best_*.pth'))
        assert ckpts, f'No checkpoint found in {results_dir}'
        state = torch.load(ckpts[-1], map_location=device)
        net.load_state_dict(state['model'])
        net = net.to(device)
        net.eval()
        with torch.no_grad():
            for i, (X, _) in enumerate(loader):
                if i == args.index:
                    mu = net(X.to(device)).cpu().numpy()[0,0] if y_true.shape[1]==1 else net(X.to(device)).cpu().numpy()[0]
                    sigma = None
                    break

    err = np.abs(mu - y)
    # Determine slice index
    axis_dict = {'x':0,'y':1,'z':2}
    ax = axis_dict[args.axis]
    shape = y.shape
    if args.slice_idx == 'center':
        k = shape[ax] // 2
    else:
        k = int(args.slice_idx)

    def get_slice(arr):
        if ax==0: return arr[0,k,:,:]  # Select first velocity component
        if ax==1: return arr[0,:,k,:]
        if ax==2: return arr[0,:,:,k]
    y_slice = get_slice(y)
    mu_slice = get_slice(mu)
    err_slice = get_slice(err)
    sigma_slice = get_slice(sigma) if sigma is not None else None

    # Plot
    plt.figure(figsize=(8,8))
    plt.subplot(2,2,1)
    plt.imshow(y_slice, cmap='viridis'); plt.title('Truth')
    plt.subplot(2,2,2)
    plt.imshow(mu_slice, cmap='viridis'); plt.title('Pred (μ)')
    plt.subplot(2,2,3)
    plt.imshow(err_slice, cmap='magma'); plt.title('|Error|')
    plt.subplot(2,2,4)
    if sigma_slice is not None:
        plt.imshow(sigma_slice, cmap='cividis'); plt.title('σ (MC std)')
    else:
        plt.axis('off')
    plt.tight_layout()
    fname = f"slice_{args.split}_{args.axis}{k}_idx{args.index}.png"
    plt.savefig(figures_dir / fname, dpi=150)
    plt.close()
    print(f"Saved {figures_dir / fname}")

if __name__ == '__main__':
    main()
