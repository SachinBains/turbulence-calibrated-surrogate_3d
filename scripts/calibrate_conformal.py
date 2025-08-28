#!/usr/bin/env python3
"""
Calibrate conformal prediction quantiles on validation data.
"""
import argparse
import json
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.dataio.hit_dataset import HITDataset
from src.uq.conformal import compute_residuals, fit_conformal

def main():
    parser = argparse.ArgumentParser(description='Calibrate conformal prediction')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--mode', choices=['absolute', 'scaled'], required=True, 
                       help='Conformal mode')
    parser.add_argument('--alpha', type=float, default=0.1, 
                       help='Miscoverage level (default: 0.1 for 90% coverage)')
    parser.add_argument('--base', choices=['mc', 'ens'], required=True,
                       help='Base UQ method (mc or ens)')
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    exp_id = cfg['experiment_id']
    results_dir = Path(cfg['paths']['results_dir']) / exp_id
    
    # Load validation predictions
    if args.base == 'mc':
        mean_path = results_dir / 'mc_mean_val.npy'
        var_path = results_dir / 'mc_var_val.npy'
    elif args.base == 'ens':
        mean_path = results_dir / 'ens_mean_val.npy'
        var_path = results_dir / 'ens_var_val.npy'
    
    if not mean_path.exists():
        raise FileNotFoundError(f"Validation predictions not found: {mean_path}")
    
    pred_mean = np.load(mean_path)
    pred_var = np.load(var_path) if var_path.exists() else None
    
    print(f"Loaded {args.base} validation predictions: {pred_mean.shape}")
    
    # Load ground truth
    val_dataset = HITDataset(cfg, 'val', eval_mode=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    y_true = []
    for _, y in val_loader:
        y_true.append(y.numpy())
    y_true = np.concatenate(y_true, axis=0)
    
    print(f"Loaded ground truth: {y_true.shape}")
    
    # Compute residuals
    residuals = compute_residuals(y_true, pred_mean)
    print(f"Computed residuals: mean={np.mean(residuals):.6f}, std={np.std(residuals):.6f}")
    
    # Fit conformal quantile
    q_alpha = fit_conformal(residuals, args.alpha)
    
    print(f"Conformal quantile (alpha={args.alpha}): {q_alpha:.6f}")
    
    # Save conformal calibration
    conformal_info = {
        'alpha': args.alpha,
        'q_alpha': float(q_alpha),
        'mode': args.mode,
        'base_method': args.base,
        'n_val_samples': len(residuals),
        'residual_stats': {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals))
        }
    }
    
    save_path = results_dir / f'conformal_{args.base}_{args.mode}.json'
    with open(save_path, 'w') as f:
        json.dump(conformal_info, f, indent=2)
    
    print(f"Saved conformal calibration: {save_path}")
    
    # Compute expected coverage
    expected_coverage = 1 - args.alpha
    print(f"Expected coverage: {expected_coverage:.1%}")

if __name__ == '__main__':
    main()
