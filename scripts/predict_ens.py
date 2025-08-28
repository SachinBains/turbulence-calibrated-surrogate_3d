#!/usr/bin/env python3
"""
Ensemble prediction for uncertainty quantification.
"""
import argparse
import json
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.devices import pick_device
from src.dataio.hit_dataset import HITDataset
from src.uq.ensembles import ensemble_predict, compute_ensemble_metrics

def main():
    parser = argparse.ArgumentParser(description='Ensemble prediction')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--split', choices=['val', 'test'], required=True, help='Dataset split')
    parser.add_argument('--save_dir', default=None, help='Save directory (default: results/{exp_id})')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--conformal', choices=['absolute', 'scaled'], default=None, help='Apply conformal prediction')
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    device = pick_device(args.cuda)
    
    # Setup paths
    exp_id = cfg['experiment_id']
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path(cfg['paths']['results_dir']) / exp_id
    
    # Load ensemble info
    ensemble_info_path = save_dir / 'ensemble_info.json'
    if not ensemble_info_path.exists():
        raise FileNotFoundError(f"Ensemble info not found: {ensemble_info_path}")
    
    with open(ensemble_info_path, 'r') as f:
        ensemble_info = json.load(f)
    
    member_paths = [Path(p) for p in ensemble_info['member_paths']]
    print(f"Loading {len(member_paths)} ensemble members")
    
    # Load dataset
    dataset = HITDataset(cfg, args.split, eval_mode=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Running ensemble prediction on {len(dataset)} {args.split} samples...")
    
    # Run ensemble prediction
    ens_mean, ens_var, all_predictions = ensemble_predict(member_paths, dataloader, device, cfg)
    
    # Get ground truth for metrics
    y_true = []
    for _, y in dataloader:
        y_true.append(y.numpy())
    y_true = np.concatenate(y_true, axis=0)
    
    # Compute metrics
    metrics = compute_ensemble_metrics(ens_mean, y_true)

    # Apply conformal prediction if requested
    if args.conformal:
        from src.uq.conformal import apply_conformal, compute_coverage_metrics
        
        # Load conformal calibration
        conformal_path = save_dir / f'conformal_ens_{args.conformal}.json'
        if conformal_path.exists():
            with open(conformal_path, 'r') as f:
                conformal_info = json.load(f)
            
            q_alpha = conformal_info['q_alpha']
            base_sigma = np.sqrt(ens_var)
            
            # Apply conformal intervals
            lo, hi = apply_conformal(ens_mean, base_sigma, q_alpha, mode=args.conformal)
            
            # Save conformal predictions
            np.save(save_dir / f'ens_conformal_lo_{args.split}.npy', lo)
            np.save(save_dir / f'ens_conformal_hi_{args.split}.npy', hi)
            
            # Add conformal coverage metrics
            conformal_metrics = compute_coverage_metrics(y_true, lo, hi)
            metrics['overall'].update({
                'conformal_coverage': conformal_metrics['coverage'],
                'conformal_width': conformal_metrics['avg_width']
            })
            
            print(f"Applied conformal prediction ({args.conformal} mode)")
        else:
            print(f"Warning: Conformal calibration not found: {conformal_path}")

    # Save predictions
    mean_path = save_dir / f'ens_mean_{args.split}.npy'
    var_path = save_dir / f'ens_var_{args.split}.npy'
    metrics_path = save_dir / f'ens_metrics_{args.split}.json'

    np.save(mean_path, ens_mean)
    np.save(var_path, ens_var)

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Saved ensemble predictions:")
    print(f"  Mean: {mean_path}")
    print(f"  Variance: {var_path}")
    print(f"  Metrics: {metrics_path}")
    
    # Print summary
    overall = metrics['overall']
    print(f"\nEnsemble Results ({args.split}):")
    print(f"  RMSE: {overall['rmse']:.6f}")
    print(f"  MAE: {overall['mae']:.6f}")
    print(f"  Samples: {overall['n_samples']}")
    print(f"  Mean uncertainty: {np.mean(ens_var):.6f}")

if __name__ == '__main__':
    main()
