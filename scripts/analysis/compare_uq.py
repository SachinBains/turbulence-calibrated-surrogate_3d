#!/usr/bin/env python3
"""
Consolidated comparison of ID vs A→B (baseline vs MC dropout).
- Loads test_metrics.json for all exps (runs eval if missing)
- For MC exps, loads mc_metrics_test.json
- Produces CSV and PNG summary tables/plots
"""
import argparse, os, json, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.devices import pick_device
from src.utils.config import load_config
from src.eval.evaluator import evaluate_baseline
from src.dataio.channel_dataset import ChannelDataset
from src.models.unet3d import UNet3D
import torch
from torch.utils.data import DataLoader

def ensure_test_metrics(config_path):
    cfg = load_config(config_path)
    exp_id = cfg['experiment_id']
    results_dir = Path(cfg['paths']['results_dir']) / exp_id
    metrics_path = results_dir / 'test_metrics.json'
    if not metrics_path.exists():
        # Run evaluation
        subprocess.run(['python', '-m', 'scripts.run_eval', '--config', config_path], check=True)
    with open(metrics_path) as f:
        return json.load(f)

def load_mc_metrics(results_dir):
    mc_metrics_path = results_dir / 'mc_metrics_test.json'
    if not mc_metrics_path.exists():
        return {}
    with open(mc_metrics_path) as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_base', required=True)
    parser.add_argument('--id_mc', required=True)
    parser.add_argument('--ood_base', required=True)
    parser.add_argument('--ood_mc', required=True)
    parser.add_argument('--T', type=int, default=32)
    parser.add_argument('--cuda', action='store_true', help='use CUDA if available')
    args = parser.parse_args()
    _ = pick_device(args.cuda)  # for consistency

    configs = {
        'ID-BASE': args.id_base,
        'ID-MC': args.id_mc,
        'OOD-BASE': args.ood_base,
        'OOD-MC': args.ood_mc
    }
    rows = []
    for label, config_path in configs.items():
        metrics = ensure_test_metrics(config_path)
        cfg = load_config(config_path)
        exp_id = cfg['experiment_id']
        results_dir = Path(cfg['paths']['results_dir']) / exp_id
        row = {
            'exp': label,
            'rmse': metrics.get('rmse', np.nan),
            'mae': metrics.get('mae', np.nan),
            'n': metrics.get('n', np.nan)
        }
        if 'MC' in label:
            mc_metrics = load_mc_metrics(results_dir)
            row['nll'] = mc_metrics.get('nll', np.nan)
            row['cov80'] = mc_metrics.get('cov80', np.nan)
            row['cov90'] = mc_metrics.get('cov90', np.nan)
            row['cov95'] = mc_metrics.get('cov95', np.nan)
            row['avg_sigma'] = mc_metrics.get('avg_sigma', np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    summary_dir = Path('results/summary')
    summary_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_dir / 'metrics_id_ood.csv', index=False)

    # Bar plot of RMSE
    fig_dir = Path('figures/summary')
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.bar(df['exp'], df['rmse'], color=['#377eb8','#4daf4a','#e41a1c','#984ea3'])
    plt.ylabel('RMSE')
    plt.title('RMSE: ID vs A→B, Baseline vs MC')
    plt.tight_layout()
    plt.savefig(fig_dir / 'metrics_bars.png', dpi=150)
    plt.close()

    # Coverage plot for MC
    mc_rows = df[df['exp'].str.contains('MC')]
    if not mc_rows.empty:
        plt.figure(figsize=(6,4))
        for cov in ['cov80','cov90','cov95']:
            if cov in mc_rows:
                plt.bar(mc_rows['exp'] + '-' + cov, mc_rows[cov], label=cov)
        plt.ylabel('Coverage')
        plt.title('Coverage vs Nominal (MC exps)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / 'metrics_mc_coverage.png', dpi=150)
        plt.close()

    print(f"Saved summary CSV and plots to {summary_dir} and {fig_dir}")

if __name__ == '__main__':
    main()
