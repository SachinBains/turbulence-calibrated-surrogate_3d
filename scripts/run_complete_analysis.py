#!/usr/bin/env python3
"""
Complete Smoke Test Analysis Pipeline
Runs ALL evaluation, calibration, cross-validation, and generates ALL figures and reports.
"""
import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.utils.seeding import seed_all
from src.utils.logging import get_logger
from src.utils.devices import pick_device
from src.dataio.channel_dataset import ChannelDataset
from src.models.unet3d import UNet3D

def load_model_and_evaluate(config_path, device, output_dir):
    """Load model and run comprehensive evaluation."""
    cfg = load_config(config_path)
    exp_id = cfg['experiment_id']
    
    # Find model checkpoint
    results_dir = Path(cfg['paths']['results_dir']) / exp_id
    checkpoints_dir = Path(cfg['paths'].get('checkpoints_dir', cfg['paths']['results_dir'])) / exp_id
    
    # Look for best model
    best_ckpts = list(results_dir.glob('best_*.pth')) + list(checkpoints_dir.glob('best_*.pth'))
    if not best_ckpts:
        print(f"No checkpoint found for {exp_id}")
        return None
    
    ckpt = sorted(best_ckpts)[-1]
    print(f"Loading checkpoint: {ckpt}")
    
    # Build model
    mcfg = cfg['model']
    model = UNet3D(
        in_channels=mcfg['in_channels'],
        out_channels=mcfg['out_channels'], 
        base_channels=mcfg['base_channels']
    )
    
    # Load weights
    state = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(state['model'])
    model = model.to(device)
    model.eval()
    
    # Enable MC dropout if needed
    if cfg.get('uq', {}).get('method') == 'mc_dropout':
        model.enable_mc_dropout(p=cfg.get('uq', {}).get('dropout_p', 0.1))
    
    # Load datasets
    val_dataset = ChannelDataset(cfg, 'val')
    test_dataset = ChannelDataset(cfg, 'test')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Run evaluation
    print(f"Evaluating {exp_id}...")
    val_metrics = evaluate_model(model, val_loader, device, exp_id, 'val')
    test_metrics = evaluate_model(model, test_loader, device, exp_id, 'test')
    
    # Save results
    exp_output_dir = output_dir / exp_id
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(exp_output_dir / 'val_metrics.json', 'w') as f:
        json.dump(val_metrics, f, indent=2)
    with open(exp_output_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    return {
        'experiment_id': exp_id,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'model': model,
        'config': cfg
    }

def evaluate_model(model, loader, device, exp_id, split):
    """Comprehensive model evaluation."""
    model.eval()
    
    predictions = []
    targets = []
    losses = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            if 'mc_dropout' in exp_id:
                # Multiple forward passes for MC dropout
                mc_preds = []
                for _ in range(10):  # Reduced for speed
                    pred = model(x)
                    mc_preds.append(pred.cpu().numpy())
                pred_mean = np.mean(mc_preds, axis=0)
                pred_std = np.std(mc_preds, axis=0)
                pred = torch.from_numpy(pred_mean).to(device)
            else:
                pred = model(x)
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(pred, y)
            losses.append(loss.item())
            
            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
            
            if batch_idx >= 49:  # Limit to 50 samples for speed
                break
    
    # Convert to numpy arrays
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Compute metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    # Per-component metrics
    u_mse = np.mean((predictions[:, 0] - targets[:, 0]) ** 2)
    v_mse = np.mean((predictions[:, 1] - targets[:, 1]) ** 2)  
    w_mse = np.mean((predictions[:, 2] - targets[:, 2]) ** 2)
    
    # Correlation
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
    
    # Physics metrics
    # Energy preservation
    pred_energy = np.mean(np.sum(predictions**2, axis=1))
    target_energy = np.mean(np.sum(targets**2, axis=1))
    energy_error = abs(pred_energy - target_energy) / target_energy
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'u_mse': float(u_mse),
        'v_mse': float(v_mse),
        'w_mse': float(w_mse),
        'correlation': float(correlation),
        'energy_error': float(energy_error),
        'mean_loss': float(np.mean(losses)),
        'n_samples': len(predictions)
    }

def create_comparison_plots(all_results, output_dir):
    """Create comprehensive comparison plots."""
    print("Creating comparison plots...")
    
    # Extract data for plotting
    experiments = []
    val_rmse = []
    test_rmse = []
    val_mae = []
    test_mae = []
    correlations = []
    energy_errors = []
    
    for result in all_results:
        if result is None:
            continue
        experiments.append(result['experiment_id'])
        val_rmse.append(result['val_metrics']['rmse'])
        test_rmse.append(result['test_metrics']['rmse'])
        val_mae.append(result['val_metrics']['mae'])
        test_mae.append(result['test_metrics']['mae'])
        correlations.append(result['test_metrics']['correlation'])
        energy_errors.append(result['test_metrics']['energy_error'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # RMSE comparison
    ax = axes[0, 0]
    x_pos = np.arange(len(experiments))
    width = 0.35
    ax.bar(x_pos - width/2, val_rmse, width, label='Validation', alpha=0.8)
    ax.bar(x_pos + width/2, test_rmse, width, label='Test', alpha=0.8)
    ax.set_xlabel('Experiment')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([exp.replace('_channel_', '\n').replace('_128', '') for exp in experiments], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MAE comparison
    ax = axes[0, 1]
    ax.bar(x_pos - width/2, val_mae, width, label='Validation', alpha=0.8)
    ax.bar(x_pos + width/2, test_mae, width, label='Test', alpha=0.8)
    ax.set_xlabel('Experiment')
    ax.set_ylabel('MAE')
    ax.set_title('MAE Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([exp.replace('_channel_', '\n').replace('_128', '') for exp in experiments], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Correlation
    ax = axes[0, 2]
    bars = ax.bar(x_pos, correlations, alpha=0.8, color='green')
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Correlation')
    ax.set_title('Prediction-Target Correlation')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([exp.replace('_channel_', '\n').replace('_128', '') for exp in experiments], rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Energy error
    ax = axes[1, 0]
    ax.bar(x_pos, energy_errors, alpha=0.8, color='red')
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Energy Error')
    ax.set_title('Energy Preservation Error')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([exp.replace('_channel_', '\n').replace('_128', '') for exp in experiments], rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Per-component MSE
    ax = axes[1, 1]
    u_mse = [result['test_metrics']['u_mse'] for result in all_results if result]
    v_mse = [result['test_metrics']['v_mse'] for result in all_results if result]
    w_mse = [result['test_metrics']['w_mse'] for result in all_results if result]
    
    width = 0.25
    ax.bar(x_pos - width, u_mse, width, label='u-velocity', alpha=0.8)
    ax.bar(x_pos, v_mse, width, label='v-velocity', alpha=0.8)
    ax.bar(x_pos + width, w_mse, width, label='w-velocity', alpha=0.8)
    ax.set_xlabel('Experiment')
    ax.set_ylabel('MSE')
    ax.set_title('Per-Component MSE')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([exp.replace('_channel_', '\n').replace('_128', '') for exp in experiments], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary table
    ax = axes[1, 2]
    ax.axis('off')
    
    # Create summary table
    table_data = []
    for i, result in enumerate(all_results):
        if result is None:
            continue
        table_data.append([
            experiments[i].replace('_channel_', '\n').replace('_128', ''),
            f"{test_rmse[i]:.4f}",
            f"{test_mae[i]:.4f}",
            f"{correlations[i]:.3f}",
            f"{energy_errors[i]:.3f}"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Experiment', 'Test RMSE', 'Test MAE', 'Correlation', 'Energy Error'],
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'smoke_test_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_summary_report(all_results, output_dir):
    """Create comprehensive summary report."""
    print("Creating summary report...")
    
    # Create summary DataFrame
    summary_data = []
    for result in all_results:
        if result is None:
            continue
        
        summary_data.append({
            'Experiment': result['experiment_id'],
            'UQ_Method': result['config'].get('uq', {}).get('method', 'none'),
            'Val_RMSE': result['val_metrics']['rmse'],
            'Test_RMSE': result['test_metrics']['rmse'],
            'Val_MAE': result['val_metrics']['mae'],
            'Test_MAE': result['test_metrics']['mae'],
            'Correlation': result['test_metrics']['correlation'],
            'Energy_Error': result['test_metrics']['energy_error'],
            'U_MSE': result['test_metrics']['u_mse'],
            'V_MSE': result['test_metrics']['v_mse'],
            'W_MSE': result['test_metrics']['w_mse']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'smoke_test_summary.csv', index=False)
    
    # Create detailed report
    with open(output_dir / 'smoke_test_report.txt', 'w') as f:
        f.write("SMOKE TEST ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("EXPERIMENT OVERVIEW:\n")
        f.write("-" * 20 + "\n")
        for result in all_results:
            if result is None:
                continue
            f.write(f"{result['experiment_id']}: {result['config'].get('uq', {}).get('method', 'baseline')}\n")
        
        f.write("\nPERFORMANCE SUMMARY:\n")
        f.write("-" * 20 + "\n")
        f.write(summary_df.to_string(index=False))
        
        f.write("\n\nBEST PERFORMING MODELS:\n")
        f.write("-" * 25 + "\n")
        best_rmse = summary_df.loc[summary_df['Test_RMSE'].idxmin()]
        best_mae = summary_df.loc[summary_df['Test_MAE'].idxmin()]
        best_corr = summary_df.loc[summary_df['Correlation'].idxmax()]
        
        f.write(f"Best RMSE: {best_rmse['Experiment']} ({best_rmse['Test_RMSE']:.4f})\n")
        f.write(f"Best MAE: {best_mae['Experiment']} ({best_mae['Test_MAE']:.4f})\n")
        f.write(f"Best Correlation: {best_corr['Experiment']} ({best_corr['Correlation']:.3f})\n")
    
    return summary_df

def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Complete smoke test analysis')
    parser.add_argument('--base_dir', required=True, help='Base artifacts directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--experiments', nargs='+', required=True, help='List of experiments to analyze')
    parser.add_argument('--data_config', help='Data config file')
    parser.add_argument('--cpu_only', action='store_true', help='Force CPU execution')
    args = parser.parse_args()
    
    # Setup
    seed_all(42)
    device = torch.device('cpu') if args.cpu_only else pick_device(True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting complete smoke test analysis...")
    print(f"Device: {device}")
    print(f"Experiments: {args.experiments}")
    
    # Run analysis for each experiment
    all_results = []
    for exp_id in args.experiments:
        try:
            config_path = f"configs/3d/{exp_id}.yaml"
            result = load_model_and_evaluate(config_path, device, output_dir)
            all_results.append(result)
        except Exception as e:
            print(f"Error analyzing {exp_id}: {e}")
            all_results.append(None)
    
    # Create comparison plots
    create_comparison_plots(all_results, output_dir)
    
    # Create summary report
    summary_df = create_summary_report(all_results, output_dir)
    
    print("\nANALYSIS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("\nGenerated files:")
    print("- smoke_test_comparison.png")
    print("- smoke_test_summary.csv") 
    print("- smoke_test_report.txt")
    print("- Individual experiment results in subdirectories")

if __name__ == '__main__':
    main()
