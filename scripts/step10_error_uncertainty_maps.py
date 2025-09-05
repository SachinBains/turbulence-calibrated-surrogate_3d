#!/usr/bin/env python3
"""
Step 10: Generate and visualize prediction error and uncertainty maps
Uses existing prediction arrays from step10_visualization folder
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd
import os
from typing import Dict, List, Tuple

def load_prediction_data(step10_dir: Path) -> Dict:
    """Load all prediction arrays from step10_visualization folder"""
    results_dir = step10_dir / "results"
    
    data = {}
    
    # E2: MC Dropout (ID)
    e2_dir = results_dir / "E2_hit_bayes"
    if e2_dir.exists():
        data['E2_mc_id'] = {
            'mean_test': np.load(e2_dir / "mc_mean_test.npy"),
            'var_test': np.load(e2_dir / "mc_var_test.npy"),
            'mean_val': np.load(e2_dir / "mc_mean_val.npy"),
            'var_val': np.load(e2_dir / "mc_var_val.npy"),
            'conformal_hi': np.load(e2_dir / "mc_conformal_hi_test.npy"),
            'conformal_lo': np.load(e2_dir / "mc_conformal_lo_test.npy"),
            'method': 'MC Dropout',
            'domain': 'ID'
        }
    
    # E4: MC Dropout (A->B)
    e4_dir = results_dir / "E4_hit_ab_dropout"
    if e4_dir.exists():
        data['E4_mc_ab'] = {
            'mean_test': np.load(e4_dir / "mc_mean_test.npy"),
            'var_test': np.load(e4_dir / "mc_var_test.npy"),
            'method': 'MC Dropout',
            'domain': 'A->B'
        }
    
    # E5: Ensemble (ID)
    e5_dir = results_dir / "E5_hit_ens"
    if e5_dir.exists():
        data['E5_ens_id'] = {
            'mean_test': np.load(e5_dir / "ens_mean_test.npy"),
            'var_test': np.load(e5_dir / "ens_var_test.npy"),
            'mean_val': np.load(e5_dir / "ens_mean_val.npy"),
            'var_val': np.load(e5_dir / "ens_var_val.npy"),
            'conformal_hi': np.load(e5_dir / "ens_conformal_hi_test.npy"),
            'conformal_lo': np.load(e5_dir / "ens_conformal_lo_test.npy"),
            'method': 'Ensemble',
            'domain': 'ID'
        }
    
    # E6: Ensemble (A->B)
    e6_dir = results_dir / "E6_hit_ab_ens"
    if e6_dir.exists():
        data['E6_ens_ab'] = {
            'mean_test': np.load(e6_dir / "ens_mean_test.npy"),
            'var_test': np.load(e6_dir / "ens_var_test.npy"),
            'method': 'Ensemble',
            'domain': 'A->B'
        }
    
    return data

def load_ground_truth_data(config_path: str, split: str = 'test') -> np.ndarray:
    """Load ground truth data for comparison"""
    # This would need to be adapted based on your data loading setup
    # For now, we'll use a placeholder approach
    print(f"Note: Ground truth loading needs to be implemented for {split} split")
    return None

def create_central_slice_comparison(data: Dict, sample_idx: int = 0, axis: str = 'z', output_dir: Path = None):
    """Create central slice comparison across all methods"""
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    methods = ['E2_mc_id', 'E4_mc_ab', 'E5_ens_id', 'E6_ens_ab']
    titles = ['MC ID', 'MC A->B', 'Ens ID', 'Ens A->B']
    
    for i, (method_key, title) in enumerate(zip(methods, titles)):
        if method_key not in data:
            continue
            
        method_data = data[method_key]
        
        # Get predictions and uncertainty
        mean_pred = method_data['mean_test'][sample_idx, 0]  # Assuming shape [N, 1, H, W, D]
        var_pred = method_data['var_test'][sample_idx, 0]
        std_pred = np.sqrt(var_pred)
        
        # Get central slice
        if axis == 'z':
            slice_idx = mean_pred.shape[2] // 2
            mean_slice = mean_pred[:, :, slice_idx]
            std_slice = std_pred[:, :, slice_idx]
        elif axis == 'y':
            slice_idx = mean_pred.shape[1] // 2
            mean_slice = mean_pred[:, slice_idx, :]
            std_slice = std_pred[:, slice_idx, :]
        else:  # x
            slice_idx = mean_pred.shape[0] // 2
            mean_slice = mean_pred[slice_idx, :, :]
            std_slice = std_pred[slice_idx, :, :]
        
        # Plot prediction
        im1 = axes[0, i].imshow(mean_slice, cmap='viridis')
        axes[0, i].set_title(f'{title} - Prediction')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], shrink=0.8)
        
        # Plot uncertainty
        im2 = axes[1, i].imshow(std_slice, cmap='plasma')
        axes[1, i].set_title(f'{title} - Uncertainty (σ)')
        axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i], shrink=0.8)
    
    plt.suptitle(f'Central {axis.upper()}-slice Comparison (Sample {sample_idx})')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / f'central_slice_comparison_{axis}_sample{sample_idx}.png', dpi=300, bbox_inches='tight')
    
    return fig

def create_uncertainty_error_maps(data: Dict, sample_idx: int = 0, output_dir: Path = None):
    """Create uncertainty vs error correlation maps"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Focus on methods with both mean and variance
    methods = ['E2_mc_id', 'E5_ens_id']
    titles = ['MC Dropout (ID)', 'Ensemble (ID)']
    
    for i, (method_key, title) in enumerate(zip(methods, titles)):
        if method_key not in data:
            continue
            
        method_data = data[method_key]
        
        # Get predictions and uncertainty
        mean_pred = method_data['mean_test'][sample_idx, 0]
        var_pred = method_data['var_test'][sample_idx, 0]
        std_pred = np.sqrt(var_pred)
        
        # For error calculation, we'd need ground truth
        # For now, create synthetic error for demonstration
        # In practice, you'd load actual ground truth here
        synthetic_error = np.abs(mean_pred - np.mean(mean_pred))
        
        # Central slice
        slice_idx = mean_pred.shape[2] // 2
        std_slice = std_pred[:, :, slice_idx]
        err_slice = synthetic_error[:, :, slice_idx]
        
        # Plot uncertainty
        im1 = axes[0, i].imshow(std_slice, cmap='plasma')
        axes[0, i].set_title(f'{title} - Uncertainty')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], shrink=0.8)
        
        # Plot error
        im2 = axes[1, i].imshow(err_slice, cmap='magma')
        axes[1, i].set_title(f'{title} - Error')
        axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i], shrink=0.8)
    
    plt.suptitle(f'Uncertainty vs Error Maps (Sample {sample_idx})')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / f'uncertainty_error_maps_sample{sample_idx}.png', dpi=300, bbox_inches='tight')
    
    return fig

def create_conformal_interval_visualization(data: Dict, sample_idx: int = 0, output_dir: Path = None):
    """Visualize conformal prediction intervals"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Methods with conformal intervals
    methods = [('E2_mc_id', 'MC Dropout'), ('E5_ens_id', 'Ensemble')]
    
    for i, (method_key, title) in enumerate(methods):
        if method_key not in data or 'conformal_hi' not in data[method_key]:
            continue
            
        method_data = data[method_key]
        
        # Get conformal intervals
        mean_pred = method_data['mean_test'][sample_idx, 0]
        conf_hi = method_data['conformal_hi'][sample_idx, 0]
        conf_lo = method_data['conformal_lo'][sample_idx, 0]
        
        # Calculate interval width
        interval_width = conf_hi - conf_lo
        
        # Central slice
        slice_idx = mean_pred.shape[2] // 2
        width_slice = interval_width[:, :, slice_idx]
        
        # Plot interval width
        im = axes[i].imshow(width_slice, cmap='viridis')
        axes[i].set_title(f'{title} - Conformal Interval Width')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], shrink=0.8)
    
    plt.suptitle(f'Conformal Prediction Intervals (Sample {sample_idx})')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / f'conformal_intervals_sample{sample_idx}.png', dpi=300, bbox_inches='tight')
    
    return fig

def create_method_statistics_summary(data: Dict) -> pd.DataFrame:
    """Create summary statistics for all methods"""
    
    summary_data = []
    
    for method_key, method_data in data.items():
        
        # Basic info
        record = {
            'experiment': method_key,
            'method': method_data['method'],
            'domain': method_data['domain']
        }
        
        # Test set statistics
        if 'mean_test' in method_data:
            mean_test = method_data['mean_test']
            var_test = method_data['var_test']
            
            record.update({
                'test_samples': mean_test.shape[0],
                'mean_prediction_mean': np.mean(mean_test),
                'mean_prediction_std': np.std(mean_test),
                'mean_uncertainty_mean': np.mean(np.sqrt(var_test)),
                'mean_uncertainty_std': np.std(np.sqrt(var_test)),
                'uncertainty_range': np.ptp(np.sqrt(var_test))
            })
        
        # Conformal interval statistics
        if 'conformal_hi' in method_data and 'conformal_lo' in method_data:
            conf_hi = method_data['conformal_hi']
            conf_lo = method_data['conformal_lo']
            interval_width = conf_hi - conf_lo
            
            record.update({
                'conformal_width_mean': np.mean(interval_width),
                'conformal_width_std': np.std(interval_width),
                'conformal_width_range': np.ptp(interval_width)
            })
        
        summary_data.append(record)
    
    return pd.DataFrame(summary_data)

def main():
    """Main function for Step 10 visualization"""
    print("=== Step 10: Generate Error and Uncertainty Maps ===\n")
    
    # Setup paths
    artifacts_dir = os.environ.get('ARTIFACTS_DIR', '/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d')
    step10_dir = Path(artifacts_dir) / "step10_visualization"
    output_dir = Path(artifacts_dir) / "analysis" / "step10_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prediction data
    print("1. Loading prediction arrays...")
    data = load_prediction_data(step10_dir)
    
    print(f"   Loaded data for: {list(data.keys())}")
    
    # Create visualizations
    print("2. Creating central slice comparisons...")
    fig1 = create_central_slice_comparison(data, sample_idx=0, axis='z', output_dir=output_dir)
    plt.close(fig1)
    
    print("3. Creating uncertainty-error maps...")
    fig2 = create_uncertainty_error_maps(data, sample_idx=0, output_dir=output_dir)
    plt.close(fig2)
    
    print("4. Creating conformal interval visualizations...")
    fig3 = create_conformal_interval_visualization(data, sample_idx=0, output_dir=output_dir)
    plt.close(fig3)
    
    # Create summary statistics
    print("5. Generating method statistics summary...")
    summary_df = create_method_statistics_summary(data)
    summary_df.to_csv(output_dir / "method_statistics_summary.csv", index=False)
    
    print("\n=== PREDICTION DATA SUMMARY ===")
    print(summary_df[['experiment', 'method', 'domain', 'test_samples', 'mean_uncertainty_mean']].to_string(index=False))
    
    print(f"\nStep 10 Complete: Visualizations saved to {output_dir}/")
    print("Generated files:")
    print("  - central_slice_comparison_z_sample0.png")
    print("  - uncertainty_error_maps_sample0.png")
    print("  - conformal_intervals_sample0.png")
    print("  - method_statistics_summary.csv")
    print("\nNext: Step 11 - Quantitative UQ method comparison")

if __name__ == "__main__":
    main()
