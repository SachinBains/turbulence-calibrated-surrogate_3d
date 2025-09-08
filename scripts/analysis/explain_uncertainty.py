#!/usr/bin/env python3
"""
Analyze the relationship between predicted uncertainty and prediction errors.
Provides interpretability tools for understanding UQ model behavior.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import r2_score
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def load_uq_predictions(results_dir: Path, method: str, split: str) -> Dict:
    """Load UQ predictions and ground truth."""
    data = {}
    
    # Load mean predictions
    mean_file = results_dir / f'{method}_mean_{split}.npy'
    if mean_file.exists():
        data['y_pred'] = np.load(mean_file)
    
    # Load uncertainty (variance -> std)
    var_file = results_dir / f'{method}_var_{split}.npy'
    if var_file.exists():
        data['uncertainty'] = np.sqrt(np.load(var_file))
    
    # Load ground truth
    gt_candidates = [
        results_dir / f'ground_truth_{split}.npy',
        results_dir / f'y_true_{split}.npy',
        results_dir / f'targets_{split}.npy'
    ]
    
    for gt_file in gt_candidates:
        if gt_file.exists():
            data['y_true'] = np.load(gt_file)
            break
    
    return data

def compute_error_uncertainty_correlation(y_true: np.ndarray, y_pred: np.ndarray, 
                                        uncertainty: np.ndarray) -> Dict[str, float]:
    """Compute correlation between prediction errors and uncertainties."""
    # Compute absolute errors
    abs_errors = np.abs(y_true - y_pred)
    squared_errors = (y_true - y_pred) ** 2
    
    # Flatten arrays
    abs_errors_flat = abs_errors.flatten()
    squared_errors_flat = squared_errors.flatten()
    uncertainty_flat = uncertainty.flatten()
    
    # Remove invalid values
    valid_mask = np.isfinite(abs_errors_flat) & np.isfinite(uncertainty_flat) & (uncertainty_flat > 0)
    abs_errors_flat = abs_errors_flat[valid_mask]
    squared_errors_flat = squared_errors_flat[valid_mask]
    uncertainty_flat = uncertainty_flat[valid_mask]
    
    if len(abs_errors_flat) < 10:
        return {'error': 'Insufficient valid data'}
    
    # Correlations
    pearson_abs, p_abs = stats.pearsonr(abs_errors_flat, uncertainty_flat)
    spearman_abs, p_spear_abs = stats.spearmanr(abs_errors_flat, uncertainty_flat)
    
    # Correlation with squared errors vs predicted variance
    predicted_var = uncertainty_flat ** 2
    pearson_sq, p_sq = stats.pearsonr(squared_errors_flat, predicted_var)
    
    return {
        'pearson_abs_error': float(pearson_abs),
        'pearson_abs_pvalue': float(p_abs),
        'spearman_abs_error': float(spearman_abs),
        'spearman_abs_pvalue': float(p_spear_abs),
        'pearson_squared_error': float(pearson_sq),
        'pearson_squared_pvalue': float(p_sq),
        'n_samples': len(abs_errors_flat)
    }

def uncertainty_error_binning_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                                     uncertainty: np.ndarray, n_bins: int = 10) -> Dict:
    """Analyze error statistics within uncertainty bins."""
    abs_errors = np.abs(y_true - y_pred)
    
    # Flatten arrays
    abs_errors_flat = abs_errors.flatten()
    uncertainty_flat = uncertainty.flatten()
    
    # Remove invalid values
    valid_mask = np.isfinite(abs_errors_flat) & np.isfinite(uncertainty_flat) & (uncertainty_flat > 0)
    abs_errors_flat = abs_errors_flat[valid_mask]
    uncertainty_flat = uncertainty_flat[valid_mask]
    
    if len(abs_errors_flat) < n_bins:
        return {'error': 'Insufficient data for binning'}
    
    # Create uncertainty bins
    uncertainty_percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(uncertainty_flat, uncertainty_percentiles)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_stats = []
    for i in range(n_bins):
        # Find points in this bin
        if i == n_bins - 1:
            in_bin = (uncertainty_flat >= bin_edges[i]) & (uncertainty_flat <= bin_edges[i + 1])
        else:
            in_bin = (uncertainty_flat >= bin_edges[i]) & (uncertainty_flat < bin_edges[i + 1])
        
        if np.sum(in_bin) > 0:
            bin_errors = abs_errors_flat[in_bin]
            bin_uncertainties = uncertainty_flat[in_bin]
            
            stats_dict = {
                'bin_center': float(bin_centers[i]),
                'bin_edges': [float(bin_edges[i]), float(bin_edges[i + 1])],
                'count': int(np.sum(in_bin)),
                'mean_uncertainty': float(np.mean(bin_uncertainties)),
                'mean_error': float(np.mean(bin_errors)),
                'std_error': float(np.std(bin_errors)),
                'median_error': float(np.median(bin_errors)),
                'error_percentiles': {
                    '25': float(np.percentile(bin_errors, 25)),
                    '75': float(np.percentile(bin_errors, 75)),
                    '90': float(np.percentile(bin_errors, 90)),
                    '95': float(np.percentile(bin_errors, 95))
                }
            }
            bin_stats.append(stats_dict)
    
    return {
        'bin_stats': bin_stats,
        'overall_correlation': compute_error_uncertainty_correlation(y_true, y_pred, uncertainty)
    }

def spatial_uncertainty_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                               uncertainty: np.ndarray) -> Dict:
    """Analyze spatial patterns in uncertainty and errors."""
    if len(y_true.shape) != 4:  # (N, C, D, H, W)
        return {'error': 'Expected 4D spatial data'}
    
    N, C, D, H, W = y_true.shape
    results = {}
    
    # Compute errors
    abs_errors = np.abs(y_true - y_pred)
    
    # Average over samples and channels
    mean_uncertainty = np.mean(uncertainty, axis=(0, 1))  # (D, H, W)
    mean_abs_error = np.mean(abs_errors, axis=(0, 1))     # (D, H, W)
    
    # Spatial correlation
    spatial_corr = np.corrcoef(mean_uncertainty.flatten(), mean_abs_error.flatten())[0, 1]
    results['spatial_correlation'] = float(spatial_corr)
    
    # Compute spatial statistics
    results['uncertainty_spatial_stats'] = {
        'mean': float(np.mean(mean_uncertainty)),
        'std': float(np.std(mean_uncertainty)),
        'min': float(np.min(mean_uncertainty)),
        'max': float(np.max(mean_uncertainty))
    }
    
    results['error_spatial_stats'] = {
        'mean': float(np.mean(mean_abs_error)),
        'std': float(np.std(mean_abs_error)),
        'min': float(np.min(mean_abs_error)),
        'max': float(np.max(mean_abs_error))
    }
    
    return results

def plot_uncertainty_error_scatter(y_true: np.ndarray, y_pred: np.ndarray,
                                 uncertainty: np.ndarray, title: str = "",
                                 max_points: int = 10000) -> plt.Figure:
    """Plot scatter of uncertainty vs absolute error."""
    abs_errors = np.abs(y_true - y_pred)
    
    # Flatten and subsample for plotting
    abs_errors_flat = abs_errors.flatten()
    uncertainty_flat = uncertainty.flatten()
    
    # Remove invalid values
    valid_mask = np.isfinite(abs_errors_flat) & np.isfinite(uncertainty_flat)
    abs_errors_flat = abs_errors_flat[valid_mask]
    uncertainty_flat = uncertainty_flat[valid_mask]
    
    # Subsample if too many points
    if len(abs_errors_flat) > max_points:
        idx = np.random.choice(len(abs_errors_flat), max_points, replace=False)
        abs_errors_flat = abs_errors_flat[idx]
        uncertainty_flat = uncertainty_flat[idx]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create 2D histogram for density
    h, xedges, yedges = np.histogram2d(uncertainty_flat, abs_errors_flat, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    im = ax.imshow(h.T, origin='lower', extent=extent, aspect='auto', 
                   cmap='Blues', alpha=0.7)
    plt.colorbar(im, ax=ax, label='Density')
    
    # Overlay scatter plot
    ax.scatter(uncertainty_flat, abs_errors_flat, alpha=0.1, s=1, c='red')
    
    # Add perfect correlation line
    max_val = max(np.max(uncertainty_flat), np.max(abs_errors_flat))
    ax.plot([0, max_val], [0, max_val], 'k--', label='Perfect Correlation', linewidth=2)
    
    # Fit line
    if len(uncertainty_flat) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(uncertainty_flat, abs_errors_flat)
        x_line = np.linspace(0, max_val, 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', 
               label=f'Fit: y={slope:.2f}x+{intercept:.2e} (R²={r_value**2:.3f})', 
               linewidth=2)
    
    ax.set_xlabel('Predicted Uncertainty')
    ax.set_ylabel('Absolute Error')
    ax.set_title(f'Uncertainty vs Error{" - " + title if title else ""}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_uncertainty_error_bins(bin_analysis: Dict, title: str = "") -> plt.Figure:
    """Plot error statistics within uncertainty bins."""
    if 'bin_stats' not in bin_analysis:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No binning data available', ha='center', va='center')
        return fig
    
    bin_stats = bin_analysis['bin_stats']
    
    bin_centers = [b['bin_center'] for b in bin_stats]
    mean_errors = [b['mean_error'] for b in bin_stats]
    std_errors = [b['std_error'] for b in bin_stats]
    counts = [b['count'] for b in bin_stats]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Top plot: Mean error vs uncertainty
    ax1.errorbar(bin_centers, mean_errors, yerr=std_errors, 
                fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax1.plot(bin_centers, bin_centers, 'k--', label='Perfect Calibration', linewidth=2)
    ax1.set_xlabel('Predicted Uncertainty (Bin Center)')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title(f'Error vs Uncertainty Bins{" - " + title if title else ""}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Sample counts per bin
    ax2.bar(range(len(bin_centers)), counts, alpha=0.7)
    ax2.set_xlabel('Uncertainty Bin')
    ax2.set_ylabel('Sample Count')
    ax2.set_title('Samples per Uncertainty Bin')
    ax2.set_xticks(range(len(bin_centers)))
    ax2.set_xticklabels([f'{c:.3f}' for c in bin_centers], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_spatial_uncertainty_maps(y_true: np.ndarray, y_pred: np.ndarray,
                                uncertainty: np.ndarray, title: str = "") -> plt.Figure:
    """Plot spatial maps of uncertainty and errors."""
    if len(y_true.shape) != 4:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'Spatial analysis requires 4D data', ha='center', va='center')
        return fig
    
    # Take middle slice and average over samples/channels
    N, C, D, H, W = y_true.shape
    mid_slice = D // 2
    
    uncertainty_slice = np.mean(uncertainty[:, :, mid_slice, :, :], axis=(0, 1))
    error_slice = np.mean(np.abs(y_true - y_pred)[:, :, mid_slice, :, :], axis=(0, 1))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Uncertainty map
    im1 = axes[0].imshow(uncertainty_slice, cmap='viridis')
    axes[0].set_title('Predicted Uncertainty')
    axes[0].set_xlabel('Width')
    axes[0].set_ylabel('Height')
    plt.colorbar(im1, ax=axes[0])
    
    # Error map
    im2 = axes[1].imshow(error_slice, cmap='plasma')
    axes[1].set_title('Absolute Error')
    axes[1].set_xlabel('Width')
    axes[1].set_ylabel('Height')
    plt.colorbar(im2, ax=axes[1])
    
    # Correlation map (local correlation in patches)
    patch_size = min(8, H//4, W//4)
    corr_map = np.zeros((H//patch_size, W//patch_size))
    
    for i in range(0, H-patch_size, patch_size):
        for j in range(0, W-patch_size, patch_size):
            patch_unc = uncertainty_slice[i:i+patch_size, j:j+patch_size].flatten()
            patch_err = error_slice[i:i+patch_size, j:j+patch_size].flatten()
            
            if len(patch_unc) > 3 and np.std(patch_unc) > 0 and np.std(patch_err) > 0:
                corr_map[i//patch_size, j//patch_size] = np.corrcoef(patch_unc, patch_err)[0, 1]
    
    im3 = axes[2].imshow(corr_map, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2].set_title('Local Correlation')
    axes[2].set_xlabel('Width (patches)')
    axes[2].set_ylabel('Height (patches)')
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle(f'Spatial Analysis{" - " + title if title else ""}')
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Analyze uncertainty-error relationships')
    parser.add_argument('--results_dir', required=True, help='Results directory')
    parser.add_argument('--method', choices=['mc', 'ens'], required=True, 
                       help='UQ method (mc or ens)')
    parser.add_argument('--split', choices=['val', 'test'], default='test',
                       help='Dataset split')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory (default: figures/{exp_id})')
    parser.add_argument('--n_bins', type=int, default=10,
                       help='Number of bins for uncertainty analysis')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        exp_id = results_dir.name
        # Load config to get artifacts_root
        from src.utils.config import load_config
        config_path = results_dir.parent.parent / 'configs' / '3d' / f'{exp_id}.yaml'
        if config_path.exists():
            cfg = load_config(str(config_path))
            output_dir = Path(cfg['paths']['artifacts_root']) / 'figures' / exp_id
        else:
            output_dir = Path('figures') / exp_id
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading UQ predictions from {results_dir}")
    data = load_uq_predictions(results_dir, args.method, args.split)
    
    # Check required data
    required_keys = ['y_pred', 'uncertainty', 'y_true']
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        print(f"Error: Missing required data: {missing_keys}")
        return
    
    y_true = data['y_true']
    y_pred = data['y_pred']
    uncertainty = data['uncertainty']
    
    print(f"Data shapes: y_true={y_true.shape}, y_pred={y_pred.shape}, uncertainty={uncertainty.shape}")
    
    # Compute analyses
    print("Computing error-uncertainty correlations...")
    correlation_results = compute_error_uncertainty_correlation(y_true, y_pred, uncertainty)
    
    print("Performing binning analysis...")
    binning_results = uncertainty_error_binning_analysis(y_true, y_pred, uncertainty, args.n_bins)
    
    print("Computing spatial analysis...")
    spatial_results = spatial_uncertainty_analysis(y_true, y_pred, uncertainty)
    
    # Combine results
    analysis_results = {
        'method': args.method,
        'split': args.split,
        'correlations': correlation_results,
        'binning_analysis': binning_results,
        'spatial_analysis': spatial_results
    }
    
    # Save results
    results_file = output_dir / f'uncertainty_analysis_{args.method}_{args.split}.json'
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"Saved analysis results to {results_file}")
    
    # Print summary
    print(f"\nUncertainty Analysis Summary ({args.method.upper()}, {args.split}):")
    if 'error' not in correlation_results:
        print(f"  Pearson correlation (abs error): {correlation_results['pearson_abs_error']:.3f}")
        print(f"  Spearman correlation (abs error): {correlation_results['spearman_abs_error']:.3f}")
        print(f"  Pearson correlation (squared error): {correlation_results['pearson_squared_error']:.3f}")
        print(f"  Sample count: {correlation_results['n_samples']:,}")
    
    if 'error' not in spatial_results:
        print(f"  Spatial correlation: {spatial_results['spatial_correlation']:.3f}")
    
    # Generate plots
    title_suffix = f"{args.method.upper()} - {args.split}"
    
    print("Generating uncertainty vs error scatter plot...")
    fig1 = plot_uncertainty_error_scatter(y_true, y_pred, uncertainty, title_suffix)
    fig1.savefig(output_dir / f'uncertainty_error_scatter_{args.method}_{args.split}.png', 
                dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    print("Generating binning analysis plot...")
    fig2 = plot_uncertainty_error_bins(binning_results, title_suffix)
    fig2.savefig(output_dir / f'uncertainty_error_bins_{args.method}_{args.split}.png', 
                dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    print("Generating spatial analysis plot...")
    fig3 = plot_spatial_uncertainty_maps(y_true, y_pred, uncertainty, title_suffix)
    fig3.savefig(output_dir / f'spatial_uncertainty_{args.method}_{args.split}.png', 
                dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    print(f"\nUncertainty analysis completed. Results saved to {output_dir}")

if __name__ == '__main__':
    main()
