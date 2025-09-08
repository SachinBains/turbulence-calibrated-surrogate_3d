#!/usr/bin/env python3
"""
Plot calibration and reliability diagrams for uncertainty quantification models.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metrics.calibration import (
    reliability_diagram_bins, compute_all_calibration_metrics,
    calibration_slope_intercept
)

def load_predictions(results_dir: Path, method: str, split: str) -> Dict:
    """Load predictions and ground truth from results directory."""
    data = {}
    
    # Load mean predictions
    mean_file = results_dir / f'{method}_mean_{split}.npy'
    if mean_file.exists():
        data['y_pred'] = np.load(mean_file)
    
    # Load variance/uncertainty
    var_file = results_dir / f'{method}_var_{split}.npy'
    if var_file.exists():
        data['uncertainty'] = np.sqrt(np.load(var_file))
    
    # Load conformal intervals if available
    conformal_lo = results_dir / f'{method}_conformal_lo_{split}.npy'
    conformal_hi = results_dir / f'{method}_conformal_hi_{split}.npy'
    if conformal_lo.exists() and conformal_hi.exists():
        data['conformal_lower'] = np.load(conformal_lo)
        data['conformal_upper'] = np.load(conformal_hi)
    
    # Load ground truth (try different naming conventions)
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

def plot_reliability_diagram(y_true: np.ndarray, y_pred: np.ndarray, 
                           uncertainty: np.ndarray, title: str = "",
                           n_bins: int = 10) -> plt.Figure:
    """Plot reliability diagram."""
    rel_data = reliability_diagram_bins(y_true, y_pred, uncertainty, n_bins)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot bars
    bin_centers = rel_data['bin_centers']
    observed = rel_data['observed_freq']
    expected = rel_data['expected_freq']
    bin_counts = rel_data['bin_counts']
    
    # Width of bars
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0
    
    # Plot observed vs expected
    ax.bar(bin_centers, observed, width=bin_width*0.8, alpha=0.7, 
           label='Observed', color='skyblue')
    ax.axhline(y=0.683, color='red', linestyle='--', 
               label='Expected (68.3%)', linewidth=2)
    
    # Add bin counts as text
    for i, (center, obs, count) in enumerate(zip(bin_centers, observed, bin_counts)):
        if count > 0:
            ax.text(center, obs + 0.02, f'n={count}', 
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Predicted Uncertainty')
    ax.set_ylabel('Observed Coverage')
    ax.set_title(f'Reliability Diagram{" - " + title if title else ""}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig

def plot_calibration_scatter(y_true: np.ndarray, y_pred: np.ndarray,
                           uncertainty: np.ndarray, title: str = "") -> plt.Figure:
    """Plot calibration scatter: predicted variance vs squared errors."""
    squared_errors = (y_true - y_pred) ** 2
    predicted_variance = uncertainty ** 2
    
    # Flatten for plotting
    squared_errors_flat = squared_errors.flatten()
    predicted_variance_flat = predicted_variance.flatten()
    
    # Remove invalid values
    valid_mask = np.isfinite(squared_errors_flat) & np.isfinite(predicted_variance_flat)
    squared_errors_flat = squared_errors_flat[valid_mask]
    predicted_variance_flat = predicted_variance_flat[valid_mask]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot with alpha for density
    ax.scatter(predicted_variance_flat, squared_errors_flat, 
              alpha=0.5, s=1, color='blue')
    
    # Perfect calibration line (y = x)
    min_val = min(np.min(predicted_variance_flat), np.min(squared_errors_flat))
    max_val = max(np.max(predicted_variance_flat), np.max(squared_errors_flat))
    ax.plot([min_val, max_val], [min_val, max_val], 
           'r--', label='Perfect Calibration', linewidth=2)
    
    # Fit line
    slope, intercept, r2 = calibration_slope_intercept(y_true, y_pred, uncertainty)
    if not np.isnan(slope):
        x_line = np.linspace(min_val, max_val, 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'g-', 
               label=f'Fit: y={slope:.2f}x+{intercept:.2e} (R²={r2:.3f})', 
               linewidth=2)
    
    ax.set_xlabel('Predicted Variance')
    ax.set_ylabel('Squared Error')
    ax.set_title(f'Calibration Scatter{" - " + title if title else ""}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    return fig

def plot_coverage_vs_confidence(y_true: np.ndarray, y_pred: np.ndarray,
                               uncertainty: np.ndarray, title: str = "") -> plt.Figure:
    """Plot empirical coverage vs confidence level."""
    confidence_levels = np.linspace(0.1, 3.0, 30)  # 0.1 to 3 sigma
    empirical_coverage = []
    
    errors = np.abs(y_true - y_pred)
    
    for conf in confidence_levels:
        coverage = np.mean(errors <= conf * uncertainty)
        empirical_coverage.append(coverage)
    
    # Theoretical coverage for normal distribution
    from scipy.stats import norm
    theoretical_coverage = 2 * norm.cdf(confidence_levels) - 1
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(confidence_levels, empirical_coverage, 'b-', 
           label='Empirical', linewidth=2)
    ax.plot(confidence_levels, theoretical_coverage, 'r--', 
           label='Theoretical (Normal)', linewidth=2)
    
    # Mark common confidence levels
    common_levels = [1.0, 1.96, 2.0]
    for level in common_levels:
        if level <= confidence_levels.max():
            emp_cov = np.interp(level, confidence_levels, empirical_coverage)
            theo_cov = 2 * norm.cdf(level) - 1
            ax.axvline(x=level, color='gray', linestyle=':', alpha=0.5)
            ax.plot(level, emp_cov, 'bo', markersize=8)
            ax.plot(level, theo_cov, 'ro', markersize=8)
    
    ax.set_xlabel('Confidence Level (σ)')
    ax.set_ylabel('Coverage Probability')
    ax.set_title(f'Coverage vs Confidence{" - " + title if title else ""}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig

def plot_interval_coverage(y_true: np.ndarray, lower: np.ndarray, 
                         upper: np.ndarray, title: str = "") -> plt.Figure:
    """Plot prediction intervals with coverage visualization."""
    # Sort by predicted mean for better visualization
    mean_pred = (lower + upper) / 2
    sort_idx = np.argsort(mean_pred.flatten())
    
    # Take subset for plotting (too many points make plot unreadable)
    n_plot = min(200, len(sort_idx))
    plot_idx = sort_idx[::len(sort_idx)//n_plot][:n_plot]
    
    y_true_plot = y_true.flatten()[plot_idx]
    lower_plot = lower.flatten()[plot_idx]
    upper_plot = upper.flatten()[plot_idx]
    mean_plot = mean_pred.flatten()[plot_idx]
    
    # Check coverage
    within_interval = (y_true_plot >= lower_plot) & (y_true_plot <= upper_plot)
    coverage = np.mean(within_interval)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(plot_idx))
    
    # Plot intervals
    ax.fill_between(x, lower_plot, upper_plot, alpha=0.3, color='lightblue', 
                   label=f'Prediction Intervals (Cov: {coverage:.1%})')
    
    # Plot predictions and truth
    ax.plot(x, mean_plot, 'b-', label='Predictions', linewidth=1)
    
    # Color-code points by coverage
    covered = y_true_plot[within_interval]
    uncovered = y_true_plot[~within_interval]
    x_covered = x[within_interval]
    x_uncovered = x[~within_interval]
    
    ax.scatter(x_covered, covered, c='green', s=20, label='Covered', alpha=0.7)
    ax.scatter(x_uncovered, uncovered, c='red', s=20, label='Uncovered', alpha=0.7)
    
    ax.set_xlabel('Sample Index (sorted)')
    ax.set_ylabel('Value')
    ax.set_title(f'Prediction Intervals{" - " + title if title else ""}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Plot calibration diagnostics')
    parser.add_argument('--results_dir', required=True, help='Results directory')
    parser.add_argument('--method', choices=['mc', 'ens'], required=True, 
                       help='UQ method (mc or ens)')
    parser.add_argument('--split', choices=['val', 'test'], default='test',
                       help='Dataset split')
    parser.add_argument('--output_dir', default=None, 
                       help='Output directory (default: figures/{exp_id})')
    parser.add_argument('--n_bins', type=int, default=10,
                       help='Number of bins for reliability diagram')
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
    print(f"Loading predictions from {results_dir}")
    data = load_predictions(results_dir, args.method, args.split)
    
    # Check required data
    required_keys = ['y_pred', 'uncertainty']
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        print(f"Error: Missing required data: {missing_keys}")
        return
    
    if 'y_true' not in data:
        print("Warning: Ground truth not found. Trying to load from dataset...")
        # Could add dataset loading logic here
        return
    
    y_true = data['y_true']
    y_pred = data['y_pred']
    uncertainty = data['uncertainty']
    
    print(f"Data shapes: y_true={y_true.shape}, y_pred={y_pred.shape}, uncertainty={uncertainty.shape}")
    
    # Compute all metrics
    metrics = compute_all_calibration_metrics(
        y_true, y_pred, uncertainty,
        lower=data.get('conformal_lower'),
        upper=data.get('conformal_upper'),
        n_bins=args.n_bins
    )
    
    # Save metrics
    metrics_file = output_dir / f'calibration_metrics_{args.method}_{args.split}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")
    
    # Print key metrics
    print(f"\nCalibration Metrics ({args.method.upper()}, {args.split}):")
    print(f"  ECE: {metrics.get('ece', 'N/A'):.4f}")
    print(f"  MCE: {metrics.get('mce', 'N/A'):.4f}")
    print(f"  Coverage 1σ: {metrics.get('coverage_1sigma', 'N/A'):.3f} (expected: 0.683)")
    print(f"  Coverage 2σ: {metrics.get('coverage_2sigma', 'N/A'):.3f} (expected: 0.954)")
    print(f"  Sharpness: {metrics.get('sharpness', 'N/A'):.4f}")
    
    # Generate plots
    title_suffix = f"{args.method.upper()} - {args.split}"
    
    # 1. Reliability diagram
    print("Generating reliability diagram...")
    fig1 = plot_reliability_diagram(y_true, y_pred, uncertainty, title_suffix, args.n_bins)
    fig1.savefig(output_dir / f'reliability_{args.method}_{args.split}.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Calibration scatter
    print("Generating calibration scatter plot...")
    fig2 = plot_calibration_scatter(y_true, y_pred, uncertainty, title_suffix)
    fig2.savefig(output_dir / f'calibration_scatter_{args.method}_{args.split}.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Coverage vs confidence
    print("Generating coverage vs confidence plot...")
    fig3 = plot_coverage_vs_confidence(y_true, y_pred, uncertainty, title_suffix)
    fig3.savefig(output_dir / f'coverage_confidence_{args.method}_{args.split}.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Interval coverage (if conformal intervals available)
    if 'conformal_lower' in data and 'conformal_upper' in data:
        print("Generating conformal interval plot...")
        fig4 = plot_interval_coverage(y_true, data['conformal_lower'], 
                                    data['conformal_upper'], f"Conformal - {title_suffix}")
        fig4.savefig(output_dir / f'conformal_intervals_{args.method}_{args.split}.png', dpi=150, bbox_inches='tight')
        plt.close(fig4)
        
        print(f"  Conformal Coverage: {metrics.get('interval_coverage', 'N/A'):.3f}")
        print(f"  Avg Interval Width: {metrics.get('avg_interval_width', 'N/A'):.4f}")
    
    print(f"\nAll plots saved to {output_dir}")

if __name__ == '__main__':
    main()
