#!/usr/bin/env python3
"""
Generate calibration plots for all primary models C3D1-C3D6
Based on existing calibration analysis framework
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

from src.eval.uncertainty_calibration import UncertaintyCalibrationDiagnostics
from src.metrics.calibration import (
    reliability_diagram_bins, compute_all_calibration_metrics,
    calibration_slope_intercept
)

def load_model_outputs(results_dir: Path, method: str, split: str = 'test') -> Dict:
    """Load model predictions, targets, and uncertainties."""
    data = {}
    
    # Load mean predictions
    mean_patterns = [
        f'{method}_mean_{split}.npy',
        f'mean_{split}.npy', 
        f'pred_{split}.npy'
    ]
    
    for pattern in mean_patterns:
        mean_file = results_dir / pattern
        if mean_file.exists():
            data['predictions'] = np.load(mean_file)
            print(f"✓ Loaded predictions from: {mean_file}")
            break
    
    # Load uncertainties/variance
    var_patterns = [
        f'{method}_var_{split}.npy',
        f'var_{split}.npy',
        f'{method}_std_{split}.npy',
        f'std_{split}.npy'
    ]
    
    for pattern in var_patterns:
        var_file = results_dir / pattern
        if var_file.exists():
            if 'std' in pattern:
                data['uncertainties'] = np.load(var_file)
            else:
                data['uncertainties'] = np.sqrt(np.load(var_file))
            print(f"✓ Loaded uncertainties from: {var_file}")
            break
    
    # Load ground truth
    gt_patterns = [
        f'ground_truth_{split}.npy',
        f'y_true_{split}.npy',
        f'targets_{split}.npy',
        f'gt_{split}.npy'
    ]
    
    for pattern in gt_patterns:
        gt_file = results_dir / pattern
        if gt_file.exists():
            data['targets'] = np.load(gt_file)
            print(f"✓ Loaded targets from: {gt_file}")
            break
    
    return data

def create_calibration_plot(predictions: np.ndarray, targets: np.ndarray, 
                          uncertainties: np.ndarray, model_name: str, 
                          output_dir: Path) -> Dict:
    """Create calibration plot and compute metrics."""
    
    # Compute prediction errors
    errors = np.abs(predictions - targets)
    
    # Initialize calibration diagnostics
    calibrator = UncertaintyCalibrationDiagnostics()
    
    # Compute reliability diagram
    reliability_data = calibrator.compute_reliability_diagram(uncertainties, errors)
    
    # Compute calibration metrics
    metrics = compute_all_calibration_metrics(uncertainties, errors)
    
    # Create calibration plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability diagram
    bin_confidences = reliability_data['bin_confidences']
    bin_accuracies = reliability_data['bin_accuracies']
    bin_counts = reliability_data['bin_counts']
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax1.scatter(bin_confidences, bin_accuracies, s=bin_counts*10, alpha=0.7, 
                c='blue', label='Observed')
    ax1.set_xlabel('Confidence (Predicted Uncertainty)')
    ax1.set_ylabel('Accuracy (Fraction Correct)')
    ax1.set_title(f'{model_name} - Reliability Diagram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Uncertainty vs Error scatter
    sample_indices = np.random.choice(len(uncertainties), 
                                    min(5000, len(uncertainties)), 
                                    replace=False)
    ax2.scatter(uncertainties[sample_indices], errors[sample_indices], 
                alpha=0.3, s=1)
    ax2.set_xlabel('Predicted Uncertainty')
    ax2.set_ylabel('Prediction Error')
    ax2.set_title(f'{model_name} - Uncertainty vs Error')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(uncertainties, errors)[0, 1]
    ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='white'))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / f'{model_name}_calibration_plots.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved calibration plot: {plot_file}")
    
    # Add correlation to metrics
    metrics['uncertainty_error_correlation'] = correlation
    
    return metrics

def generate_calibration_report(all_metrics: Dict, output_dir: Path):
    """Generate summary calibration report."""
    
    report_file = output_dir / 'calibration_report.md'
    
    with open(report_file, 'w') as f:
        f.write("# Calibration Analysis Report - Primary Models\n\n")
        f.write("## Summary\n\n")
        
        # Create summary table
        f.write("| Model | ECE | MCE | Brier Score | Correlation |\n")
        f.write("|-------|-----|-----|-------------|-------------|\n")
        
        for model_name, metrics in all_metrics.items():
            ece = metrics.get('expected_calibration_error', 'N/A')
            mce = metrics.get('maximum_calibration_error', 'N/A')
            brier = metrics.get('brier_score', 'N/A')
            corr = metrics.get('uncertainty_error_correlation', 'N/A')
            
            f.write(f"| {model_name} | {ece:.4f} | {mce:.4f} | {brier:.4f} | {corr:.3f} |\n")
        
        f.write("\n## Detailed Metrics\n\n")
        
        for model_name, metrics in all_metrics.items():
            f.write(f"### {model_name}\n\n")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"- **{metric_name}**: {value:.6f}\n")
                else:
                    f.write(f"- **{metric_name}**: {value}\n")
            f.write("\n")
    
    print(f"✓ Saved calibration report: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate calibration plots for all primary models")
    parser.add_argument("--artifacts_root", required=True, help="Artifacts root directory")
    parser.add_argument("--output_dir", help="Output directory for plots (default: artifacts_root/figures/calibration)")
    parser.add_argument("--split", default="test", help="Dataset split to analyze")
    args = parser.parse_args()
    
    artifacts_root = Path(args.artifacts_root)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = artifacts_root / 'figures' / 'calibration'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define primary models
    models = {
        'C3D1': {
            'name': 'C3D1 Baseline',
            'results_dir': artifacts_root / 'results' / 'C3D1_channel_baseline_128',
            'method': 'baseline'
        },
        'C3D2': {
            'name': 'C3D2 MC Dropout',
            'results_dir': artifacts_root / 'results' / 'C3D2_channel_mc_dropout_128',
            'method': 'mc'
        },
        'C3D3': {
            'name': 'C3D3 Ensemble',
            'results_dir': artifacts_root / 'results' / 'C3D3_channel_ensemble_128',
            'method': 'ens'
        },
        'C3D4': {
            'name': 'C3D4 Variational',
            'results_dir': artifacts_root / 'results' / 'C3D4_channel_variational_128',
            'method': 'variational'
        },
        'C3D5': {
            'name': 'C3D5 SWAG',
            'results_dir': artifacts_root / 'results' / 'C3D5_channel_swag_128',
            'method': 'swag'
        },
        'C3D6': {
            'name': 'C3D6 Physics-Informed',
            'results_dir': artifacts_root / 'results' / 'C3D6_channel_physics_informed_128',
            'method': 'physics'
        }
    }
    
    all_metrics = {}
    
    print("=== Generating Calibration Plots for Primary Models ===\n")
    
    for model_id, model_info in models.items():
        print(f"Processing {model_info['name']}...")
        
        results_dir = model_info['results_dir']
        if not results_dir.exists():
            print(f"⚠️  Results directory not found: {results_dir}")
            continue
        
        # Load model outputs
        data = load_model_outputs(results_dir, model_info['method'], args.split)
        
        if not all(key in data for key in ['predictions', 'targets', 'uncertainties']):
            print(f"⚠️  Missing required data for {model_id}")
            continue
        
        # Generate calibration plot and compute metrics
        try:
            metrics = create_calibration_plot(
                data['predictions'], 
                data['targets'], 
                data['uncertainties'],
                model_info['name'],
                output_dir
            )
            all_metrics[model_id] = metrics
            print(f"✓ Completed {model_info['name']}\n")
            
        except Exception as e:
            print(f"❌ Error processing {model_id}: {e}\n")
            continue
    
    # Generate summary report
    if all_metrics:
        generate_calibration_report(all_metrics, output_dir)
        
        # Save metrics as JSON
        metrics_file = output_dir / 'calibration_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        print(f"✓ Saved metrics: {metrics_file}")
    
    print(f"\n=== Calibration Analysis Complete ===")
    print(f"Output directory: {output_dir}")
    print(f"Models processed: {len(all_metrics)}")

if __name__ == "__main__":
    main()
