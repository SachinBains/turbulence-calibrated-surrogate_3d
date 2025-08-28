#!/usr/bin/env python3
"""
Generate publication-ready figures for UQ analysis.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Optional
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set matplotlib style for publication
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def setup_publication_style():
    """Setup matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

def load_experiment_results(results_dir: Path, methods: List[str], splits: List[str]) -> Dict:
    """Load results from multiple methods and splits."""
    results = {}
    
    for method in methods:
        results[method] = {}
        for split in splits:
            split_data = {}
            
            # Load metrics
            metrics_file = results_dir / f'{method}_metrics_{split}.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    split_data['metrics'] = json.load(f)
            
            # Load calibration metrics
            cal_metrics_file = Path('figures') / results_dir.name / f'calibration_metrics_{method}_{split}.json'
            if cal_metrics_file.exists():
                with open(cal_metrics_file, 'r') as f:
                    split_data['calibration'] = json.load(f)
            
            # Load uncertainty analysis
            unc_analysis_file = Path('figures') / results_dir.name / f'uncertainty_analysis_{method}_{split}.json'
            if unc_analysis_file.exists():
                with open(unc_analysis_file, 'r') as f:
                    split_data['uncertainty_analysis'] = json.load(f)
            
            # Load physics validation
            physics_file = Path('figures') / results_dir.name / f'physics_validation_{method}_{split}.json'
            if physics_file.exists():
                with open(physics_file, 'r') as f:
                    split_data['physics'] = json.load(f)
            
            results[method][split] = split_data
    
    return results

def create_performance_comparison(results: Dict, output_dir: Path) -> plt.Figure:
    """Create performance comparison across methods."""
    methods = list(results.keys())
    splits = ['val', 'test']
    
    # Metrics to compare
    metrics = ['rmse_vs_mu', 'nll', 'coverage_1sigma', 'ece']
    metric_labels = ['RMSE', 'NLL', 'Coverage (1σ)', 'ECE']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        
        x_pos = np.arange(len(methods))
        width = 0.35
        
        val_values = []
        test_values = []
        
        for method in methods:
            # Get metric value
            val_val = results[method].get('val', {}).get('metrics', {}).get(metric, np.nan)
            test_val = results[method].get('test', {}).get('metrics', {}).get(metric, np.nan)
            
            # Try calibration metrics if not in main metrics
            if np.isnan(val_val) and 'calibration' in results[method].get('val', {}):
                val_val = results[method]['val']['calibration'].get(metric, np.nan)
            if np.isnan(test_val) and 'calibration' in results[method].get('test', {}):
                test_val = results[method]['test']['calibration'].get(metric, np.nan)
            
            val_values.append(val_val)
            test_values.append(test_val)
        
        # Plot bars
        ax.bar(x_pos - width/2, val_values, width, label='Validation', alpha=0.8)
        ax.bar(x_pos + width/2, test_values, width, label='Test', alpha=0.8)
        
        ax.set_xlabel('Method')
        ax.set_ylabel(label)
        ax.set_title(f'{label} Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.upper() for m in methods])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for j, (val, test) in enumerate(zip(val_values, test_values)):
            if not np.isnan(val):
                ax.text(j - width/2, val + 0.01*max(val_values), f'{val:.3f}', 
                       ha='center', va='bottom', fontsize=8)
            if not np.isnan(test):
                ax.text(j + width/2, test + 0.01*max(test_values), f'{test:.3f}', 
                       ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('UQ Method Performance Comparison')
    plt.tight_layout()
    return fig

def create_calibration_summary(results: Dict, output_dir: Path) -> plt.Figure:
    """Create calibration summary plot."""
    methods = list(results.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Coverage plot
    ax1 = axes[0]
    confidence_levels = [68.3, 95.4, 99.7]
    coverage_keys = ['coverage_1sigma', 'coverage_2sigma', 'coverage_3sigma']
    
    for method in methods:
        test_results = results[method].get('test', {})
        coverages = []
        
        for key in coverage_keys:
            cov = test_results.get('metrics', {}).get(key, np.nan)
            if np.isnan(cov):
                cov = test_results.get('calibration', {}).get(key, np.nan)
            coverages.append(cov * 100 if not np.isnan(cov) else np.nan)
        
        ax1.plot(confidence_levels, coverages, 'o-', label=method.upper(), linewidth=2, markersize=8)
    
    ax1.plot(confidence_levels, confidence_levels, 'k--', label='Perfect Calibration', linewidth=2)
    ax1.set_xlabel('Nominal Coverage (%)')
    ax1.set_ylabel('Empirical Coverage (%)')
    ax1.set_title('Coverage Calibration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(60, 100)
    ax1.set_ylim(60, 100)
    
    # Calibration error comparison
    ax2 = axes[1]
    cal_metrics = ['ece', 'mce', 'sharpness']
    cal_labels = ['ECE', 'MCE', 'Sharpness']
    
    x_pos = np.arange(len(cal_metrics))
    width = 0.35
    
    for i, method in enumerate(methods):
        test_results = results[method].get('test', {})
        values = []
        
        for metric in cal_metrics:
            val = test_results.get('calibration', {}).get(metric, np.nan)
            values.append(val if not np.isnan(val) else 0)
        
        ax2.bar(x_pos + i * width, values, width, label=method.upper(), alpha=0.8)
    
    ax2.set_xlabel('Metric')
    ax2.set_ylabel('Value')
    ax2.set_title('Calibration Metrics')
    ax2.set_xticks(x_pos + width/2)
    ax2.set_xticklabels(cal_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Calibration Analysis Summary')
    plt.tight_layout()
    return fig

def create_uncertainty_correlation_summary(results: Dict, output_dir: Path) -> plt.Figure:
    """Create uncertainty-error correlation summary."""
    methods = list(results.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Correlation comparison
    ax1 = axes[0]
    corr_types = ['pearson_abs_error', 'spearman_abs_error', 'pearson_squared_error']
    corr_labels = ['Pearson (Abs)', 'Spearman (Abs)', 'Pearson (Sq)']
    
    x_pos = np.arange(len(corr_types))
    width = 0.35
    
    for i, method in enumerate(methods):
        test_results = results[method].get('test', {})
        correlations = []
        
        for corr_type in corr_types:
            corr = test_results.get('uncertainty_analysis', {}).get('correlations', {}).get(corr_type, np.nan)
            correlations.append(corr if not np.isnan(corr) else 0)
        
        ax1.bar(x_pos + i * width, correlations, width, label=method.upper(), alpha=0.8)
    
    ax1.set_xlabel('Correlation Type')
    ax1.set_ylabel('Correlation Coefficient')
    ax1.set_title('Uncertainty-Error Correlations')
    ax1.set_xticks(x_pos + width/2)
    ax1.set_xticklabels(corr_labels, rotation=15)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Spatial correlation
    ax2 = axes[1]
    spatial_corrs = []
    method_names = []
    
    for method in methods:
        test_results = results[method].get('test', {})
        spatial_corr = test_results.get('uncertainty_analysis', {}).get('spatial_analysis', {}).get('spatial_correlation', np.nan)
        if not np.isnan(spatial_corr):
            spatial_corrs.append(spatial_corr)
            method_names.append(method.upper())
    
    if spatial_corrs:
        ax2.bar(method_names, spatial_corrs, alpha=0.8)
        ax2.set_xlabel('Method')
        ax2.set_ylabel('Spatial Correlation')
        ax2.set_title('Spatial Uncertainty-Error Correlation')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
    else:
        ax2.text(0.5, 0.5, 'No spatial correlation data', ha='center', va='center', transform=ax2.transAxes)
    
    plt.suptitle('Uncertainty Analysis Summary')
    plt.tight_layout()
    return fig

def create_physics_validation_summary(results: Dict, output_dir: Path) -> plt.Figure:
    """Create physics validation summary."""
    methods = list(results.keys())
    
    # Physics metrics to compare
    physics_metrics = ['divergence_rms', 'kinetic_energy', 'turbulent_ke', 'enstrophy', 'inertial_slope']
    physics_labels = ['Div RMS', 'KE', 'TKE', 'Enstrophy', 'Inertial Slope']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (metric, label) in enumerate(zip(physics_metrics, physics_labels)):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        pred_values = []
        gt_values = []
        method_names = []
        
        for method in methods:
            test_results = results[method].get('test', {})
            physics_data = test_results.get('physics', {})
            
            pred_val = physics_data.get('predictions', {}).get('summary', {}).get(metric, np.nan)
            gt_val = physics_data.get('ground_truth', {}).get('summary', {}).get(metric, np.nan)
            
            if not (np.isnan(pred_val) and np.isnan(gt_val)):
                pred_values.append(pred_val if not np.isnan(pred_val) else 0)
                gt_values.append(gt_val if not np.isnan(gt_val) else 0)
                method_names.append(method.upper())
        
        if pred_values and gt_values:
            x_pos = np.arange(len(method_names))
            width = 0.35
            
            ax.bar(x_pos - width/2, gt_values, width, label='Ground Truth', alpha=0.8)
            ax.bar(x_pos + width/2, pred_values, width, label='Prediction', alpha=0.8)
            
            ax.set_xlabel('Method')
            ax.set_ylabel(label)
            ax.set_title(f'{label}')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(method_names)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No {label} data', ha='center', va='center', transform=ax.transAxes)
    
    # Remove empty subplot
    if len(axes) > len(physics_metrics):
        fig.delaxes(axes[-1])
    
    plt.suptitle('Physics Validation Summary')
    plt.tight_layout()
    return fig

def create_method_comparison_table(results: Dict, output_dir: Path) -> str:
    """Create LaTeX table comparing methods."""
    methods = list(results.keys())
    
    # Key metrics for comparison
    metrics = [
        ('rmse_vs_mu', 'RMSE', '.4f'),
        ('nll', 'NLL', '.4f'),
        ('coverage_1sigma', 'Coverage 1σ', '.3f'),
        ('ece', 'ECE', '.4f'),
        ('pearson_abs_error', 'Corr(σ,|e|)', '.3f')
    ]
    
    latex_table = "\\begin{table}[h]\n"
    latex_table += "\\centering\n"
    latex_table += "\\caption{UQ Method Comparison}\n"
    latex_table += "\\begin{tabular}{l" + "c" * len(methods) + "}\n"
    latex_table += "\\toprule\n"
    latex_table += "Metric & " + " & ".join([m.upper() for m in methods]) + " \\\\\n"
    latex_table += "\\midrule\n"
    
    for metric_key, metric_name, fmt in metrics:
        row = metric_name
        for method in methods:
            test_results = results[method].get('test', {})
            
            # Try different sources for the metric
            value = test_results.get('metrics', {}).get(metric_key, np.nan)
            if np.isnan(value):
                value = test_results.get('calibration', {}).get(metric_key, np.nan)
            if np.isnan(value):
                value = test_results.get('uncertainty_analysis', {}).get('correlations', {}).get(metric_key, np.nan)
            
            if not np.isnan(value):
                row += f" & {value:{fmt}}"
            else:
                row += " & --"
        
        row += " \\\\\n"
        latex_table += row
    
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}\n"
    
    return latex_table

def main():
    parser = argparse.ArgumentParser(description='Generate publication figures')
    parser.add_argument('--results_dir', required=True, help='Results directory')
    parser.add_argument('--methods', nargs='+', default=['mc', 'ens'], 
                       help='UQ methods to include')
    parser.add_argument('--splits', nargs='+', default=['val', 'test'],
                       help='Dataset splits to include')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory (default: figures/{exp_id})')
    args = parser.parse_args()
    
    # Setup publication style
    setup_publication_style()
    
    results_dir = Path(args.results_dir)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        exp_id = results_dir.name
        output_dir = Path('figures') / exp_id
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"Loading results from {results_dir}")
    results = load_experiment_results(results_dir, args.methods, args.splits)
    
    # Generate figures
    print("Generating performance comparison...")
    fig1 = create_performance_comparison(results, output_dir)
    fig1.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    fig1.savefig(output_dir / 'performance_comparison.pdf', bbox_inches='tight')
    plt.close(fig1)
    
    print("Generating calibration summary...")
    fig2 = create_calibration_summary(results, output_dir)
    fig2.savefig(output_dir / 'calibration_summary.png', dpi=300, bbox_inches='tight')
    fig2.savefig(output_dir / 'calibration_summary.pdf', bbox_inches='tight')
    plt.close(fig2)
    
    print("Generating uncertainty correlation summary...")
    fig3 = create_uncertainty_correlation_summary(results, output_dir)
    fig3.savefig(output_dir / 'uncertainty_correlation_summary.png', dpi=300, bbox_inches='tight')
    fig3.savefig(output_dir / 'uncertainty_correlation_summary.pdf', bbox_inches='tight')
    plt.close(fig3)
    
    print("Generating physics validation summary...")
    fig4 = create_physics_validation_summary(results, output_dir)
    fig4.savefig(output_dir / 'physics_validation_summary.png', dpi=300, bbox_inches='tight')
    fig4.savefig(output_dir / 'physics_validation_summary.pdf', bbox_inches='tight')
    plt.close(fig4)
    
    # Generate LaTeX table
    print("Generating comparison table...")
    latex_table = create_method_comparison_table(results, output_dir)
    with open(output_dir / 'method_comparison_table.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"\nPublication figures generated and saved to {output_dir}")
    print("Generated files:")
    print("  - performance_comparison.png/.pdf")
    print("  - calibration_summary.png/.pdf") 
    print("  - uncertainty_correlation_summary.png/.pdf")
    print("  - physics_validation_summary.png/.pdf")
    print("  - method_comparison_table.tex")

if __name__ == '__main__':
    main()
