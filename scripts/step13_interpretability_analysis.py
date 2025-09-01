#!/usr/bin/env python3
"""
Step 13: Interpretability and Feature Analysis
Analyzes prediction patterns and spatial characteristics to understand
what features different UQ methods focus on for turbulence prediction.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal, ndimage
from typing import Dict, List, Tuple, Any

def analyze_prediction_patterns(prediction: np.ndarray, method_name: str) -> Dict[str, Any]:
    """Analyze spatial patterns and characteristics in predictions"""
    
    # Remove batch dimension if present
    if prediction.ndim == 4 and prediction.shape[0] == 1:
        pred = prediction[0]  # Shape: (32, 32, 32)
    else:
        pred = prediction
    
    results = {}
    
    # Basic statistics
    results['mean_value'] = float(np.mean(pred))
    results['std_value'] = float(np.std(pred))
    results['min_value'] = float(np.min(pred))
    results['max_value'] = float(np.max(pred))
    
    # Spatial gradients (measure of local variation)
    grad_x = np.gradient(pred, axis=0)
    grad_y = np.gradient(pred, axis=1)
    grad_z = np.gradient(pred, axis=2)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
    results['gradient_mean'] = float(np.mean(gradient_magnitude))
    results['gradient_std'] = float(np.std(gradient_magnitude))
    results['gradient_max'] = float(np.max(gradient_magnitude))
    
    # Spatial autocorrelation (measure of spatial structure)
    center_slice = pred[pred.shape[0]//2]
    autocorr = signal.correlate(center_slice, center_slice, mode='same')
    autocorr_normalized = autocorr / autocorr.max()
    
    results['spatial_autocorr_peak'] = float(autocorr_normalized.max())
    results['spatial_autocorr_width'] = float(np.sum(autocorr_normalized > 0.5))
    
    # Energy distribution across scales
    fft_pred = np.fft.fftn(pred)
    power_spectrum = np.abs(fft_pred)**2
    
    # Radial average of power spectrum
    center = np.array(pred.shape) // 2
    y, x, z = np.ogrid[:pred.shape[0], :pred.shape[1], :pred.shape[2]]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2 + (z - center[2])**2)
    
    # Bin by radius
    r_bins = np.arange(0, min(center) + 1)
    power_radial = []
    for i in range(len(r_bins)-1):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        if np.any(mask):
            power_radial.append(np.mean(power_spectrum[mask]))
        else:
            power_radial.append(0)
    
    results['power_spectrum'] = power_radial
    results['spectral_peak_freq'] = float(np.argmax(power_radial[1:]) + 1)  # Skip DC component
    results['spectral_energy_ratio'] = float(np.sum(power_radial[1:5]) / np.sum(power_radial[1:]))
    
    # Feature localization analysis
    # Find regions of high activity
    pred_abs = np.abs(pred)
    threshold_95 = np.percentile(pred_abs, 95)
    high_activity_mask = pred_abs > threshold_95
    
    results['high_activity_fraction'] = float(np.mean(high_activity_mask))
    results['high_activity_clusters'] = int(ndimage.label(high_activity_mask)[1])
    
    # Spatial moments (center of mass, spread)
    total_mass = np.sum(pred_abs)
    if total_mass > 0:
        coords = np.mgrid[:pred.shape[0], :pred.shape[1], :pred.shape[2]]
        center_of_mass = [float(np.sum(coords[i] * pred_abs) / total_mass) for i in range(3)]
        results['center_of_mass'] = center_of_mass
        
        # Spread around center of mass
        spread = np.sum(pred_abs * np.sum([(coords[i] - center_of_mass[i])**2 for i in range(3)], axis=0)) / total_mass
        results['spatial_spread'] = float(spread)
    else:
        results['center_of_mass'] = [16.0, 16.0, 16.0]  # Center of 32x32x32 grid
        results['spatial_spread'] = 0.0
    
    return results

def create_prediction_pattern_analysis(
    interp_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
    sample_idx: int
):
    """Create plots analyzing prediction patterns"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data for plotting
    methods = list(interp_results.keys())
    
    # Plot 1: Gradient magnitude comparison
    ax = axes[0, 0]
    gradient_means = [interp_results[m]['gradient_mean'] for m in methods]
    gradient_stds = [interp_results[m]['gradient_std'] for m in methods]
    
    x_pos = np.arange(len(methods))
    ax.bar(x_pos, gradient_means, yerr=gradient_stds, capsize=5, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45)
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title('Spatial Gradient Analysis')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Value distribution comparison
    ax = axes[0, 1]
    means = [interp_results[m]['mean_value'] for m in methods]
    stds = [interp_results[m]['std_value'] for m in methods]
    
    ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='orange')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45)
    ax.set_ylabel('Prediction Value')
    ax.set_title('Prediction Statistics')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Spatial autocorrelation
    ax = axes[1, 0]
    autocorr_peaks = [interp_results[m]['spatial_autocorr_peak'] for m in methods]
    autocorr_widths = [interp_results[m]['spatial_autocorr_width'] for m in methods]
    
    ax.scatter(autocorr_peaks, autocorr_widths, s=100, alpha=0.7)
    for i, method in enumerate(methods):
        ax.annotate(method, (autocorr_peaks[i], autocorr_widths[i]), 
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Autocorrelation Peak')
    ax.set_ylabel('Autocorrelation Width')
    ax.set_title('Spatial Structure Analysis')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Power spectrum comparison
    ax = axes[1, 1]
    for method in methods:
        power_spec = interp_results[method]['power_spectrum']
        freqs = np.arange(len(power_spec))
        ax.loglog(freqs[1:], power_spec[1:], label=method, marker='o', markersize=4)
    
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')
    ax.set_title('Power Spectrum Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'prediction_pattern_analysis_sample{sample_idx}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

def create_spatial_analysis(
    interp_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
    sample_idx: int
):
    """Create spatial analysis plots"""
    
    n_methods = len(interp_results)
    fig, axes = plt.subplots(n_methods, 2, figsize=(12, 5*n_methods))
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    
    for i, (method, results) in enumerate(interp_results.items()):
        # Plot power spectrum
        ax = axes[i, 0]
        power_spec = results['power_spectrum']
        freqs = np.arange(len(power_spec))
        ax.loglog(freqs[1:], power_spec[1:], 'o-')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Power')
        ax.set_title(f'{method}\nPower Spectrum')
        ax.grid(True, alpha=0.3)
        
        # Plot spatial characteristics summary
        ax = axes[i, 1]
        metrics = ['gradient_mean', 'spatial_autocorr_peak', 'spectral_energy_ratio', 'high_activity_fraction']
        values = [results[m] for m in metrics]
        
        # Normalize values for comparison
        if max(values) > min(values):
            normalized_values = [(v - min(values)) / (max(values) - min(values)) for v in values]
        else:
            normalized_values = [0.5] * len(values)
        
        bars = ax.bar(metrics, normalized_values, alpha=0.7)
        ax.set_ylabel('Normalized Value')
        ax.set_title(f'{method}\nSpatial Characteristics')
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'spatial_analysis_sample{sample_idx}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

def create_interpretability_summary(
    interp_results: Dict[str, Dict[str, Any]],
    output_dir: Path
) -> pd.DataFrame:
    """Create summary table of interpretability metrics"""
    
    summary_data = []
    
    for method, results in interp_results.items():
        summary_data.append({
            'Method': method,
            'Mean_Gradient': f"{results['gradient_mean']:.4f}",
            'Spatial_Autocorr': f"{results['spatial_autocorr_peak']:.3f}",
            'Spectral_Peak_Freq': f"{results['spectral_peak_freq']:.1f}",
            'Energy_Ratio': f"{results['spectral_energy_ratio']:.3f}",
            'High_Activity_Fraction': f"{results['high_activity_fraction']:.3f}",
            'Activity_Clusters': f"{results['high_activity_clusters']:.0f}",
            'Spatial_Spread': f"{results['spatial_spread']:.2f}",
            'Prediction_Range': f"{results['max_value'] - results['min_value']:.4f}",
            'Prediction_Std': f"{results['std_value']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'interpretability_summary.csv', index=False)
    
    # Save detailed results
    with open(output_dir / 'detailed_interpretability_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for method, results in interp_results.items():
            serializable_results[method] = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[method][key] = value.tolist()
                elif isinstance(value, (np.floating, np.integer)):
                    serializable_results[method][key] = float(value)
                elif isinstance(value, np.bool_):
                    serializable_results[method][key] = bool(value)
                else:
                    serializable_results[method][key] = value
        
        json.dump(serializable_results, f, indent=2)
    
    return summary_df

def main():
    """Main function for Step 13: Interpretability Analysis"""
    
    print("=== Step 13: Interpretability/Feature Analysis ===\n")
    
    # Setup paths
    base_dir = Path.cwd()
    step10_dir = Path(r'C:\Users\Sachi\OneDrive\Desktop\Dissertation\step10_visualization')
    output_dir = base_dir / 'step13_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Check available prediction files
    print("1. Checking available prediction files...")
    available_methods = {}
    
    for results_subdir in step10_dir.glob('results/*/'):
        method_name = results_subdir.name
        mean_file = results_subdir / 'mc_mean_test.npy' if 'mc' in method_name.lower() else results_subdir / 'ens_mean_test.npy'
        
        if mean_file.exists():
            available_methods[method_name] = mean_file
            print(f"  Found: {method_name} -> {mean_file}")
    
    if not available_methods:
        print("ERROR: No prediction files found in step10_visualization/results/")
        return
    
    # For interpretability analysis, we'll focus on prediction patterns
    # since we don't have trained models locally
    print("\n2. Analyzing prediction patterns and spatial characteristics...")
    
    interp_results = {}
    sample_idx = 0
    
    # For interpretability analysis, we'll focus on prediction patterns
    # since we don't have trained models locally
    print("  Focusing on prediction pattern analysis")
    
    # Analyze prediction characteristics for each method
    for method_name, pred_file in available_methods.items():
        print(f"\nAnalyzing {method_name}...")
        
        # Load predictions
        predictions = np.load(pred_file)  # Shape: (N, 1, 32, 32, 32)
        
        if predictions.shape[0] <= sample_idx:
            print(f"  Warning: Sample {sample_idx} not available, using sample 0")
            sample_idx = 0
        
        pred_sample = predictions[sample_idx]  # Shape: (1, 32, 32, 32)
        
        # Analyze spatial patterns in predictions
        results = analyze_prediction_patterns(pred_sample, method_name)
        interp_results[method_name] = results
    
    # Create visualizations
    print("\n3. Creating interpretability visualizations...")
    create_prediction_pattern_analysis(interp_results, output_dir, sample_idx)
    
    # Create spatial analysis
    print("4. Creating spatial importance analysis...")
    create_spatial_analysis(interp_results, output_dir, sample_idx)
    
    # Create summary statistics
    print("5. Creating interpretability summary...")
    summary_df = create_interpretability_summary(interp_results, output_dir)
    
    # Print summary
    print("\n=== INTERPRETABILITY ANALYSIS SUMMARY ===")
    print(summary_df.to_string(index=False))
    
    print(f"\nStep 13 Complete: Interpretability analysis saved to {output_dir}/")
    print("Generated files:")
    print("  - prediction_pattern_analysis_sample0.png")
    print("  - spatial_analysis_sample0.png")
    print("  - interpretability_summary.csv")
    print("  - detailed_interpretability_results.json")
    
    print("\nNext: Step 14 - Automated summary report generation")

if __name__ == "__main__":
    main()
