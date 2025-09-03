#!/usr/bin/env python3
"""
Validate physics properties of predicted turbulent flow fields.
"""
import os
import sys
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.physics.fields import validate_field_physics
from src.physics.spectrum import analyze_energy_spectrum

def load_predictions(results_dir: Path, method: str, split: str) -> Dict:
    """Load predictions from results directory."""
    data = {}
    
    # Load mean predictions
    mean_file = results_dir / f'{method}_mean_{split}.npy'
    if mean_file.exists():
        data['predictions'] = np.load(mean_file)
    
    # Load ground truth
    gt_candidates = [
        results_dir / f'ground_truth_{split}.npy',
        results_dir / f'y_true_{split}.npy',
        results_dir / f'targets_{split}.npy'
    ]
    
    for gt_file in gt_candidates:
        if gt_file.exists():
            data['ground_truth'] = np.load(gt_file)
            break
    
    return data

def validate_single_field(field: np.ndarray, field_name: str, dx: float = 1.0,
                         nu: float = 1e-4) -> Dict:
    """Validate physics of a single velocity field."""
    print(f"  Validating {field_name}...")
    
    results = {'field_name': field_name}
    
    # Ensure field has correct shape (3, D, H, W)
    if len(field.shape) == 4 and field.shape[0] == 1:
        # Remove batch dimension
        field = field[0]
    
    if len(field.shape) != 4 or field.shape[0] != 3:
        print(f"    Warning: Expected shape (3, D, H, W), got {field.shape}")
        return results
    
    try:
        # Field physics validation
        field_results = validate_field_physics(field, dx, nu)
        results['field_physics'] = field_results
        
        # Energy spectrum analysis
        spectrum_results = analyze_energy_spectrum(field, dx, nu)
        results['spectrum_analysis'] = spectrum_results
        
        # Summary metrics
        results['summary'] = {
            'is_incompressible': field_results.get('is_incompressible', False),
            'divergence_rms': field_results.get('div_rms', np.nan),
            'kinetic_energy': field_results.get('kinetic_energy', np.nan),
            'turbulent_ke': field_results.get('turbulent_ke', np.nan),
            'enstrophy': field_results.get('enstrophy', np.nan),
            'inertial_slope': spectrum_results.get('inertial_range', {}).get('slope', np.nan),
            'integral_length_scale': spectrum_results.get('integral_length_scale', np.nan)
        }
        
    except Exception as e:
        print(f"    Error in physics validation: {e}")
        results['error'] = str(e)
    
    return results

def plot_spectrum_comparison(pred_results: Dict, gt_results: Dict, 
                           output_dir: Path, method: str, split: str):
    """Plot energy spectrum comparison between predictions and ground truth."""
    try:
        pred_spectrum = pred_results.get('spectrum_analysis', {})
        gt_spectrum = gt_results.get('spectrum_analysis', {})
        
        pred_k = np.array(pred_spectrum.get('wavenumbers', []))
        pred_E = np.array(pred_spectrum.get('energy_spectrum', []))
        gt_k = np.array(gt_spectrum.get('wavenumbers', []))
        gt_E = np.array(gt_spectrum.get('energy_spectrum', []))
        
        if len(pred_k) == 0 or len(gt_k) == 0:
            print("    Warning: No spectrum data available for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot spectra
        ax.loglog(gt_k, gt_E, 'b-', label='Ground Truth', linewidth=2)
        ax.loglog(pred_k, pred_E, 'r--', label='Prediction', linewidth=2)
        
        # Plot theoretical -5/3 slope
        if len(gt_k) > 0:
            k_ref = gt_k[len(gt_k)//2]
            E_ref = np.interp(k_ref, gt_k, gt_E)
            k_theory = np.logspace(np.log10(k_ref/2), np.log10(k_ref*2), 10)
            E_theory = E_ref * (k_theory/k_ref)**(-5/3)
            ax.loglog(k_theory, E_theory, 'k:', label='-5/3 slope', linewidth=1)
        
        ax.set_xlabel('Wavenumber k')
        ax.set_ylabel('Energy E(k)')
        ax.set_title(f'Energy Spectrum Comparison - {method.upper()} ({split})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'spectrum_comparison_{method}_{split}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    Error plotting spectrum: {e}")

def plot_physics_summary(results: List[Dict], output_dir: Path, method: str, split: str):
    """Plot summary of physics validation metrics."""
    try:
        # Extract metrics
        metrics = ['divergence_rms', 'kinetic_energy', 'turbulent_ke', 'enstrophy', 'inertial_slope']
        pred_values = []
        gt_values = []
        
        for result in results:
            if result['field_name'] == 'predictions':
                pred_summary = result.get('summary', {})
                pred_values = [pred_summary.get(m, np.nan) for m in metrics]
            elif result['field_name'] == 'ground_truth':
                gt_summary = result.get('summary', {})
                gt_values = [gt_summary.get(m, np.nan) for m in metrics]
        
        if not pred_values or not gt_values:
            print("    Warning: Insufficient data for physics summary plot")
            return
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        metric_labels = ['Divergence RMS', 'Kinetic Energy', 'Turbulent KE', 
                        'Enstrophy', 'Inertial Slope']
        
        for i, (metric, pred_val, gt_val, label) in enumerate(zip(metrics, pred_values, gt_values, metric_labels)):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            if not (np.isnan(pred_val) or np.isnan(gt_val)):
                # Bar comparison
                ax.bar(['Ground Truth', 'Prediction'], [gt_val, pred_val], 
                      color=['blue', 'red'], alpha=0.7)
                ax.set_title(label)
                ax.set_ylabel('Value')
                
                # Add percentage difference
                if gt_val != 0:
                    pct_diff = 100 * (pred_val - gt_val) / gt_val
                    ax.text(0.5, max(gt_val, pred_val) * 1.1, f'{pct_diff:+.1f}%', 
                           ha='center', transform=ax.transData)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(label)
        
        # Remove empty subplot
        if len(axes) > len(metrics):
            fig.delaxes(axes[-1])
        
        plt.suptitle(f'Physics Validation Summary - {method.upper()} ({split})')
        plt.tight_layout()
        plt.savefig(output_dir / f'physics_summary_{method}_{split}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    Error plotting physics summary: {e}")

def main():
    parser = argparse.ArgumentParser(description='Validate physics of predicted flow fields')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--results_dir', required=True, help='Results directory')
    parser.add_argument('--method', choices=['mc', 'ens'], required=True, 
                       help='UQ method (mc or ens)')
    parser.add_argument('--split', choices=['val', 'test'], default='test',
                       help='Dataset split')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory (default: figures/{exp_id})')
    parser.add_argument('--dx', type=float, default=1.0, help='Grid spacing')
    parser.add_argument('--nu', type=float, default=1e-4, help='Kinematic viscosity')
    parser.add_argument('--max_samples', type=int, default=5, 
                       help='Maximum number of samples to validate')
    args = parser.parse_args()
    
    # Load config
    from src.utils.config import load_config
    cfg = load_config(args.config)
    results_dir = Path(args.results_dir)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        exp_id = results_dir.name
        output_dir = Path(cfg['paths']['artifacts_root']) / 'figures' / exp_id
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {results_dir}")
    data = load_predictions(results_dir, args.method, args.split)
    
    if 'predictions' not in data:
        print(f"Error: No predictions found in {results_dir}")
        return
    
    # Validate predictions and ground truth
    validation_results = []
    
    # Validate predictions
    predictions = data['predictions']
    print(f"Predictions shape: {predictions.shape}")
    
    # Take subset of samples if too many
    n_samples = min(predictions.shape[0], args.max_samples)
    if n_samples < predictions.shape[0]:
        print(f"Validating {n_samples} out of {predictions.shape[0]} samples")
        sample_indices = np.linspace(0, predictions.shape[0]-1, n_samples, dtype=int)
        predictions = predictions[sample_indices]
    
    # Validate each prediction sample
    for i in range(n_samples):
        sample_results = validate_single_field(
            predictions[i], f'predictions_sample_{i}', args.dx, args.nu
        )
        validation_results.append(sample_results)
    
    # Compute average metrics across samples
    avg_results = {'field_name': 'predictions'}
    if validation_results and 'summary' in validation_results[0]:
        avg_summary = {}
        for key in validation_results[0]['summary'].keys():
            values = [r['summary'].get(key, np.nan) for r in validation_results if 'summary' in r]
            valid_values = [v for v in values if not np.isnan(v)]
            avg_summary[key] = np.mean(valid_values) if valid_values else np.nan
        avg_results['summary'] = avg_summary
    
    # Validate ground truth if available
    gt_results = None
    if 'ground_truth' in data:
        print(f"Ground truth shape: {data['ground_truth'].shape}")
        gt_sample = data['ground_truth'][0] if len(data['ground_truth']) > 0 else None
        if gt_sample is not None:
            gt_results = validate_single_field(gt_sample, 'ground_truth', args.dx, args.nu)
    
    # Save results
    results_file = output_dir / f'physics_validation_{args.method}_{args.split}.json'
    save_data = {
        'method': args.method,
        'split': args.split,
        'parameters': {'dx': args.dx, 'nu': args.nu},
        'predictions': avg_results,
        'individual_samples': validation_results[:3],  # Save first 3 for detail
        'ground_truth': gt_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"Saved validation results to {results_file}")
    
    # Print summary
    print(f"\nPhysics Validation Summary ({args.method.upper()}, {args.split}):")
    if 'summary' in avg_results:
        summary = avg_results['summary']
        print(f"  Incompressible: {summary.get('is_incompressible', 'N/A')}")
        print(f"  Divergence RMS: {summary.get('divergence_rms', 'N/A'):.2e}")
        print(f"  Kinetic Energy: {summary.get('kinetic_energy', 'N/A'):.4f}")
        print(f"  Turbulent KE: {summary.get('turbulent_ke', 'N/A'):.4f}")
        print(f"  Enstrophy: {summary.get('enstrophy', 'N/A'):.4f}")
        print(f"  Inertial Slope: {summary.get('inertial_slope', 'N/A'):.2f} (expected: -1.67)")
        print(f"  Integral Length Scale: {summary.get('integral_length_scale', 'N/A'):.4f}")
    
    # Generate plots
    if gt_results and 'spectrum_analysis' in avg_results:
        print("Generating spectrum comparison plot...")
        plot_spectrum_comparison(avg_results, gt_results, output_dir, args.method, args.split)
    
    if gt_results:
        print("Generating physics summary plot...")
        plot_physics_summary([avg_results, gt_results], output_dir, args.method, args.split)
    
    print(f"\nPhysics validation completed. Results saved to {output_dir}")

if __name__ == '__main__':
    main()
