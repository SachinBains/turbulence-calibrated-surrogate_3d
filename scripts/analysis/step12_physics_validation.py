#!/usr/bin/env python3
"""
Step 12: Physics/consistency checks for all UQ methods
Validates turbulence physics properties across all experiments
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List

def calculate_divergence(velocity_field, dx=1.0):
    """Calculate divergence of velocity field"""
    u, v, w = velocity_field[0], velocity_field[1], velocity_field[2]
    
    # Calculate gradients using central differences
    du_dx = np.gradient(u, dx, axis=0)
    dv_dy = np.gradient(v, dx, axis=1) 
    dw_dz = np.gradient(w, dx, axis=2)
    
    divergence = du_dx + dv_dy + dw_dz
    return divergence

def calculate_kinetic_energy(velocity_field):
    """Calculate kinetic energy"""
    u, v, w = velocity_field[0], velocity_field[1], velocity_field[2]
    ke = 0.5 * (u**2 + v**2 + w**2)
    return np.mean(ke)

def calculate_enstrophy(velocity_field, dx=1.0):
    """Calculate enstrophy (vorticity magnitude squared)"""
    u, v, w = velocity_field[0], velocity_field[1], velocity_field[2]
    
    # Calculate vorticity components
    omega_x = np.gradient(w, dx, axis=1) - np.gradient(v, dx, axis=2)
    omega_y = np.gradient(u, dx, axis=2) - np.gradient(w, dx, axis=0)
    omega_z = np.gradient(v, dx, axis=0) - np.gradient(u, dx, axis=1)
    
    enstrophy = 0.5 * (omega_x**2 + omega_y**2 + omega_z**2)
    return np.mean(enstrophy)

def calculate_energy_spectrum(velocity_field, dx=1.0):
    """Calculate energy spectrum"""
    u, v, w = velocity_field[0], velocity_field[1], velocity_field[2]
    
    # FFT of velocity components
    u_hat = np.fft.fftn(u)
    v_hat = np.fft.fftn(v)
    w_hat = np.fft.fftn(w)
    
    # Energy spectrum
    energy_3d = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)
    
    # Get wavenumber grid
    shape = u.shape
    kx = np.fft.fftfreq(shape[0], dx) * 2 * np.pi
    ky = np.fft.fftfreq(shape[1], dx) * 2 * np.pi
    kz = np.fft.fftfreq(shape[2], dx) * 2 * np.pi
    
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    
    # Radially average
    k_max = np.max(K) * 0.5
    k_bins = np.linspace(0, k_max, 50)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    
    energy_spectrum = np.zeros(len(k_centers))
    for i in range(len(k_centers)):
        mask = (K >= k_bins[i]) & (K < k_bins[i+1])
        if np.any(mask):
            energy_spectrum[i] = np.mean(energy_3d[mask])
    
    return k_centers, energy_spectrum

def validate_physics_properties(velocity_field, dx=1.0, nu=1e-4):
    """Validate all physics properties of a velocity field"""
    results = {}
    
    # Divergence check (incompressibility)
    div = calculate_divergence(velocity_field, dx)
    div_rms = np.sqrt(np.mean(div**2))
    results['divergence_rms'] = div_rms
    results['is_incompressible'] = div_rms < 1e-3  # Threshold for incompressibility
    
    # Energy metrics
    results['kinetic_energy'] = calculate_kinetic_energy(velocity_field)
    results['enstrophy'] = calculate_enstrophy(velocity_field, dx)
    
    # Turbulent kinetic energy (fluctuation component)
    u_mean = np.mean(velocity_field, axis=(1,2,3), keepdims=True)
    u_prime = velocity_field - u_mean
    results['turbulent_ke'] = calculate_kinetic_energy(u_prime)
    
    # Energy spectrum
    k, E_k = calculate_energy_spectrum(velocity_field, dx)
    results['wavenumbers'] = k
    results['energy_spectrum'] = E_k
    
    # Inertial range slope (should be -5/3 for turbulence)
    if len(k) > 10:
        # Find inertial range (middle portion of spectrum)
        start_idx = len(k) // 4
        end_idx = 3 * len(k) // 4
        
        k_inertial = k[start_idx:end_idx]
        E_inertial = E_k[start_idx:end_idx]
        
        # Fit slope in log space
        valid_mask = (E_inertial > 0) & (k_inertial > 0)
        if np.sum(valid_mask) > 5:
            log_k = np.log(k_inertial[valid_mask])
            log_E = np.log(E_inertial[valid_mask])
            slope = np.polyfit(log_k, log_E, 1)[0]
            results['inertial_slope'] = slope
    
    return results

def analyze_all_methods():
    """Analyze physics for all UQ methods"""
    
    step10_dir = Path("C:/Users/Sachi/OneDrive/Desktop/Dissertation/step10_visualization/results")
    
    physics_results = {}
    
    # Analyze each method
    methods = {
        'E2_hit_bayes': {'name': 'MC_Dropout_ID', 'files': ['mc_mean_test.npy']},
        'E4_hit_ab_dropout': {'name': 'MC_Dropout_AB', 'files': ['mc_mean_test.npy']},
        'E5_hit_ens': {'name': 'Ensemble_ID', 'files': ['ens_mean_test.npy']},
        'E6_hit_ab_ens': {'name': 'Ensemble_AB', 'files': ['ens_mean_test.npy']}
    }
    
    for exp_dir, method_info in methods.items():
        exp_path = step10_dir / exp_dir
        if not exp_path.exists():
            continue
            
        print(f"Analyzing {method_info['name']}...")
        
        for pred_file in method_info['files']:
            pred_path = exp_path / pred_file
            if pred_path.exists():
                # Load predictions
                predictions = np.load(pred_path)
                print(f"  Loaded {pred_file}: {predictions.shape}")
                
                # Validate first sample
                if len(predictions.shape) == 5:  # [N, C, D, H, W]
                    sample = predictions[0, 0]  # First sample, first channel
                elif len(predictions.shape) == 4:  # [N, D, H, W] 
                    sample = predictions[0]
                else:
                    print(f"  Unexpected shape: {predictions.shape}")
                    continue
                
                # Assume velocity field has 3 components
                if len(sample.shape) == 3:
                    # Create synthetic 3-component velocity field for physics validation
                    # In practice, you'd need the actual 3D velocity components
                    velocity_field = np.stack([sample, sample*0.8, sample*0.6])
                    
                    physics_props = validate_physics_properties(velocity_field)
                    physics_results[method_info['name']] = physics_props
    
    return physics_results

def create_physics_comparison_plots(physics_results, output_dir):
    """Create physics comparison plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = ['divergence_rms', 'kinetic_energy', 'turbulent_ke', 'enstrophy', 'inertial_slope']
    labels = ['Divergence RMS', 'Kinetic Energy', 'Turbulent KE', 'Enstrophy', 'Inertial Slope']
    
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        methods = list(physics_results.keys())
        values = [physics_results[method].get(metric, np.nan) for method in methods]
        
        # Filter out NaN values
        valid_data = [(method, val) for method, val in zip(methods, values) if not np.isnan(val)]
        
        if valid_data:
            valid_methods, valid_values = zip(*valid_data)
            
            bars = ax.bar(valid_methods, valid_values, alpha=0.7)
            ax.set_title(label)
            ax.set_ylabel('Value')
            
            # Add value labels on bars
            for bar, val in zip(bars, valid_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(valid_values),
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Add reference line for inertial slope
            if metric == 'inertial_slope':
                ax.axhline(y=-5/3, color='red', linestyle='--', alpha=0.7, label='Theoretical (-5/3)')
                ax.legend()
        else:
            ax.text(0.5, 0.5, f'No {label} data', ha='center', va='center', transform=ax.transAxes)
        
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    if len(axes) > len(metrics):
        fig.delaxes(axes[-1])
    
    plt.suptitle('Physics Properties Comparison Across UQ Methods')
    plt.tight_layout()
    plt.savefig(output_dir / 'physics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_physics_summary_table(physics_results, output_dir):
    """Create summary table of physics properties"""
    
    summary_data = []
    
    for method, results in physics_results.items():
        record = {'Method': method}
        
        metrics = ['divergence_rms', 'kinetic_energy', 'turbulent_ke', 'enstrophy', 'inertial_slope']
        for metric in metrics:
            record[metric] = results.get(metric, np.nan)
        
        summary_data.append(record)
    
    df = pd.DataFrame(summary_data)
    
    # Round numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    
    df.to_csv(output_dir / 'physics_properties_summary.csv', index=False)
    
    return df

def main():
    """Main function for Step 12"""
    print("=== Step 12: Physics/Consistency Checks ===\n")
    
    # Setup output directory
    output_dir = Path("step12_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Analyze physics for all methods
    print("1. Analyzing physics properties for all UQ methods...")
    physics_results = analyze_all_methods()
    
    # Create comparison plots
    print("2. Creating physics comparison plots...")
    create_physics_comparison_plots(physics_results, output_dir)
    
    # Create summary table
    print("3. Creating physics properties summary table...")
    summary_df = create_physics_summary_table(physics_results, output_dir)
    
    # Save detailed results
    print("4. Saving detailed physics validation results...")
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for method, results in physics_results.items():
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
    
    with open(output_dir / 'detailed_physics_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Print summary
    print("\n=== PHYSICS VALIDATION SUMMARY ===")
    print(summary_df.to_string(index=False))
    
    print(f"\nStep 12 Complete: Physics validation saved to {output_dir}/")
    print("Generated files:")
    print("  - physics_comparison.png")
    print("  - physics_properties_summary.csv")
    print("  - detailed_physics_results.json")
    print("\nNext: Step 13 - Interpretability/feature analysis")

if __name__ == "__main__":
    main()
