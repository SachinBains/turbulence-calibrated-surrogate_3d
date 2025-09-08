#!/usr/bin/env python3
"""
Unified report pack generator for thesis scope.
Produces all required outputs and pass/fail table with fixed configs, seeds, and git hashes.
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.uq.yplus_conformal import YPlusConformPredictor
from src.interp.germano_coeff import GermanoCoeffRecovery
from src.eval.physics_gates import PhysicsGateValidator
from src.eval.multiscale_physics import MultiScalePhysicsValidator

class ThesisReportPack:
    """Unified report pack generator for thesis scope."""
    
    def __init__(self, output_dir: str):
        """Initialize report pack generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'figures' / 'calibration').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'figures' / 'coverage').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'figures' / 'physics').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'figures' / 'coeff').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'figures' / 'maps').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'results').mkdir(parents=True, exist_ok=True)
        
        # Initialize validators with updated Y+ bands
        yplus_bands = [(0, 30), (30, 100), (100, 370), (370, 1000)]
        self.conformal_predictor = YPlusConformPredictor(yplus_bands)
        self.germano_recovery = GermanoCoeffRecovery()
        self.physics_validator = PhysicsGateValidator()
        
        # Thesis scope thresholds
        self.thresholds = {
            'picp_target': 0.90,
            'picp_tolerance': 0.02,
            'ece_max': 0.05,
            'divergence_max_ratio': 1.2,
            'spectra_deviation_max': 0.10,
            'wall_law_max_deviation': 0.15,
            'coeff_error_correlation_min': 0.6
        }
    
    def get_git_info(self) -> Dict[str, str]:
        """Get current git hash and status."""
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
            git_status = subprocess.check_output(['git', 'status', '--porcelain']).decode().strip()
            
            return {
                'git_hash': git_hash,
                'git_dirty': len(git_status) > 0,
                'git_status': git_status
            }
        except:
            return {
                'git_hash': 'unknown',
                'git_dirty': True,
                'git_status': 'git not available'
            }
    
    def load_model_predictions(self, model_dir: Path) -> Dict:
        """Load model predictions and metadata."""
        predictions = {}
        
        # Load predictions
        for pred_type in ['mean', 'var', 'samples']:
            for split in ['val', 'test']:
                pred_file = model_dir / f'{pred_type}_{split}.npy'
                if pred_file.exists():
                    predictions[f'{pred_type}_{split}'] = np.load(pred_file)
        
        # Load ground truth
        for split in ['val', 'test']:
            gt_file = model_dir / f'ground_truth_{split}.npy'
            if gt_file.exists():
                predictions[f'ground_truth_{split}'] = np.load(gt_file)
        
        # Load metadata
        metadata_file = model_dir / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                predictions['metadata'] = json.load(f)
        
        return predictions
    
    def evaluate_risk_control(self, predictions: Dict, model_name: str) -> Dict:
        """Evaluate Y+-conditional conformal prediction."""
        print(f"  Evaluating risk control for {model_name}...")
        
        # Get predictions and ground truth
        y_pred_val = predictions.get('mean_val')
        y_true_val = predictions.get('ground_truth_val')
        y_pred_test = predictions.get('mean_test')
        y_true_test = predictions.get('ground_truth_test')
        
        if any(x is None for x in [y_pred_val, y_true_val, y_pred_test, y_true_test]):
            return {'error': 'Missing prediction data'}
        
        # Compute Y+ coordinates (assuming 96^3 grid)
        grid_shape = y_pred_val.shape[-3:]
        yplus_coords = self.conformal_predictor.compute_yplus_coordinates(grid_shape)
        
        # Calibrate on validation set
        calibration_info = self.conformal_predictor.calibrate(
            y_true_val, y_pred_val, yplus_coords, alpha=0.1
        )
        
        # Generate prediction intervals for test set
        lower_bounds, upper_bounds = self.conformal_predictor.predict_intervals(
            y_pred_test, yplus_coords
        )
        
        # Evaluate coverage
        coverage_metrics = self.conformal_predictor.evaluate_coverage(
            y_true_test, lower_bounds, upper_bounds, yplus_coords
        )
        
        # Check pass/fail criteria
        band_results = []
        overall_pass = True
        
        for band_idx, band_metrics in coverage_metrics['by_band'].items():
            picp = band_metrics['picp']
            target = self.thresholds['picp_target']
            tolerance = self.thresholds['picp_tolerance']
            
            band_pass = abs(picp - target) <= tolerance
            if not band_pass:
                overall_pass = False
            
            band_results.append({
                'band_idx': band_idx,
                'yplus_range': band_metrics['yplus_range'],
                'picp': picp,
                'mpiw': band_metrics['mpiw'],
                'target': target,
                'pass': band_pass
            })
        
        return {
            'calibration_info': calibration_info,
            'coverage_metrics': coverage_metrics,
            'band_results': band_results,
            'overall_pass': overall_pass,
            'prediction_intervals': {
                'lower_bounds': lower_bounds,
                'upper_bounds': upper_bounds
            }
        }
    
    def evaluate_physics_gates(self, predictions: Dict, model_name: str) -> Dict:
        """Evaluate physics admissibility gates."""
        print(f"  Evaluating physics gates for {model_name}...")
        
        y_pred_test = predictions.get('mean_test')
        y_true_test = predictions.get('ground_truth_test')
        
        if y_pred_test is None:
            return {'error': 'Missing prediction data'}
        
        # Compute Y+ coordinates
        grid_shape = y_pred_test.shape[-3:]
        yplus_coords = self.conformal_predictor.compute_yplus_coordinates(grid_shape)
        
        # Prepare reference data if available
        reference_data = {}
        if y_true_test is not None:
            # Compute reference divergence from DNS
            ref_validation = self.physics_validator.comprehensive_physics_validation(
                y_true_test[0] if y_true_test.ndim == 5 else y_true_test,
                yplus_coords
            )
            reference_data['divergence_norm'] = ref_validation['incompressibility']['divergence_norm']
            reference_data['energy_spectrum'] = ref_validation['energy_spectrum']['energy_spectrum']
        
        # Validate physics for predictions
        physics_results = self.physics_validator.comprehensive_physics_validation(
            y_pred_test, yplus_coords, reference_data
        )
        
        return physics_results
    
    def evaluate_coefficient_recovery(self, predictions: Dict, model_name: str) -> Dict:
        """Evaluate Germano coefficient recovery and interpretability."""
        print(f"  Evaluating coefficient recovery for {model_name}...")
        
        y_pred_test = predictions.get('mean_test')
        y_true_test = predictions.get('ground_truth_test')
        
        if y_pred_test is None:
            return {'error': 'Missing prediction data'}
        
        # Take first sample for analysis
        if y_pred_test.ndim == 5:
            pred_sample = y_pred_test[0]  # (3, D, H, W)
            true_sample = y_true_test[0] if y_true_test is not None else None
        else:
            pred_sample = y_pred_test
            true_sample = y_true_test
        
        # Compute dynamic Smagorinsky coefficient
        cs_squared = self.germano_recovery.compute_dynamic_coefficient(
            pred_sample[0], pred_sample[1], pred_sample[2],
            filter_type='box', averaging='local3'
        )
        
        # Compute Y+ coordinates
        grid_shape = pred_sample.shape[-3:]
        yplus_coords = self.germano_recovery.compute_yplus_coordinates(grid_shape)
        
        # Aggregate by Y+ bands
        aggregated_results = self.germano_recovery.aggregate_coefficient_by_yplus(
            cs_squared, yplus_coords
        )
        
        # Analyze reliability (correlation with error/uncertainty)
        reliability_results = {}
        if true_sample is not None:
            prediction_error = pred_sample - true_sample
            
            # Get uncertainty if available
            pred_var = predictions.get('var_test')
            uncertainty = np.sqrt(pred_var[0]) if pred_var is not None else None
            
            reliability_results = self.germano_recovery.analyze_coefficient_reliability(
                cs_squared, prediction_error[0], uncertainty[0] if uncertainty is not None else None, yplus_coords
            )
        
        # Check pass/fail criterion
        coeff_pass = True
        if 'error_coefficient_correlation' in reliability_results:
            correlation = reliability_results['error_coefficient_correlation']
            coeff_pass = correlation >= self.thresholds['coeff_error_correlation_min']
        
        return {
            'cs_squared_field': cs_squared,
            'aggregated_results': aggregated_results,
            'reliability_results': reliability_results,
            'coefficient_pass': coeff_pass
        }
    
    def create_summary_table(self, all_results: Dict[str, Dict]) -> pd.DataFrame:
        """Create comprehensive summary table."""
        summary_data = []
        
        for model_name, results in all_results.items():
            # Extract metrics for each split and band
            for split in ['test']:  # Focus on test split for final results
                # Overall metrics
                row_base = {
                    'model': model_name,
                    'split': split,
                    'band': 'overall'
                }
                
                # Risk metrics
                risk_results = results.get('risk_control', {})
                if 'coverage_metrics' in risk_results:
                    overall_metrics = risk_results['coverage_metrics']['overall']
                    row_base.update({
                        'picp': overall_metrics['picp'],
                        'mpiw': overall_metrics['mpiw']
                    })
                
                # Physics metrics
                physics_results = results.get('physics_gates', {})
                if 'incompressibility' in physics_results:
                    row_base['divergence_norm'] = physics_results['incompressibility']['divergence_norm']
                
                if 'energy_spectrum' in physics_results:
                    spectrum = physics_results['energy_spectrum']
                    row_base['spectra_deviation'] = spectrum.get('max_spectral_deviation', 1.0)
                
                if 'wall_law' in physics_results:
                    wall = physics_results['wall_law']
                    row_base['wall_law_deviation'] = wall.get('max_log_deviation', 1.0)
                
                # Coefficient metrics
                coeff_results = results.get('coefficient_recovery', {})
                if 'reliability_results' in coeff_results:
                    reliability = coeff_results['reliability_results']
                    row_base['coeff_error_correlation'] = reliability.get('error_coefficient_correlation', 0.0)
                
                # Pass/fail status
                row_base.update({
                    'risk_pass': risk_results.get('overall_pass', False),
                    'physics_pass': physics_results.get('overall_pass', False),
                    'coeff_pass': coeff_results.get('coefficient_pass', False),
                    'overall_pass': (risk_results.get('overall_pass', False) and 
                                   physics_results.get('overall_pass', False) and 
                                   coeff_results.get('coefficient_pass', False))
                })
                
                summary_data.append(row_base.copy())
                
                # Band-wise metrics
                if 'band_results' in risk_results:
                    for band_result in risk_results['band_results']:
                        row_band = row_base.copy()
                        row_band.update({
                            'band': f"Y+{band_result['yplus_range']}",
                            'picp': band_result['picp'],
                            'mpiw': band_result['mpiw'],
                            'risk_pass': band_result['pass']
                        })
                        summary_data.append(row_band)
        
        return pd.DataFrame(summary_data)
    
    def create_figures(self, all_results: Dict[str, Dict]):
        """Create all required figures."""
        print("  Creating figures...")
        
        for model_name, results in all_results.items():
            # Risk control figures
            if 'risk_control' in results:
                self.create_coverage_figures(results['risk_control'], model_name)
            
            # Physics figures
            if 'physics_gates' in results:
                self.create_physics_figures(results['physics_gates'], model_name)
            
            # Coefficient figures
            if 'coefficient_recovery' in results:
                self.create_coefficient_figures(results['coefficient_recovery'], model_name)
    
    def create_coverage_figures(self, risk_results: Dict, model_name: str):
        """Create coverage and calibration figures."""
        if 'band_results' not in risk_results:
            return
        
        # PICP/MPIW by Y+ band
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        band_results = risk_results['band_results']
        band_centers = [np.mean(br['yplus_range']) for br in band_results]
        picps = [br['picp'] for br in band_results]
        mpiws = [br['mpiw'] for br in band_results]
        
        # PICP plot
        ax1.plot(band_centers, picps, 'o-', label=model_name)
        ax1.axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
        ax1.fill_between(band_centers, 0.88, 0.92, alpha=0.2, color='green', label='Tolerance')
        ax1.set_xlabel('Y⁺ (band center)')
        ax1.set_ylabel('PICP')
        ax1.set_title('Prediction Interval Coverage Probability')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MPIW plot
        ax2.plot(band_centers, mpiws, 's-', label=model_name, color='orange')
        ax2.set_xlabel('Y⁺ (band center)')
        ax2.set_ylabel('MPIW')
        ax2.set_title('Mean Prediction Interval Width')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'coverage' / f'picp_mpiw_by_yplus_{model_name}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_physics_figures(self, physics_results: Dict, model_name: str):
        """Create physics validation figures."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Divergence histogram
        if 'incompressibility' in physics_results:
            incomp = physics_results['incompressibility']
            if 'divergence_field' in incomp:
                div_field = incomp['divergence_field']
                axes[0, 0].hist(div_field.flatten(), bins=50, alpha=0.7, density=True)
                axes[0, 0].axvline(x=incomp['divergence_norm'], color='r', linestyle='--', 
                                 label=f'L2 norm: {incomp["divergence_norm"]:.2e}')
                axes[0, 0].set_xlabel('∇·u')
                axes[0, 0].set_ylabel('Density')
                axes[0, 0].set_title('Divergence Distribution')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
        
        # Energy spectrum
        if 'energy_spectrum' in physics_results:
            spectrum = physics_results['energy_spectrum']
            k = spectrum['wavenumbers']
            E_k = spectrum['energy_spectrum']
            
            valid = E_k > 0
            axes[0, 1].loglog(k[valid], E_k[valid], 'b-', label='Prediction')
            
            if 'reference_spectrum' in spectrum:
                ref_E_k = spectrum['reference_spectrum']
                axes[0, 1].loglog(k[valid], ref_E_k[valid], 'r--', label='Reference')
            
            # Kolmogorov -5/3 line
            k_ref = k[(k > 2) & (k < 20)]
            if len(k_ref) > 0:
                E_ref = k_ref**(-5/3) * E_k[k > 2][0] * (k_ref[0]**(5/3))
                axes[0, 1].loglog(k_ref, E_ref, 'k:', label='-5/3 slope')
            
            axes[0, 1].set_xlabel('Wavenumber k')
            axes[0, 1].set_ylabel('Energy E(k)')
            axes[0, 1].set_title('Energy Spectrum')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Wall law
        if 'wall_law' in physics_results:
            wall = physics_results['wall_law']
            if 'yplus' in wall and 'u_plus' in wall:
                yplus = wall['yplus']
                u_plus = wall['u_plus']
                
                axes[1, 0].semilogx(yplus, u_plus, 'b-', label='Prediction')
                
                # Theoretical lines
                y_visc = np.linspace(0.1, 5, 50)
                u_visc = y_visc
                axes[1, 0].semilogx(y_visc, u_visc, 'r--', label='U⁺ = y⁺')
                
                y_log = np.linspace(30, 300, 50)
                u_log = (1/0.41) * np.log(y_log) + 5.2
                axes[1, 0].semilogx(y_log, u_log, 'g--', label='Log law')
                
                axes[1, 0].set_xlabel('y⁺')
                axes[1, 0].set_ylabel('U⁺')
                axes[1, 0].set_title('Wall Law')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
        
        # Physics gates summary
        gate_names = []
        gate_status = []
        
        if 'incompressibility' in physics_results:
            gate_names.append('Incompressibility')
            gate_status.append(1 if physics_results['incompressibility']['pass'] else 0)
        
        if 'energy_spectrum' in physics_results and 'pass' in physics_results['energy_spectrum']:
            gate_names.append('Energy Spectrum')
            gate_status.append(1 if physics_results['energy_spectrum']['pass'] else 0)
        
        if 'wall_law' in physics_results:
            gate_names.append('Wall Law')
            gate_status.append(1 if physics_results['wall_law'].get('wall_law_pass', False) else 0)
        
        if gate_names:
            colors = ['green' if status else 'red' for status in gate_status]
            axes[1, 1].bar(gate_names, gate_status, color=colors, alpha=0.7)
            axes[1, 1].set_ylabel('Pass (1) / Fail (0)')
            axes[1, 1].set_title('Physics Gates Status')
            axes[1, 1].set_ylim(0, 1.2)
            for i, status in enumerate(gate_status):
                axes[1, 1].text(i, status + 0.05, 'PASS' if status else 'FAIL', 
                               ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'physics' / f'physics_validation_{model_name}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_coefficient_figures(self, coeff_results: Dict, model_name: str):
        """Create coefficient recovery figures."""
        if 'aggregated_results' not in coeff_results:
            return
        
        aggregated = coeff_results['aggregated_results']
        
        # Create coefficient profile plot
        self.germano_recovery.plot_coefficient_profiles(
            aggregated,
            save_path=str(self.output_dir / 'figures' / 'coeff' / f'Cs_eff_yplus_{model_name}_Re1000.png'),
            re_tau_label=f"{model_name}"
        )
        
        # Coefficient drift vs error correlation
        if 'reliability_results' in coeff_results:
            reliability = coeff_results['reliability_results']
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # Show correlation value
            corr = reliability.get('error_coefficient_correlation', 0.0)
            ax.text(0.5, 0.7, f'Coefficient-Error Correlation\nr = {corr:.3f}', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=16, bbox=dict(boxstyle='round', facecolor='lightblue'))
            
            # Show pass/fail status
            pass_status = "PASS" if corr >= self.thresholds['coeff_error_correlation_min'] else "FAIL"
            color = 'green' if pass_status == 'PASS' else 'red'
            ax.text(0.5, 0.3, f'Status: {pass_status}', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=18, fontweight='bold', color=color)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f'Coefficient Reliability Analysis - {model_name}')
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'figures' / 'coeff' / f'Cs_drift_vs_error_{model_name}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()

def main():
    """Main function for unified report pack generation."""
    parser = argparse.ArgumentParser(description='Generate unified thesis report pack')
    parser.add_argument('--runs', nargs='+', required=True, 
                       help='List of experiment run names (e.g., C3D1_primary C3D6_primary)')
    parser.add_argument('--results_base', default='results', 
                       help='Base directory containing model results')
    parser.add_argument('--output_dir', default='thesis_report_pack', 
                       help='Output directory for report pack')
    
    args = parser.parse_args()
    
    print("=== Thesis Report Pack Generator ===\n")
    
    # Initialize report pack
    report_pack = ThesisReportPack(args.output_dir)
    
    # Get git information
    git_info = report_pack.get_git_info()
    print(f"Git hash: {git_info['git_hash']}")
    print(f"Git status: {'dirty' if git_info['git_dirty'] else 'clean'}")
    
    # Process each model
    all_results = {}
    
    for run_name in args.runs:
        print(f"\nProcessing {run_name}...")
        
        # Load model predictions
        model_dir = Path(args.results_base) / run_name
        if not model_dir.exists():
            print(f"  Warning: {model_dir} not found, skipping...")
            continue
        
        predictions = report_pack.load_model_predictions(model_dir)
        if not predictions:
            print(f"  Warning: No predictions found for {run_name}, skipping...")
            continue
        
        # Run comprehensive evaluation
        results = {}
        
        # 1. Risk control evaluation
        results['risk_control'] = report_pack.evaluate_risk_control(predictions, run_name)
        
        # 2. Physics gates evaluation
        results['physics_gates'] = report_pack.evaluate_physics_gates(predictions, run_name)
        
        # 3. Coefficient recovery evaluation
        results['coefficient_recovery'] = report_pack.evaluate_coefficient_recovery(predictions, run_name)
        
        all_results[run_name] = results
    
    if not all_results:
        print("No valid results found. Exiting.")
        return
    
    # Create summary table
    print("\nCreating summary table...")
    summary_df = report_pack.create_summary_table(all_results)
    summary_df.to_csv(report_pack.output_dir / 'results' / 'summary.csv', index=False)
    
    # Create all figures
    print("Creating figures...")
    report_pack.create_figures(all_results)
    
    # Save metadata
    metadata = {
        'git_info': git_info,
        'thresholds': report_pack.thresholds,
        'runs_processed': list(all_results.keys()),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(report_pack.output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print(f"\n=== REPORT PACK COMPLETE ===")
    print(f"Output directory: {report_pack.output_dir}")
    print(f"Processed models: {', '.join(all_results.keys())}")
    
    # Show pass/fail summary
    print("\nPass/fail summary:")
    for model_name, results in all_results.items():
        risk_pass = results.get('risk_control', {}).get('overall_pass', False)
        physics_pass = results.get('physics_gates', {}).get('overall_pass', False)
        coeff_pass = results.get('coefficient_recovery', {}).get('coefficient_pass', False)
        overall_pass = risk_pass and physics_pass and coeff_pass
        
        status = "PASS" if overall_pass else "FAIL"
        print(f"  {model_name}: {status} (Risk: {risk_pass}, Physics: {physics_pass}, Coeff: {coeff_pass})")
    
    print(f"\nGenerated files:")
    print(f"  - results/summary.csv")
    print(f"  - figures/coverage/picp_mpiw_by_yplus_*.png")
    print(f"  - figures/physics/physics_validation_*.png")
    print(f"  - figures/coeff/Cs_eff_yplus_*.png")
    print(f"  - figures/coeff/Cs_drift_vs_error_*.png")
    print(f"  - metadata.json")

if __name__ == '__main__':
    main()
