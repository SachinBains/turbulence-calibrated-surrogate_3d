"""
Per-band evaluation framework for thesis-compliant Y+ stratified analysis.

This module implements band-wise metrics reporting as required by the thesis:
"All metrics, uncertainties, and interpretability results are reported per band"
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import h5py
from collections import defaultdict

class BandEvaluator:
    """Evaluates model performance separately for each Y+ band."""
    
    def __init__(self, splits_path: str, yplus_bands: List[Tuple[float, float]] = None):
        """
        Initialize band evaluator.
        
        Args:
            splits_path: Path to splits metadata JSON
            yplus_bands: Y+ band definitions [(min, max), ...]
        """
        self.splits_path = Path(splits_path)
        
        # Default Y+ bands from thesis
        self.yplus_bands = yplus_bands or [
            (0, 30),      # B1: Viscous sublayer and buffer
            (30, 100),    # B2: Lower log region  
            (100, 370),   # B3: Upper log region
            (370, 1000)   # B4: Inner-outer interface
        ]
        
        self.band_names = [f"B{i+1}" for i in range(len(self.yplus_bands))]
        self.load_splits_metadata()
    
    def load_splits_metadata(self):
        """Load splits metadata and band assignments."""
        with open(self.splits_path, 'r') as f:
            self.splits_meta = json.load(f)
        
        # Extract file paths and band assignments
        self.file_paths = self.splits_meta['file_paths']
        self.band_assignments = {}
        
        # Assign files to bands based on filename
        for idx, filepath in enumerate(self.file_paths):
            band_idx = self.extract_band_from_filename(filepath)
            self.band_assignments[idx] = band_idx
    
    def extract_band_from_filename(self, filepath: str) -> int:
        """Extract band index from filename pattern."""
        import re
        
        # Pattern: chan96_band1_sample001_t101_ix796_iy416_iz253.h5
        match = re.search(r'band(\d+)', filepath)
        if match:
            band_num = int(match.group(1))
            return band_num - 1  # Convert to 0-indexed
        
        # Fallback: check batch folder structure
        if 'Batch1_Data' in filepath:
            return 0
        elif 'Batch2_Data' in filepath:
            return 1
        elif 'Batch3_Data' in filepath:
            return 2
        elif 'Batch4_Data' in filepath:
            return 3
        
        raise ValueError(f"Cannot determine band for file: {filepath}")
    
    def compute_band_metrics(self, 
                           predictions: np.ndarray,
                           targets: np.ndarray, 
                           uncertainties: Optional[np.ndarray] = None,
                           split_name: str = 'test') -> Dict:
        """
        Compute metrics separately for each Y+ band.
        
        Args:
            predictions: Model predictions [N, ...]
            targets: Ground truth targets [N, ...]
            uncertainties: Uncertainty estimates [N, ...] (optional)
            split_name: Which split to evaluate ('train', 'val', 'test', 'cal')
            
        Returns:
            Dictionary with per-band metrics
        """
        split_indices = self.splits_meta[f'{split_name}_indices']
        
        # Group indices by band
        band_indices = defaultdict(list)
        for idx in split_indices:
            band_idx = self.band_assignments[idx]
            band_indices[band_idx].append(idx)
        
        results = {
            'split': split_name,
            'bands': {},
            'global': {},
            'admissible_bands': []
        }
        
        # Compute metrics for each band
        for band_idx in range(len(self.yplus_bands)):
            if band_idx not in band_indices:
                continue
                
            indices = band_indices[band_idx]
            band_name = self.band_names[band_idx]
            yplus_range = self.yplus_bands[band_idx]
            
            # Extract band-specific data
            band_preds = predictions[indices]
            band_targets = targets[indices]
            band_uncertainties = uncertainties[indices] if uncertainties is not None else None
            
            # Compute band metrics
            band_metrics = self._compute_metrics(
                band_preds, band_targets, band_uncertainties
            )
            
            # Add band metadata
            band_metrics.update({
                'yplus_range': yplus_range,
                'n_samples': len(indices),
                'sample_indices': indices
            })
            
            results['bands'][band_name] = band_metrics
            
            # Check admissibility (example criteria - adjust as needed)
            if self._is_band_admissible(band_metrics):
                results['admissible_bands'].append(band_name)
        
        # Compute global metrics only if all bands are admissible
        if len(results['admissible_bands']) == len(self.yplus_bands):
            results['global'] = self._compute_metrics(
                predictions[split_indices], 
                targets[split_indices],
                uncertainties[split_indices] if uncertainties is not None else None
            )
            results['global']['conditional_on_admissibility'] = True
        else:
            results['global']['conditional_on_admissibility'] = False
            results['global']['inadmissible_bands'] = [
                band for band in self.band_names 
                if band not in results['admissible_bands']
            ]
        
        return results
    
    def _compute_metrics(self, predictions: np.ndarray, 
                        targets: np.ndarray,
                        uncertainties: Optional[np.ndarray] = None) -> Dict:
        """Compute standard regression and UQ metrics."""
        
        # Flatten for easier computation
        preds_flat = predictions.flatten()
        targets_flat = targets.flatten()
        
        # Basic regression metrics
        mse = np.mean((preds_flat - targets_flat) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds_flat - targets_flat))
        
        # R-squared
        ss_res = np.sum((targets_flat - preds_flat) ** 2)
        ss_tot = np.sum((targets_flat - np.mean(targets_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
        
        # Uncertainty quantification metrics
        if uncertainties is not None:
            unc_flat = uncertainties.flatten()
            
            # Prediction Interval Coverage Probability (PICP)
            # Assuming uncertainties represent standard deviations
            z_score = 1.96  # 95% confidence
            lower_bound = preds_flat - z_score * unc_flat
            upper_bound = preds_flat + z_score * unc_flat
            
            coverage = np.mean(
                (targets_flat >= lower_bound) & (targets_flat <= upper_bound)
            )
            
            # Mean Prediction Interval Width (MPIW)
            interval_width = upper_bound - lower_bound
            mpiw = np.mean(interval_width)
            
            # Normalized MPIW
            target_range = np.max(targets_flat) - np.min(targets_flat)
            mpiw_normalized = mpiw / target_range if target_range > 0 else 0
            
            # Uncertainty calibration (reliability)
            calibration_error = self._compute_calibration_error(
                preds_flat, targets_flat, unc_flat
            )
            
            metrics.update({
                'picp_95': float(coverage),
                'mpiw': float(mpiw),
                'mpiw_normalized': float(mpiw_normalized),
                'calibration_error': float(calibration_error),
                'mean_uncertainty': float(np.mean(unc_flat))
            })
        
        return metrics
    
    def _compute_calibration_error(self, predictions: np.ndarray,
                                 targets: np.ndarray, 
                                 uncertainties: np.ndarray,
                                 n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE)."""
        
        # Compute prediction errors
        errors = np.abs(predictions - targets)
        
        # Bin by uncertainty magnitude
        bin_boundaries = np.linspace(0, np.max(uncertainties), n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        total_samples = len(predictions)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this uncertainty bin
            in_bin = (uncertainties > bin_lower) & (uncertainties <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Average confidence (uncertainty) in bin
                avg_confidence = uncertainties[in_bin].mean()
                
                # Average accuracy (1 - normalized error) in bin
                avg_error = errors[in_bin].mean()
                avg_accuracy = 1 - (avg_error / np.std(targets))  # Normalized by target std
                
                # Expected vs actual accuracy
                ece += np.abs(avg_accuracy - (1 - avg_confidence)) * prop_in_bin
        
        return ece
    
    def _is_band_admissible(self, band_metrics: Dict) -> bool:
        """
        Determine if a band's performance is admissible.
        
        Adjust these criteria based on your thesis requirements.
        """
        # Example admissibility criteria
        criteria = [
            band_metrics['r2'] > 0.5,  # Minimum R²
            band_metrics['rmse'] < 2.0,  # Maximum RMSE threshold
        ]
        
        # Add UQ criteria if available
        if 'picp_95' in band_metrics:
            criteria.extend([
                band_metrics['picp_95'] > 0.90,  # Minimum coverage
                band_metrics['calibration_error'] < 0.1  # Maximum calibration error
            ])
        
        return all(criteria)
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Band evaluation results saved to: {output_path}")
    
    def generate_report(self, results: Dict) -> str:
        """Generate a human-readable evaluation report."""
        
        report = []
        report.append("=" * 60)
        report.append("Y+ BAND-WISE EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"Split: {results['split']}")
        report.append(f"Admissible bands: {results['admissible_bands']}")
        report.append("")
        
        # Per-band results
        for band_name, metrics in results['bands'].items():
            yplus_range = metrics['yplus_range']
            n_samples = metrics['n_samples']
            
            report.append(f"Band {band_name} [Y+ {yplus_range[0]}-{yplus_range[1]}): {n_samples} samples")
            report.append("-" * 40)
            report.append(f"  RMSE: {metrics['rmse']:.4f}")
            report.append(f"  MAE:  {metrics['mae']:.4f}")
            report.append(f"  R²:   {metrics['r2']:.4f}")
            
            if 'picp_95' in metrics:
                report.append(f"  PICP (95%): {metrics['picp_95']:.3f}")
                report.append(f"  MPIW:       {metrics['mpiw']:.4f}")
                report.append(f"  Cal. Error: {metrics['calibration_error']:.4f}")
            
            report.append("")
        
        # Global results (conditional)
        if results['global'].get('conditional_on_admissibility', False):
            report.append("GLOBAL METRICS (All bands admissible)")
            report.append("-" * 40)
            global_metrics = results['global']
            report.append(f"  RMSE: {global_metrics['rmse']:.4f}")
            report.append(f"  MAE:  {global_metrics['mae']:.4f}")
            report.append(f"  R²:   {global_metrics['r2']:.4f}")
            
            if 'picp_95' in global_metrics:
                report.append(f"  PICP (95%): {global_metrics['picp_95']:.3f}")
                report.append(f"  MPIW:       {global_metrics['mpiw']:.4f}")
        else:
            report.append("GLOBAL METRICS: NOT REPORTED")
            report.append("(Conditional on band-wise admissibility - FAILED)")
            inadmissible = results['global'].get('inadmissible_bands', [])
            report.append(f"Inadmissible bands: {inadmissible}")
        
        return "\n".join(report)


def evaluate_model_by_bands(model_output_path: str,
                          splits_path: str,
                          output_dir: str,
                          split_name: str = 'test'):
    """
    Convenience function to evaluate a model's outputs by Y+ bands.
    
    Args:
        model_output_path: Path to model predictions/uncertainties
        splits_path: Path to splits metadata
        output_dir: Directory to save results
        split_name: Which split to evaluate
    """
    
    # Load model outputs (adjust based on your format)
    # This is a placeholder - modify based on your actual output format
    with h5py.File(model_output_path, 'r') as f:
        predictions = f['predictions'][:]
        targets = f['targets'][:]
        uncertainties = f.get('uncertainties', None)
        if uncertainties is not None:
            uncertainties = uncertainties[:]
    
    # Initialize evaluator
    evaluator = BandEvaluator(splits_path)
    
    # Compute band-wise metrics
    results = evaluator.compute_band_metrics(
        predictions, targets, uncertainties, split_name
    )
    
    # Save results
    output_path = Path(output_dir) / f"band_evaluation_{split_name}.json"
    evaluator.save_results(results, output_path)
    
    # Generate and save report
    report = evaluator.generate_report(results)
    report_path = Path(output_dir) / f"band_evaluation_{split_name}_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model by Y+ bands")
    parser.add_argument("--model_output", required=True, help="Path to model outputs")
    parser.add_argument("--splits", required=True, help="Path to splits metadata")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--split", default="test", help="Split to evaluate")
    
    args = parser.parse_args()
    
    evaluate_model_by_bands(
        args.model_output,
        args.splits, 
        args.output_dir,
        args.split
    )
