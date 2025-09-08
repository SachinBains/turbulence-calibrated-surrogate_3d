#!/usr/bin/env python3
"""
Zero-shot evaluation of Re_τ=1000 trained models on Re_τ=5200 secondary dataset.
NO retraining, NO recalibration - reuses frozen primary artifacts exactly.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import h5py
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from dataio.channel_dataset import ChannelDataset
from eval.band_evaluation import BandEvaluator
from eval.physics_gates import PhysicsGateValidator
# # from utils.seeding import set_deterministic_seeds
# # from utils.devices import get_device
# from utils.logging import setup_logging

class SecondaryDatasetEvaluator:
    """Zero-shot evaluator for Re_τ=5200 secondary dataset."""
    
    def __init__(self, config_path: str):
        """Initialize evaluator with secondary config."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set deterministic seeds
        if self.config.get('seed'):
            import random
            import numpy as np
            import torch
            seed = self.config['seed']
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize physics gate validator
        self.physics_validator = PhysicsGateValidator()
        
        # Load secondary dataset manifest
        self.manifest_path = Path(self.config['paths']['artifacts_root']) / 'datasets' / 'channel3d_secondary' / 'manifest.csv'
        
    def load_frozen_artifacts(self) -> Dict:
        """Load frozen artifacts from primary Re_τ=1000 training."""
        artifacts = {}
        
        # 1. Load trained model checkpoint
        checkpoint_path = self.config['primary_artifacts']['checkpoint_path']
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Primary checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        artifacts['checkpoint'] = checkpoint
        
        # 2. Load frozen normalization stats
        stats_path = self.config['primary_artifacts']['normalization_stats']
        if not Path(stats_path).exists():
            raise FileNotFoundError(f"Normalization stats not found: {stats_path}")
        
        print(f"Loading normalization stats: {stats_path}")
        stats = np.load(stats_path)
        artifacts['normalization'] = {
            'mean': stats['mean'],
            'std': stats['std']
        }
        
        # 3. Load conformal quantiles (if available)
        quantiles_path = self.config['primary_artifacts']['conformal_quantiles']
        if Path(quantiles_path).exists():
            print(f"Loading conformal quantiles: {quantiles_path}")
            quantiles = np.load(quantiles_path)
            artifacts['conformal_quantiles'] = quantiles
        else:
            print("Warning: Conformal quantiles not found - will skip interval evaluation")
            artifacts['conformal_quantiles'] = None
        
        return artifacts
    
    def create_secondary_dataset(self) -> ChannelDataset:
        """Create secondary dataset with frozen normalization."""
        # Create manifest if it doesn't exist
        if not self.manifest_path.exists():
            self._create_manifest()
        
        # Create dataset with frozen normalization
        dataset_config = self.config['dataset'].copy()
        dataset_config['split'] = 'test'  # Secondary is test-only
        dataset_config['eval_mode'] = True
        dataset_config['frozen_stats_path'] = self.config['primary_artifacts']['normalization_stats']
        
        dataset = ChannelDataset(dataset_config, split='test', eval_mode=True)
        return dataset
    
    def _create_manifest(self):
        """Create secondary dataset manifest."""
        print("Creating secondary dataset manifest...")
        
        data_dirs = self.config['dataset']['data_dirs']
        manifest_entries = []
        
        for data_dir in data_dirs:
            data_path = Path(data_dir)
            if not data_path.exists():
                print(f"Warning: Directory {data_dir} does not exist")
                continue
            
            h5_files = list(data_path.glob("*.h5"))
            print(f"Found {len(h5_files)} files in {data_dir}")
            
            for h5_file in h5_files:
                # Extract metadata from filename
                # Format: 5200_96_sample105_t5_ix6773_iy123_iz4197_yplus1155.h5
                filename = h5_file.name
                parts = filename.replace('.h5', '').split('_')
                
                try:
                    sample_id = int(parts[3].replace('sample', ''))
                    yplus = int(parts[7].replace('yplus', ''))
                    
                    # Assign y+ band
                    if 0 <= yplus < 30:
                        yplus_band = 1
                    elif 30 <= yplus < 100:
                        yplus_band = 2
                    elif 100 <= yplus < 370:
                        yplus_band = 3
                    elif 370 <= yplus <= 1000:
                        yplus_band = 4
                    else:
                        continue  # Skip files outside range
                    
                    entry = {
                        'file_path': str(h5_file.absolute()),
                        'sample_id': sample_id,
                        'yplus': yplus,
                        'yplus_band': yplus_band,
                        're_tau': 5200
                    }
                    manifest_entries.append(entry)
                    
                except (IndexError, ValueError) as e:
                    print(f"Warning: Could not parse {filename}: {e}")
                    continue
        
        # Save manifest
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(manifest_entries)
        df = df.sort_values('sample_id').reset_index(drop=True)
        df.to_csv(self.manifest_path, index=False)
        
        print(f"Created manifest with {len(df)} files")
        print(f"Y+ band distribution:")
        for band in [1, 2, 3, 4]:
            count = len(df[df['yplus_band'] == band])
            print(f"  Band {band}: {count} cubes")
    
    def load_model(self, checkpoint: Dict) -> torch.nn.Module:
        """Load model from checkpoint."""
        # Import model based on config
        model_name = self.config['model']['name']
        
        if model_name == 'unet3d':
            from models.unet3d import UNet3D
            model = UNet3D(
                in_channels=self.config['model']['in_channels'],
                out_channels=self.config['model']['out_channels'],
                base_channels=self.config['model']['base_channels'],
                depth=self.config['model']['depth']
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def predict_on_secondary(self, model: torch.nn.Module, dataset: ChannelDataset) -> Dict:
        """Generate predictions on secondary dataset."""
        print(f"Generating predictions on {len(dataset)} secondary cubes...")
        
        predictions = []
        ground_truths = []
        metadata = []
        
        # Load manifest for metadata
        manifest_df = pd.read_csv(self.manifest_path)
        
        with torch.no_grad():
            for i in range(len(dataset)):
                # Get data
                input_tensor, target_tensor = dataset[i]
                input_batch = input_tensor.unsqueeze(0).to(self.device)
                
                # Predict
                pred_batch = model(input_batch)
                pred_tensor = pred_batch.squeeze(0).cpu()
                
                predictions.append(pred_tensor.numpy())
                ground_truths.append(target_tensor.numpy())
                
                # Get metadata
                file_info = manifest_df.iloc[i]
                metadata.append({
                    'sample_id': file_info['sample_id'],
                    'yplus': file_info['yplus'],
                    'yplus_band': file_info['yplus_band'],
                    'file_path': file_info['file_path']
                })
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(dataset)} cubes")
        
        return {
            'predictions': np.array(predictions),
            'ground_truths': np.array(ground_truths),
            'metadata': metadata
        }
    
    def evaluate_physics_gates(self, predictions: np.ndarray, metadata: List[Dict]) -> Dict:
        """Evaluate physics gates per band."""
        print("Evaluating physics gates...")
        
        gate_results = {'by_band': {}, 'overall': {}}
        
        # Group by bands
        for band in [1, 2, 3, 4]:
            band_indices = [i for i, meta in enumerate(metadata) if meta['yplus_band'] == band]
            if not band_indices:
                continue
            
            band_predictions = predictions[band_indices]
            print(f"  Band {band}: {len(band_indices)} cubes")
            
            # Run physics validation on band
            band_results = []
            for pred in band_predictions:
                # Convert from (3, 96, 96, 96) to physics validator format
                velocity_field = pred  # Already in correct format
                
                # Run validation (without y+ coords for now - would need to compute from Re_τ=5200)
                result = self.physics_validator.comprehensive_physics_validation(
                    velocity_field, dx=1.0  # Normalized grid spacing
                )
                band_results.append(result)
            
            # Aggregate band results
            pass_rates = {
                'incompressibility': np.mean([r['incompressibility']['pass'] for r in band_results]),
                'overall': np.mean([r['overall_pass'] for r in band_results])
            }
            
            gate_results['by_band'][band] = {
                'n_cubes': len(band_indices),
                'pass_rates': pass_rates,
                'detailed_results': band_results
            }
        
        return gate_results
    
    def evaluate_metrics(self, predictions: np.ndarray, ground_truths: np.ndarray, 
                        metadata: List[Dict], gate_results: Dict) -> Dict:
        """Evaluate metrics per band where physics gates pass."""
        print("Evaluating metrics per band...")
        
        # Use existing band evaluator
        evaluator = BandEvaluator()
        
        # Convert to band evaluator format
        results_by_band = {}
        
        for band in [1, 2, 3, 4]:
            band_indices = [i for i, meta in enumerate(metadata) if meta['yplus_band'] == band]
            if not band_indices:
                continue
            
            band_preds = predictions[band_indices]
            band_targets = ground_truths[band_indices]
            
            # Check if band passes physics gates
            band_gate_results = gate_results['by_band'].get(band, {})
            overall_pass_rate = band_gate_results.get('pass_rates', {}).get('overall', 0.0)
            
            if overall_pass_rate < 0.8:  # Require 80% pass rate
                print(f"  Band {band}: SKIPPED (physics gate pass rate: {overall_pass_rate:.1%})")
                results_by_band[band] = {
                    'status': 'FAILED_PHYSICS_GATES',
                    'pass_rate': overall_pass_rate,
                    'n_cubes': len(band_indices)
                }
                continue
            
            print(f"  Band {band}: Computing metrics ({len(band_indices)} cubes)")
            
            # Compute metrics
            band_metrics = {}
            for component_idx, component in enumerate(['u', 'v', 'w']):
                pred_comp = band_preds[:, component_idx]
                target_comp = band_targets[:, component_idx]
                
                # Flatten for metrics computation
                pred_flat = pred_comp.flatten()
                target_flat = target_comp.flatten()
                
                # Compute metrics
                mse = np.mean((pred_flat - target_flat) ** 2)
                mae = np.mean(np.abs(pred_flat - target_flat))
                
                # R²
                ss_res = np.sum((target_flat - pred_flat) ** 2)
                ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                band_metrics[component] = {
                    'RMSE': np.sqrt(mse),
                    'MAE': mae,
                    'R2': r2
                }
            
            results_by_band[band] = {
                'status': 'COMPUTED',
                'pass_rate': overall_pass_rate,
                'n_cubes': len(band_indices),
                'metrics': band_metrics
            }
        
        return results_by_band
    
    def save_results(self, results: Dict):
        """Save evaluation results."""
        results_dir = Path(self.config['paths']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = results_dir / 'secondary_evaluation_metrics.json'
        import json
        with open(metrics_file, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                return obj
            
            json.dump(results, f, indent=2, default=convert_numpy)
        
        print(f"Results saved to: {metrics_file}")
        
        # Generate summary report
        self._generate_summary_report(results, results_dir)
    
    def _generate_summary_report(self, results: Dict, results_dir: Path):
        """Generate human-readable summary report."""
        report_file = results_dir / 'secondary_evaluation_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# Re_τ=5200 Secondary Dataset Evaluation Report\n\n")
            f.write("Zero-shot evaluation of Re_τ=1000 trained model on Re_τ=5200 data.\n\n")
            
            f.write("## Physics Gates Summary\n\n")
            for band, band_results in results['physics_gates']['by_band'].items():
                pass_rate = band_results['pass_rates']['overall']
                n_cubes = band_results['n_cubes']
                status = "PASS" if pass_rate >= 0.8 else "FAIL"
                f.write(f"- Band {band}: {status} ({pass_rate:.1%} pass rate, {n_cubes} cubes)\n")
            
            f.write("\n## Accuracy Metrics (Physics-Admissible Bands Only)\n\n")
            for band, band_results in results['metrics_by_band'].items():
                if band_results['status'] == 'COMPUTED':
                    f.write(f"### Band {band}\n")
                    for component, metrics in band_results['metrics'].items():
                        f.write(f"**{component.upper()} component:**\n")
                        f.write(f"- RMSE: {metrics['RMSE']:.6f}\n")
                        f.write(f"- MAE: {metrics['MAE']:.6f}\n")
                        f.write(f"- R²: {metrics['R2']:.4f}\n")
                        f.write("\n")
                elif band_results['status'] == 'FAILED_PHYSICS_GATES':
                    f.write(f"### Band {band}: SKIPPED (Failed Physics Gates)\n")
                    f.write(f"Pass rate: {band_results['pass_rate']:.1%}\n\n")
        
        print(f"Summary report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation on Re_τ=5200 secondary dataset")
    parser.add_argument("--config", required=True, help="Secondary evaluation config file")
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    try:
        # Initialize evaluator
        evaluator = SecondaryDatasetEvaluator(args.config)
        
        # Load frozen artifacts from primary training
        artifacts = evaluator.load_frozen_artifacts()
        
        # Create secondary dataset
        dataset = evaluator.create_secondary_dataset()
        
        # Load model
        model = evaluator.load_model(artifacts['checkpoint'])
        
        # Generate predictions
        prediction_results = evaluator.predict_on_secondary(model, dataset)
        
        # Evaluate physics gates
        gate_results = evaluator.evaluate_physics_gates(
            prediction_results['predictions'], 
            prediction_results['metadata']
        )
        
        # Evaluate metrics (only where physics gates pass)
        metrics_results = evaluator.evaluate_metrics(
            prediction_results['predictions'],
            prediction_results['ground_truths'],
            prediction_results['metadata'],
            gate_results
        )
        
        # Combine results
        final_results = {
            'config': evaluator.config,
            'dataset_info': {
                'n_cubes': len(dataset),
                'resolution': '96x96x96',
                're_tau': 5200
            },
            'physics_gates': gate_results,
            'metrics_by_band': metrics_results
        }
        
        # Save results
        evaluator.save_results(final_results)
        
        print("\n" + "="*60)
        print("SECONDARY EVALUATION COMPLETED")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"Error during secondary evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
