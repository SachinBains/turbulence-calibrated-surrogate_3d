import argparse, torch
import json
import numpy as np
from src.utils.devices import pick_device
from pathlib import Path
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.utils.seeding import seed_all
from src.utils.logging import get_logger
from src.dataio.channel_dataset import ChannelDataset
from src.models.unet3d import UNet3D
from src.eval.evaluator import evaluate_baseline
from src.eval.temp_scaling import TemperatureScaler
from src.eval.conformal import conformal_wrap
import torch.nn.functional as F
from tqdm import tqdm

def load_ensemble_models(cfg, ensemble_dir, device):
    """Load all ensemble members from the members directory."""
    models = []
    member_dirs = sorted([d for d in ensemble_dir.iterdir() if d.is_dir() and d.name.startswith('m')])
    
    if not member_dirs:
        raise ValueError(f"No ensemble member directories found in {ensemble_dir}")
    
    print(f"Loading {len(member_dirs)} ensemble members...")
    
    for member_dir in member_dirs:
        # Try different checkpoint naming patterns
        checkpoint_patterns = ['best_model.pth', 'best_*.pth', '*.pth']
        ckpt = None
        
        for pattern in checkpoint_patterns:
            ckpts = sorted(member_dir.glob(pattern))
            if ckpts:
                if pattern == '*.pth':
                    ckpts = [f for f in ckpts if 'best' in f.name.lower() or 'model' in f.name.lower()]
                if ckpts:
                    ckpt = ckpts[-1]
                    break
        
        if ckpt is None:
            print(f"Warning: No checkpoint found in {member_dir}")
            continue
            
        # Load model
        mcfg = cfg['model']
        net = UNet3D(mcfg['in_channels'], mcfg['out_channels'], base_ch=mcfg['base_channels'])
        state = torch.load(ckpt, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'model' in state:
            model_state = state['model']
        elif 'swa_model_state_dict' in state:
            model_state = state['swa_model_state_dict']
        elif 'model_state_dict' in state:
            model_state = state['model_state_dict']
        else:
            model_state = state
        
        # Handle DataParallel wrapper
        if any(k.startswith('module.') for k in model_state.keys()):
            model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
        
        net.load_state_dict(model_state)
        net = net.to(device)
        net.eval()
        
        models.append({
            'model': net,
            'member_id': member_dir.name,
            'checkpoint': ckpt.name
        })
        print(f"Loaded {member_dir.name}: {ckpt.name}")
    
    return models

def ensemble_predict(models, data_loader, device):
    """Make ensemble predictions and compute uncertainty metrics."""
    all_predictions = []  # List of prediction arrays from each model
    all_targets = []
    all_inputs = []
    
    print("Making ensemble predictions...")
    
    # Get predictions from each ensemble member
    for model_info in models:
        model = model_info['model']
        model.eval()
        
        predictions = []
        targets = []
        inputs = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Predicting {model_info['member_id']}"):
                if isinstance(batch, (list, tuple)):
                    input_data, target = batch
                else:
                    input_data = batch['input']
                    target = batch['target']
                
                input_data = input_data.to(device)
                target = target.to(device)
                
                # Forward pass
                pred = model(input_data)
                
                predictions.append(pred.cpu().numpy())
                targets.append(target.cpu().numpy())
                inputs.append(input_data.cpu().numpy())
        
        # Concatenate all batches for this model
        model_predictions = np.concatenate(predictions, axis=0)
        all_predictions.append(model_predictions)
        
        # Only need to store targets and inputs once
        if len(all_targets) == 0:
            all_targets = np.concatenate(targets, axis=0)
            all_inputs = np.concatenate(inputs, axis=0)
    
    # Convert to numpy arrays: [n_models, n_samples, ...]
    ensemble_predictions = np.stack(all_predictions, axis=0)
    
    return ensemble_predictions, all_targets, all_inputs

def compute_ensemble_metrics(ensemble_predictions, targets):
    """Compute ensemble statistics and uncertainty metrics."""
    # Ensemble statistics
    ensemble_mean = np.mean(ensemble_predictions, axis=0)  # Mean across models
    ensemble_var = np.var(ensemble_predictions, axis=0)    # Variance across models
    ensemble_std = np.std(ensemble_predictions, axis=0)    # Standard deviation
    
    # Compute metrics for ensemble mean
    mse = np.mean((ensemble_mean - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(ensemble_mean - targets))
    
    # R-squared
    ss_res = np.sum((targets - ensemble_mean) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Uncertainty metrics
    mean_predictive_variance = np.mean(ensemble_var)
    mean_predictive_std = np.mean(ensemble_std)
    
    # Calibration metrics (simplified)
    residuals = np.abs(ensemble_mean - targets)
    uncertainty = ensemble_std
    
    # Correlation between uncertainty and error
    if uncertainty.size > 0 and residuals.size > 0:
        uncertainty_flat = uncertainty.flatten()
        residuals_flat = residuals.flatten()
        
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(uncertainty_flat) & np.isfinite(residuals_flat)
        if np.sum(valid_mask) > 1:
            uncertainty_error_corr = np.corrcoef(
                uncertainty_flat[valid_mask], 
                residuals_flat[valid_mask]
            )[0, 1]
        else:
            uncertainty_error_corr = 0.0
    else:
        uncertainty_error_corr = 0.0
    
    # Coverage analysis (what fraction of predictions fall within 1, 2, 3 std)
    coverage_1std = np.mean(np.abs(ensemble_mean - targets) <= ensemble_std)
    coverage_2std = np.mean(np.abs(ensemble_mean - targets) <= 2 * ensemble_std)
    coverage_3std = np.mean(np.abs(ensemble_mean - targets) <= 3 * ensemble_std)
    
    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'mse': float(mse),
        'r2': float(r2),
        'mean_predictive_variance': float(mean_predictive_variance),
        'mean_predictive_std': float(mean_predictive_std),
        'uncertainty_error_correlation': float(uncertainty_error_corr),
        'coverage_1std': float(coverage_1std),
        'coverage_2std': float(coverage_2std),
        'coverage_3std': float(coverage_3std),
        'ensemble_size': ensemble_predictions.shape[0]
    }
    
    return metrics, ensemble_mean, ensemble_var, ensemble_std

def save_ensemble_outputs(save_dir, ensemble_mean, ensemble_var, ensemble_std, targets, inputs):
    """Save ensemble predictions and uncertainty maps."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy arrays
    np.save(save_dir / 'ensemble_mean.npy', ensemble_mean)
    np.save(save_dir / 'ensemble_variance.npy', ensemble_var)
    np.save(save_dir / 'ensemble_std.npy', ensemble_std)
    np.save(save_dir / 'targets.npy', targets)
    np.save(save_dir / 'inputs.npy', inputs)
    
    # Save central slice visualizations
    try:
        save_central_slices(save_dir, ensemble_mean, ensemble_std, targets, inputs)
    except Exception as e:
        print(f"Warning: Could not save central slices: {e}")
    
    print(f"Saved ensemble outputs to {save_dir}")

def save_central_slices(save_dir, predictions, uncertainty, targets, inputs):
    """Save central slice visualizations for ensemble results."""
    import matplotlib.pyplot as plt
    
    # Take first sample and middle slice
    if len(predictions.shape) == 5:  # [batch, channels, depth, height, width]
        sample_idx = 0
        depth_center = predictions.shape[2] // 2
        
        pred_slice = predictions[sample_idx, :, depth_center, :, :]  # [channels, H, W]
        unc_slice = uncertainty[sample_idx, :, depth_center, :, :]   # [channels, H, W] 
        target_slice = targets[sample_idx, :, depth_center, :, :]    # [channels, H, W]
        input_slice = inputs[sample_idx, :, depth_center, :, :]      # [channels, H, W]
        
        # Create visualization for each velocity component
        for ch in range(pred_slice.shape[0]):
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Input
            im1 = axes[0,0].imshow(input_slice[ch], cmap='RdBu_r')
            axes[0,0].set_title(f'Input (Channel {ch})')
            plt.colorbar(im1, ax=axes[0,0])
            
            # Target
            im2 = axes[0,1].imshow(target_slice[ch], cmap='RdBu_r')
            axes[0,1].set_title(f'Target (Channel {ch})')
            plt.colorbar(im2, ax=axes[0,1])
            
            # Ensemble Prediction
            im3 = axes[1,0].imshow(pred_slice[ch], cmap='RdBu_r')
            axes[1,0].set_title(f'Ensemble Prediction (Channel {ch})')
            plt.colorbar(im3, ax=axes[1,0])
            
            # Uncertainty
            im4 = axes[1,1].imshow(unc_slice[ch], cmap='plasma')
            axes[1,1].set_title(f'Predictive Uncertainty (Channel {ch})')
            plt.colorbar(im4, ax=axes[1,1])
            
            plt.tight_layout()
            plt.savefig(save_dir / f'central_slice_channel_{ch}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        print(f"Saved central slice visualizations to {save_dir}")
    else:
        print(f"Unexpected prediction shape: {predictions.shape}, skipping slice visualization")

def main(cfg_path, seed, mc_samples, temperature_scale, conformal, cuda):
    cfg = load_config(cfg_path)
    seed_all(seed or cfg.get('seed', 42))
    log = get_logger()
    
    exp_id = cfg.get('experiment_id', 'EXPERIMENT')
    base_out = Path(cfg['paths']['results_dir']) / exp_id
    
    # Check if this is an ensemble experiment
    if 'ensemble' not in exp_id.lower():
        print("Warning: This script is designed for ensemble experiments!")
        return
    
    # Find ensemble members directory
    ensemble_dir = base_out / exp_id / 'members'
    if not ensemble_dir.exists():
        # Try alternative structure
        ensemble_dir = base_out / 'members'
        if not ensemble_dir.exists():
            raise ValueError(f"No ensemble members directory found. Tried: {base_out / exp_id / 'members'} and {base_out / 'members'}")
    
    # Create output directory for ensemble results
    ensemble_out = base_out / 'ensemble_results'
    ensemble_out.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    val = ChannelDataset(cfg, 'val')
    test = ChannelDataset(cfg, 'test')
    val_loader = DataLoader(val, batch_size=1, shuffle=False)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)
    
    # Setup device
    device = pick_device(cuda)
    
    # Load all ensemble models
    models = load_ensemble_models(cfg, ensemble_dir, device)
    
    if not models:
        raise ValueError("No ensemble models could be loaded!")
    
    print(f"\nSuccessfully loaded {len(models)} ensemble members")
    
    # Evaluate on validation set
    print("\n=== Validation Set ===")
    val_predictions, val_targets, val_inputs = ensemble_predict(models, val_loader, device)
    val_metrics, val_mean, val_var, val_std = compute_ensemble_metrics(val_predictions, val_targets)
    
    print(f"VAL Ensemble RMSE: {val_metrics['rmse']:.4f}")
    print(f"VAL Ensemble MAE: {val_metrics['mae']:.4f}")
    print(f"VAL Mean Uncertainty (std): {val_metrics['mean_predictive_std']:.4f}")
    print(f"VAL Coverage 1σ: {val_metrics['coverage_1std']:.3f}")
    print(f"VAL Coverage 2σ: {val_metrics['coverage_2std']:.3f}")
    
    # Evaluate on test set
    print("\n=== Test Set ===")
    test_predictions, test_targets, test_inputs = ensemble_predict(models, test_loader, device)
    test_metrics, test_mean, test_var, test_std = compute_ensemble_metrics(test_predictions, test_targets)
    
    print(f"TEST Ensemble RMSE: {test_metrics['rmse']:.4f}")
    print(f"TEST Ensemble MAE: {test_metrics['mae']:.4f}")
    print(f"TEST Mean Uncertainty (std): {test_metrics['mean_predictive_std']:.4f}")
    print(f"TEST Coverage 1σ: {test_metrics['coverage_1std']:.3f}")
    print(f"TEST Coverage 2σ: {test_metrics['coverage_2std']:.3f}")
    
    # Save metrics
    val_metrics_json = dict(val_metrics, split='val')
    test_metrics_json = dict(test_metrics, split='test')
    
    with open(ensemble_out / 'val_ensemble_metrics.json', 'w') as f:
        json.dump(val_metrics_json, f, indent=2)
    with open(ensemble_out / 'test_ensemble_metrics.json', 'w') as f:
        json.dump(test_metrics_json, f, indent=2)
    
    # Save ensemble outputs
    save_ensemble_outputs(ensemble_out / 'val', val_mean, val_var, val_std, val_targets, val_inputs)
    save_ensemble_outputs(ensemble_out / 'test', test_mean, test_var, test_std, test_targets, test_inputs)
    
    # Save summary
    summary = {
        'experiment_id': exp_id,
        'ensemble_size': len(models),
        'ensemble_members': [m['member_id'] for m in models],
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics
    }
    
    with open(ensemble_out / 'ensemble_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Results Summary ===")
    print(f"Ensemble size: {len(models)} members")
    print(f"Results saved to: {ensemble_out}")
    print(f"Validation RMSE: {val_metrics['rmse']:.4f} ± {val_metrics['mean_predictive_std']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f} ± {test_metrics['mean_predictive_std']:.4f}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--mc-samples', type=int, default=None)
    ap.add_argument('--temperature-scale', action='store_true')
    ap.add_argument('--conformal', action='store_true')
    ap.add_argument('--cuda', action='store_true', help='use CUDA if available')
    a = ap.parse_args()
    main(a.config, a.seed, a.mc_samples, a.temperature_scale, a.conformal, a.cuda)