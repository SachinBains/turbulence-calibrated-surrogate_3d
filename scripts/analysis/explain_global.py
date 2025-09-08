#!/usr/bin/env python3
"""
Global feature importance analysis using engineered features.
Computes turbulence features and relates them to model error/uncertainty.
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import torch

from src.utils.config import load_config
from src.utils.devices import pick_device
from src.dataio.channel_dataset import ChannelDataset
from src.models.unet3d import UNet3D

def load_model(cfg, device):
    """Load trained model from checkpoint."""
    exp_id = cfg['experiment_id']
    results_dir = Path(cfg['paths']['results_dir']) / exp_id
    
    # Find best checkpoint
    ckpts = sorted(results_dir.glob('best_*.pth'))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {results_dir}")
    
    ckpt_path = ckpts[-1]
    
    # Build model
    mcfg = cfg['model']
    model = UNet3D(
        mcfg['in_channels'], 
        mcfg['out_channels'], 
        base_ch=mcfg['base_channels']
    )
    
    # Load weights
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'])
    model = model.to(device)
    model.eval()
    
    return model

def compute_turbulence_features(x):
    """
    Compute engineered turbulence features from velocity field.
    
    Args:
        x: Input tensor (1, C, D, H, W) where C includes velocity components
    
    Returns:
        Dictionary of features
    """
    # Convert to numpy for easier computation
    if isinstance(x, torch.Tensor):
        x_np = x.cpu().numpy()
    else:
        x_np = x
    
    # Assume first 3 channels are u, v, w velocity components
    u = x_np[0, 0] if x_np.shape[1] > 0 else np.zeros_like(x_np[0, 0])
    v = x_np[0, 1] if x_np.shape[1] > 1 else np.zeros_like(x_np[0, 0])
    w = x_np[0, 2] if x_np.shape[1] > 2 else np.zeros_like(x_np[0, 0])
    
    features = {}
    
    # Basic statistics
    features['u_mean'] = np.mean(u)
    features['v_mean'] = np.mean(v)
    features['w_mean'] = np.mean(w)
    features['u_std'] = np.std(u)
    features['v_std'] = np.std(v)
    features['w_std'] = np.std(w)
    
    # Velocity magnitude statistics
    vel_mag = np.sqrt(u**2 + v**2 + w**2)
    features['vel_mag_mean'] = np.mean(vel_mag)
    features['vel_mag_std'] = np.std(vel_mag)
    features['vel_mag_max'] = np.max(vel_mag)
    
    # Compute gradients (approximate derivatives)
    try:
        # Gradients in each direction
        du_dx, du_dy, du_dz = np.gradient(u)
        dv_dx, dv_dy, dv_dz = np.gradient(v)
        dw_dx, dw_dy, dw_dz = np.gradient(w)
        
        # Vorticity components (curl of velocity)
        omega_x = dw_dy - dv_dz
        omega_y = du_dz - dw_dx
        omega_z = dv_dx - du_dy
        
        # Enstrophy (vorticity magnitude squared)
        enstrophy = omega_x**2 + omega_y**2 + omega_z**2
        features['enstrophy_mean'] = np.mean(enstrophy)
        features['enstrophy_std'] = np.std(enstrophy)
        features['enstrophy_max'] = np.max(enstrophy)
        
        # Strain rate tensor components
        s11 = du_dx
        s22 = dv_dy
        s33 = dw_dz
        s12 = 0.5 * (du_dy + dv_dx)
        s13 = 0.5 * (du_dz + dw_dx)
        s23 = 0.5 * (dv_dz + dw_dy)
        
        # Strain rate magnitude
        strain_mag = np.sqrt(2 * (s11**2 + s22**2 + s33**2 + 2*(s12**2 + s13**2 + s23**2)))
        features['strain_rate_mean'] = np.mean(strain_mag)
        features['strain_rate_std'] = np.std(strain_mag)
        
        # Divergence
        divergence = du_dx + dv_dy + dw_dz
        features['divergence_mean'] = np.mean(np.abs(divergence))
        features['divergence_std'] = np.std(divergence)
        
    except Exception as e:
        print(f"Warning: Could not compute gradient-based features: {e}")
        # Set default values
        for key in ['enstrophy_mean', 'enstrophy_std', 'enstrophy_max', 
                   'strain_rate_mean', 'strain_rate_std', 'divergence_mean', 'divergence_std']:
            features[key] = 0.0
    
    # Spectral features (simplified)
    try:
        # FFT of velocity magnitude
        vel_fft = np.fft.fftn(vel_mag)
        power_spectrum = np.abs(vel_fft)**2
        
        # Radial binning (simplified - just use percentiles)
        power_flat = power_spectrum.flatten()
        features['spectral_low'] = np.percentile(power_flat, 25)
        features['spectral_mid'] = np.percentile(power_flat, 50)
        features['spectral_high'] = np.percentile(power_flat, 75)
        features['spectral_energy_total'] = np.sum(power_spectrum)
        
    except Exception as e:
        print(f"Warning: Could not compute spectral features: {e}")
        for key in ['spectral_low', 'spectral_mid', 'spectral_high', 'spectral_energy_total']:
            features[key] = 0.0
    
    return features

def load_mc_predictions(cfg, split):
    """Load MC predictions if available."""
    exp_id = cfg['experiment_id']
    results_dir = Path(cfg['paths']['results_dir']) / exp_id
    
    mean_path = results_dir / f"mc_mean_{split}.npy"
    var_path = results_dir / f"mc_var_{split}.npy"
    
    if mean_path.exists() and var_path.exists():
        mean = np.load(mean_path)
        var = np.load(var_path)
        sigma = np.sqrt(var)
        return mean, sigma
    else:
        return None, None

def compute_model_predictions(model, loader, device):
    """Compute model predictions for all samples."""
    predictions = []
    
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            pred = model(x)
            predictions.append(pred.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)

def main():
    parser = argparse.ArgumentParser(description='Global feature importance analysis')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--split', choices=['val', 'test'], default='test', help='Dataset split')
    parser.add_argument('--target', choices=['error', 'sigma'], required=True, 
                       help='Target variable (error for baseline, sigma for MC)')
    parser.add_argument('--n_samples', type=int, default=50, help='Number of samples to analyze')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load config and setup paths
    cfg = load_config(args.config)
    exp_id = cfg['experiment_id']
    device = pick_device(args.cuda)
    
    results_dir = Path(cfg['paths']['results_dir']) / exp_id / 'global'
    figures_dir = Path(cfg['paths']['artifacts_root']) / 'figures' / exp_id / 'global'
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = ChannelDataset(cfg, args.split, eval_mode=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load model
    model = load_model(cfg, device)
    
    print(f"Computing features for {min(args.n_samples, len(dataset))} samples...")
    
    # Collect features and targets
    features_list = []
    targets = []
    
    # Get ground truth
    y_true = []
    for _, y in loader:
        y_true.append(y.numpy())
    y_true = np.concatenate(y_true, axis=0)
    
    # Get model predictions
    predictions = compute_model_predictions(model, loader, device)
    
    # Load MC predictions if needed
    if args.target == 'sigma':
        mc_mean, mc_sigma = load_mc_predictions(cfg, args.split)
        if mc_sigma is None:
            raise ValueError(f"MC predictions not found for {exp_id}. Run MC prediction first.")
    
    # Process samples
    sample_count = 0
    for i, (x, y) in enumerate(loader):
        if sample_count >= args.n_samples:
            break
        
        # Compute features
        features = compute_turbulence_features(x)
        features_list.append(features)
        
        # Compute target
        if args.target == 'error':
            # Absolute error
            error = np.abs(predictions[i] - y_true[i]).mean()
            targets.append(error)
        elif args.target == 'sigma':
            # MC uncertainty (mean sigma)
            sigma_mean = mc_sigma[i].mean()
            targets.append(sigma_mean)
        
        sample_count += 1
        
        if (sample_count % 10) == 0:
            print(f"  Processed {sample_count}/{args.n_samples} samples")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    features_df['target'] = targets
    
    print(f"Computed {len(features_df.columns)-1} features for {len(features_df)} samples")
    
    # Prepare data for modeling
    X = features_df.drop('target', axis=1)
    y = features_df['target']
    
    # Handle any NaN or infinite values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=args.seed)
    rf.fit(X_scaled, y)
    
    # Compute permutation importance
    perm_importance = permutation_importance(rf, X_scaled, y, n_repeats=10, random_state=args.seed)
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    # Save importance results
    csv_path = results_dir / 'importance.csv'
    importance_df.to_csv(csv_path, index=False)
    print(f"Saved importance results to: {csv_path}")
    
    # Create importance bar plot
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(15)  # Top 15 features
    
    plt.barh(range(len(top_features)), top_features['importance_mean'], 
             xerr=top_features['importance_std'], capsize=3)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Permutation Importance')
    plt.title(f'Global Feature Importance ({args.target})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save plot
    plot_path = figures_dir / 'importance_bar.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved importance plot to: {plot_path}")
    
    # Create ALE-style plots for top 3 features
    top_3_features = top_features.head(3)['feature'].tolist()
    
    for feature in top_3_features:
        plt.figure(figsize=(8, 6))
        
        # Sort by feature value and compute running average of target
        sorted_indices = np.argsort(X[feature])
        sorted_feature = X[feature].iloc[sorted_indices]
        sorted_target = y.iloc[sorted_indices]
        
        # Compute moving average
        window_size = max(5, len(sorted_target) // 20)
        moving_avg = pd.Series(sorted_target).rolling(window=window_size, center=True).mean()
        
        plt.plot(sorted_feature, moving_avg, 'b-', linewidth=2, label='Moving Average')
        plt.scatter(sorted_feature, sorted_target, alpha=0.3, s=10, label='Data Points')
        
        plt.xlabel(feature)
        plt.ylabel(f'Target ({args.target})')
        plt.title(f'Feature Effect: {feature}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save ALE plot
        ale_path = figures_dir / f'ale_{feature}.png'
        plt.savefig(ale_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved ALE plot to: {ale_path}")
    
    # Print summary
    print(f"\nTop 10 Most Important Features ({args.target}):")
    print("Feature                | Importance ± Std")
    print("--------------------- |------------------")
    for _, row in importance_df.head(10).iterrows():
        print(f"{row['feature']:20s} | {row['importance_mean']:8.6f} ± {row['importance_std']:6.6f}")
    
    print(f"\nModel R² Score: {rf.score(X_scaled, y):.3f}")
    print(f"Results saved to: {results_dir}")
    print(f"Figures saved to: {figures_dir}")

if __name__ == '__main__':
    main()
