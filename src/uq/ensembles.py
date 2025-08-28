"""
Deep ensemble utilities for uncertainty quantification.
"""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from src.models.unet3d import UNet3D
from src.train.trainer import train_loop
from src.train.losses import make_loss
from src.utils.seeding import seed_all
from src.utils.logging import get_logger
import json

def build_member(cfg, seed):
    """Build a fresh ensemble member with given seed."""
    seed_all(seed)
    
    mcfg = cfg['model']
    dropout_p = cfg.get('uq', {}).get('dropout_p', 0.0)
    
    net = UNet3D(
        mcfg['in_channels'],
        mcfg['out_channels'],
        mcfg['base_channels'],
        dropout_p
    )
    
    return net

def train_ensemble(cfg, seeds, device, train_loader, val_loader, results_dir):
    """Train ensemble members with different seeds."""
    logger = get_logger()
    members_dir = Path(results_dir) / 'members'
    members_dir.mkdir(parents=True, exist_ok=True)
    
    member_paths = []
    
    for i, seed in enumerate(seeds):
        logger.info(f"Training ensemble member {i:02d} with seed {seed}")
        
        # Build member
        net = build_member(cfg, seed)
        net = net.to(device)
        
        # Setup training
        criterion = make_loss(cfg)
        optimizer = torch.optim.AdamW(
            net.parameters(), 
            lr=cfg['train']['lr'], 
            weight_decay=cfg['train']['weight_decay']
        )
        scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg['train']['amp']))
        
        # Member output directory
        member_dir = members_dir / f'm{i:02d}'
        member_dir.mkdir(exist_ok=True)
        
        # Train member
        best_path = train_loop(
            cfg, net, criterion, optimizer, scaler, 
            train_loader, val_loader, member_dir, logger, 
            device=device
        )
        
        if best_path:
            member_paths.append(best_path)
            logger.info(f"Member {i:02d} best checkpoint: {best_path}")
        else:
            logger.warning(f"Member {i:02d} training failed")
    
    return member_paths

def ensemble_predict(model_paths, dataloader, device, cfg):
    """Run ensemble prediction and compute statistics."""
    predictions = []
    
    # Load all ensemble members
    models = []
    for path in model_paths:
        net = build_member(cfg, seed=42)  # Seed doesn't matter for inference
        checkpoint = torch.load(path, map_location=device)
        net.load_state_dict(checkpoint['model'])
        net = net.to(device)
        net.eval()
        models.append(net)
    
    # Collect predictions from all members
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            
            batch_preds = []
            for model in models:
                pred = model(x)
                batch_preds.append(pred.cpu().numpy())
            
            # Stack predictions: (n_members, batch_size, ...)
            batch_preds = np.stack(batch_preds, axis=0)
            predictions.append(batch_preds)
    
    # Concatenate all batches: (n_members, n_samples, ...)
    all_predictions = np.concatenate(predictions, axis=1)
    
    # Compute ensemble statistics
    ens_mean = np.mean(all_predictions, axis=0)  # (n_samples, ...)
    ens_var = np.var(all_predictions, axis=0)    # (n_samples, ...)
    
    return ens_mean, ens_var, all_predictions

def compute_ensemble_metrics(ens_mean, y_true):
    """Compute per-sample metrics for ensemble predictions."""
    # Flatten for per-sample metrics
    ens_mean_flat = ens_mean.reshape(len(ens_mean), -1)
    y_true_flat = y_true.reshape(len(y_true), -1)
    
    metrics = []
    for i in range(len(ens_mean)):
        pred_i = ens_mean_flat[i]
        true_i = y_true_flat[i]
        
        # RMSE and MAE per sample
        mse = np.mean((pred_i - true_i) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_i - true_i))
        
        metrics.append({
            'sample_idx': i,
            'rmse': float(rmse),
            'mae': float(mae)
        })
    
    # Overall metrics
    overall_rmse = np.sqrt(np.mean((ens_mean_flat - y_true_flat) ** 2))
    overall_mae = np.mean(np.abs(ens_mean_flat - y_true_flat))
    
    return {
        'per_sample': metrics,
        'overall': {
            'rmse': float(overall_rmse),
            'mae': float(overall_mae),
            'n_samples': len(ens_mean)
        }
    }
