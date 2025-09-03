#!/usr/bin/env python3
"""
Train model with Stochastic Weight Averaging (SWA).
"""
import os
import sys
import argparse
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.devices import pick_device
from src.dataio.channel_dataset import ChannelDataset
from src.models.unet3d import UNet3D
from src.train.swa import (
    create_swa_model, create_swa_scheduler, update_bn_stats,
    swa_train_epoch, evaluate_swa_model, save_swa_checkpoint
)

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_model(cfg: dict, device: torch.device) -> nn.Module:
    """Build model from config."""
    model_cfg = cfg['model']
    dropout_p = cfg.get('uq', {}).get('dropout_p', 0.0)
    
    model = UNet3D(
        in_channels=model_cfg['in_channels'],
        out_channels=model_cfg['out_channels'],
        base_channels=model_cfg['base_channels'],
        dropout_p=dropout_p
    ).to(device)
    
    return model

def build_dataloaders(cfg: dict) -> tuple:
    """Build train and validation dataloaders."""
    train_dataset = ChannelDataset(cfg['dataset']['data_dir'], 'train')
    val_dataset = ChannelDataset(cfg['dataset']['data_dir'], 'val')
    
    batch_size = cfg['train']['batch_size']
    num_workers = cfg['train'].get('num_workers', 0)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    
    return train_loader, val_loader

def load_pretrained_weights(model: nn.Module, pretrained_path: str, device: torch.device):
    """Load pretrained weights into model."""
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")
    
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Loaded pretrained weights from: {pretrained_path}")

def main():
    parser = argparse.ArgumentParser(description='Train with SWA')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--pretrained', required=True, help='Pretrained model checkpoint')
    parser.add_argument('--swa_start', type=int, default=10, help='Epoch to start SWA')
    parser.add_argument('--swa_lr', type=float, default=0.05, help='SWA learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()
    
    # Load config and setup device
    cfg = load_config(args.config)
    device = pick_device(args.cuda)
    
    # Setup paths
    exp_id = cfg['experiment_id'] + '_swa'
    results_dir = Path(cfg['paths']['results_dir']) / exp_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Build model and load pretrained weights
    model = build_model(cfg, device)
    load_pretrained_weights(model, args.pretrained, device)
    
    # Create SWA model
    swa_model = create_swa_model(model)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['learning_rate'])
    swa_scheduler = create_swa_scheduler(optimizer, args.swa_start, args.swa_lr)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Build dataloaders
    train_loader, val_loader = build_dataloaders(cfg)
    
    # Training loop
    best_val_loss = float('inf')
    training_log = []
    
    print(f"Starting SWA training for {args.epochs} epochs")
    print(f"SWA will start at epoch {args.swa_start}")
    print(f"Device: {device}")
    
    for epoch in range(args.epochs):
        # Train epoch
        train_metrics = swa_train_epoch(
            model, swa_model, optimizer, swa_scheduler, criterion,
            train_loader, device, epoch, args.swa_start
        )
        
        # Evaluate on validation set
        if epoch >= args.swa_start:
            # Update BN stats before evaluation
            update_bn_stats(swa_model, train_loader, device)
            val_metrics = evaluate_swa_model(swa_model, criterion, val_loader, device)
        else:
            # Evaluate regular model before SWA starts
            model.eval()
            total_loss = 0.0
            num_samples = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item() * inputs.size(0)
                    num_samples += inputs.size(0)
            val_metrics = {'val_loss': total_loss / num_samples if num_samples > 0 else 0.0}
        
        # Combine metrics
        epoch_metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
        training_log.append(epoch_metrics)
        
        # Print progress
        swa_status = "SWA" if train_metrics['swa_active'] else "Regular"
        print(f"Epoch {epoch:3d} [{swa_status:7s}] | "
              f"Train Loss: {train_metrics['train_loss']:.6f} | "
              f"Val Loss: {val_metrics['val_loss']:.6f}")
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            
            if epoch >= args.swa_start:
                # Save SWA model
                save_swa_checkpoint(
                    swa_model, optimizer, swa_scheduler, epoch, epoch_metrics,
                    results_dir / 'best_swa_model.pth'
                )
            else:
                # Save regular model
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'metrics': epoch_metrics
                }, results_dir / 'best_model.pth')
    
    # Final BN update and save
    if args.epochs >= args.swa_start:
        update_bn_stats(swa_model, train_loader, device)
        save_swa_checkpoint(
            swa_model, optimizer, swa_scheduler, args.epochs - 1, training_log[-1],
            results_dir / 'final_swa_model.pth'
        )
    
    # Save training log
    with open(results_dir / 'swa_training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    # Save SWA config
    swa_config = {
        'base_config': args.config,
        'pretrained_model': args.pretrained,
        'swa_start': args.swa_start,
        'swa_lr': args.swa_lr,
        'total_epochs': args.epochs,
        'best_val_loss': best_val_loss,
        'experiment_id': exp_id
    }
    
    with open(results_dir / 'swa_config.json', 'w') as f:
        json.dump(swa_config, f, indent=2)
    
    print(f"\nSWA training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Results saved to: {results_dir}")

if __name__ == '__main__':
    main()
