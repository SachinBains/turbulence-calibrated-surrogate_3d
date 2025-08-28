"""
Stochastic Weight Averaging (SWA) utilities for improved generalization.
"""
import os
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Tuple

def create_swa_model(model: nn.Module) -> AveragedModel:
    """Create SWA averaged model wrapper."""
    return AveragedModel(model)

def create_swa_scheduler(optimizer, swa_start: int, swa_lr: float = 0.05):
    """Create SWA learning rate scheduler."""
    return SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=10, anneal_strategy='cos')

def update_bn_stats(swa_model: AveragedModel, loader: DataLoader, device: torch.device):
    """Update batch normalization statistics for SWA model."""
    swa_model.train()
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            _ = swa_model(inputs)

def swa_train_epoch(model: nn.Module, swa_model: AveragedModel, 
                   optimizer, swa_scheduler, criterion, 
                   train_loader: DataLoader, device: torch.device,
                   epoch: int, swa_start: int) -> Dict[str, float]:
    """
    Train one epoch with SWA updates.
    
    Args:
        model: Base model
        swa_model: SWA averaged model
        optimizer: Optimizer
        swa_scheduler: SWA scheduler
        criterion: Loss function
        train_loader: Training data loader
        device: Device
        epoch: Current epoch
        swa_start: Epoch to start SWA
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update SWA model if we're in SWA phase
        if epoch >= swa_start:
            swa_model.update_parameters(model)
    
    # Update SWA scheduler if in SWA phase
    if epoch >= swa_start:
        swa_scheduler.step()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        'train_loss': avg_loss,
        'swa_active': epoch >= swa_start
    }

def evaluate_swa_model(swa_model: AveragedModel, criterion, 
                      val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate SWA model on validation set."""
    swa_model.eval()
    total_loss = 0.0
    total_mse = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = swa_model(inputs)
            
            loss = criterion(outputs, targets)
            mse = ((outputs - targets) ** 2).mean()
            
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_mse += mse.item() * batch_size
            num_samples += batch_size
    
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    avg_mse = total_mse / num_samples if num_samples > 0 else 0.0
    rmse = np.sqrt(avg_mse)
    
    return {
        'val_loss': avg_loss,
        'val_mse': avg_mse,
        'val_rmse': rmse
    }

def save_swa_checkpoint(swa_model: AveragedModel, optimizer, swa_scheduler,
                       epoch: int, metrics: Dict[str, float], 
                       save_path: str):
    """Save SWA checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'swa_model_state_dict': swa_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'swa_scheduler_state_dict': swa_scheduler.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)

def load_swa_checkpoint(swa_model: AveragedModel, optimizer, swa_scheduler,
                       checkpoint_path: str, device: torch.device) -> Tuple[int, Dict[str, float]]:
    """Load SWA checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    swa_model.load_state_dict(checkpoint['swa_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    swa_scheduler.load_state_dict(checkpoint['swa_scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    metrics = checkpoint.get('metrics', {})
    
    return epoch, metrics

def swa_predict(swa_model: AveragedModel, loader: DataLoader, 
               device: torch.device) -> np.ndarray:
    """Make predictions using SWA model."""
    swa_model.eval()
    predictions = []
    
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = swa_model(inputs)
            predictions.append(outputs.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)
