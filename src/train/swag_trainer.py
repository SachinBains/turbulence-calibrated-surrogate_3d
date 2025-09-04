"""
SWAG (Stochastic Weight Averaging Gaussian) training loop.
"""
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json

def swag_train_loop(net, train_loader, val_loader, optimizer, scheduler, 
                   output_dir, logger, cfg, resume_path=None, device='cuda'):
    """Training loop for SWAG models with weight averaging."""
    
    # Training parameters
    epochs = cfg['train']['epochs']
    log_interval = cfg.get('log_interval', 100)
    
    # SWAG parameters
    swag_start = cfg['model'].get('swag_start', 100)
    swag_lr = cfg['model'].get('swag_lr', 0.00001)
    swag_c_epochs = cfg['model'].get('swag_c_epochs', 1)
    
    # Early stopping
    early_stop = cfg['train'].get('early_stopping', {})
    patience = early_stop.get('patience', 20)
    min_delta = early_stop.get('min_delta', 1e-6)
    
    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = 0
    
    # Create SWA model
    swa_model = AveragedModel(net)
    swa_scheduler = SWALR(optimizer, swa_lr=swag_lr, anneal_epochs=10)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    logger.info(f"Starting SWAG training for {epochs} epochs")
    logger.info(f"SWAG collection starts at epoch {swag_start}")
    logger.info(f"SWAG learning rate: {swag_lr}")
    
    for epoch in range(start_epoch, epochs):
        # Training phase
        net.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                output = net(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # Log training progress
            if batch_idx % log_interval == 0:
                logger.info(
                    f'Epoch {epoch}, Step {batch_idx}: '
                    f'loss={loss.item():.6f}'
                )
        
        # Update learning rate
        if epoch < swag_start:
            scheduler.step()
        else:
            swa_scheduler.step()
            
        # Update SWA model (collect weights)
        if epoch >= swag_start and (epoch - swag_start) % swag_c_epochs == 0:
            swa_model.update_parameters(net)
            logger.info(f"Updated SWAG model at epoch {epoch}")
        
        # Validation phase
        if epoch >= swag_start:
            # Use SWA model for validation
            swa_model.eval()
            # Update batch norm statistics
            update_bn_stats(swa_model, train_loader, device)
            
            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    with torch.cuda.amp.autocast():
                        output = swa_model(data)
                        loss = criterion(output, target)
                    val_loss += loss.item()
        else:
            # Use regular model for validation before SWA starts
            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    with torch.cuda.amp.autocast():
                        output = net(data)
                        loss = criterion(output, target)
                    val_loss += loss.item()
        
        # Average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Log epoch results
        logger.info(
            f'Epoch {epoch}: '
            f'train_loss={train_loss:.6f}, '
            f'val_loss={val_loss:.6f}'
        )
        
        # Early stopping and checkpointing
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            if epoch >= swag_start:
                # Save SWA model
                torch.save({
                    'epoch': epoch,
                    'swa_model_state_dict': swa_model.state_dict(),
                    'base_model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'swa_scheduler_state_dict': swa_scheduler.state_dict(),
                    'val_loss': val_loss,
                    'swag_start': swag_start,
                    'n_averaged': swa_model.n_averaged
                }, output_dir / 'best_model.pth')
            else:
                # Save regular model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss
                }, output_dir / 'best_model.pth')
                
            logger.info(f"New best model saved with val_loss: {val_loss:.6f}")
        else:
            patience_counter += 1
            
        # Early stopping check
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {patience} epochs without improvement")
            break
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            if epoch >= swag_start:
                torch.save({
                    'epoch': epoch,
                    'swa_model_state_dict': swa_model.state_dict(),
                    'base_model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'swa_scheduler_state_dict': swa_scheduler.state_dict(),
                    'val_loss': val_loss,
                    'swag_start': swag_start,
                    'n_averaged': swa_model.n_averaged
                }, output_dir / f'checkpoint_epoch_{epoch}.pth')
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss
                }, output_dir / f'checkpoint_epoch_{epoch}.pth')
    
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    return best_val_loss

def update_bn_stats(swa_model, loader, device):
    """Update batch normalization statistics for SWA model."""
    swa_model.train()
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            _ = swa_model(data)
