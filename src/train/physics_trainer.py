import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import time
from .losses import physics_informed_loss

def physics_train_loop(net, train_loader, val_loader, optimizer, scheduler, 
                      output_dir, logger, cfg, resume_path=None, device='cuda'):
    """Physics-informed training loop with Navier-Stokes constraints."""
    
    # Training parameters
    epochs = cfg['train']['epochs']
    save_every = cfg['train'].get('save_every', 10)
    early_stop_patience = cfg['train'].get('early_stop_patience', 20)
    use_amp = cfg['train'].get('amp', True)
    
    # Physics loss weights
    physics_weight = cfg['train'].get('physics_weight', 0.1)
    continuity_weight = cfg['train'].get('continuity_weight', 0.05)
    momentum_weight = cfg['train'].get('momentum_weight', 0.05)
    
    # Initialize training state
    start_epoch = 1
    best_val_loss = float('inf')
    patience_counter = 0
    scaler = GradScaler() if use_amp else None
    
    # Resume from checkpoint if provided
    if resume_path and Path(resume_path).exists():
        logger.info(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        logger.info(f"Resumed from epoch {start_epoch-1}, best_val_loss: {best_val_loss:.6f}")
    
    logger.info(f"Starting physics-informed training from epoch {start_epoch}")
    logger.info(f"Physics weights - physics: {physics_weight}, continuity: {continuity_weight}, momentum: {momentum_weight}")
    
    for epoch in range(start_epoch, epochs + 1):
        # Training phase
        net.train()
        train_loss = 0.0
        train_data_loss = 0.0
        train_physics_loss = 0.0
        train_continuity_loss = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    pred = net(data)
                    total_loss, data_loss, physics_loss, continuity_loss = physics_informed_loss(
                        pred, target, physics_weight=physics_weight,
                        continuity_weight=continuity_weight, momentum_weight=momentum_weight
                    )
                
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = net(data)
                total_loss, data_loss, physics_loss, continuity_loss = physics_informed_loss(
                    pred, target, physics_weight=physics_weight,
                    continuity_weight=continuity_weight, momentum_weight=momentum_weight
                )
                total_loss.backward()
                optimizer.step()
            
            train_loss += total_loss.item()
            train_data_loss += data_loss.item()
            train_physics_loss += physics_loss.item()
            train_continuity_loss += continuity_loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}: '
                           f'Loss={total_loss.item():.6f}, Data={data_loss.item():.6f}, '
                           f'Physics={physics_loss.item():.6f}, Continuity={continuity_loss.item():.6f}')
        
        # Average training losses
        train_loss /= num_batches
        train_data_loss /= num_batches
        train_physics_loss /= num_batches
        train_continuity_loss /= num_batches
        
        # Validation phase
        net.eval()
        val_loss = 0.0
        val_data_loss = 0.0
        val_physics_loss = 0.0
        val_continuity_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                if use_amp:
                    with autocast():
                        pred = net(data)
                        total_loss, data_loss, physics_loss, continuity_loss = physics_informed_loss(
                            pred, target, physics_weight=physics_weight,
                            continuity_weight=continuity_weight, momentum_weight=momentum_weight
                        )
                else:
                    pred = net(data)
                    total_loss, data_loss, physics_loss, continuity_loss = physics_informed_loss(
                        pred, target, physics_weight=physics_weight,
                        continuity_weight=continuity_weight, momentum_weight=momentum_weight
                    )
                
                val_loss += total_loss.item()
                val_data_loss += data_loss.item()
                val_physics_loss += physics_loss.item()
                val_continuity_loss += continuity_loss.item()
                num_val_batches += 1
        
        # Average validation losses
        val_loss /= num_val_batches
        val_data_loss /= num_val_batches
        val_physics_loss /= num_val_batches
        val_continuity_loss /= num_val_batches
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Log epoch results
        logger.info(f'Epoch {epoch:3d}/{epochs} ({epoch_time:.1f}s): '
                   f'Train Loss={train_loss:.6f} (Data={train_data_loss:.6f}, Physics={train_physics_loss:.6f}, Cont={train_continuity_loss:.6f}) | '
                   f'Val Loss={val_loss:.6f} (Data={val_data_loss:.6f}, Physics={val_physics_loss:.6f}, Cont={val_continuity_loss:.6f}) | '
                   f'LR={optimizer.param_groups[0]["lr"]:.2e}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'train_loss': train_loss
            }, output_dir / 'best_model.pth')
            logger.info(f'New best model saved with val_loss: {val_loss:.6f}')
        else:
            patience_counter += 1
        
        # Save periodic checkpoint
        if epoch % save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'train_loss': train_loss
            }, checkpoint_path)
            logger.info(f'Checkpoint saved: {checkpoint_path}')
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            logger.info(f'Early stopping triggered after {patience_counter} epochs without improvement')
            break
    
    logger.info(f'Training completed. Best validation loss: {best_val_loss:.6f}')
    return best_val_loss
