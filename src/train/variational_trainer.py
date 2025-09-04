import torch
from src.train.trainer import train_loop as base_train_loop
from src.utils.logging import get_logger
import os

def variational_train_loop(cfg, net, criterion, optimizer, scaler, train_loader, val_loader, 
                          output_dir, logger, resume_path=None, device='cuda'):
    """Training loop for variational models with KL divergence and β warm-up."""
    
    # Training parameters
    epochs = cfg['train']['epochs']
    log_interval = cfg.get('log_interval', 100)
    
    # Calculate total samples per epoch for KL normalization
    samples_per_epoch = len(train_loader) * train_loader.batch_size
    
    # β warm-up schedule parameters (BNN-appropriate scale)
    warmup_epochs = 30
    beta_start = 1e-7
    beta_end = 6e-7
    
    # Early stopping
    early_stop = cfg['train'].get('early_stopping', {})
    patience = early_stop.get('patience', 20)
    min_delta = early_stop.get('min_delta', 1e-6)
    
    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = 0
    
    # Resume from checkpoint if provided
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}")
    
    for epoch in range(start_epoch, epochs):
        # Update β (KL weight) with warm-up schedule
        if epoch < warmup_epochs:
            beta = beta_start + (beta_end - beta_start) * (epoch / warmup_epochs)
        else:
            beta = beta_end
        
        # Update model's kl_weight for logging compatibility
        net.kl_weight = beta
        
        logger.info(f'Epoch {epoch:03d} | β (KL weight): {beta:.6f}')
        
        # Training phase
        net.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_raw = 0.0
        train_kl_per_sample = 0.0
        train_kl_weighted = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                output = net(data)
                recon_loss = criterion(output, target)
                
                # Compute raw KL and normalize per sample
                kl_raw = net.raw_kl_divergence()
                kl_per_sample = kl_raw / samples_per_epoch
                kl_weighted = beta * kl_per_sample
                
                total_loss = recon_loss + kl_weighted
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += total_loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_raw += kl_raw.item()
            train_kl_per_sample += kl_per_sample.item()
            train_kl_weighted += kl_weighted.item()
            
            # Log training progress with all KL components and ratio
            if batch_idx % log_interval == 0:
                kl_recon_ratio = kl_weighted.item() / recon_loss.item() if recon_loss.item() > 0 else 0
                logger.info(
                    f'Step {epoch * len(train_loader) + batch_idx}: '
                    f'recon_loss={recon_loss.item():.6f}, '
                    f'kl_raw={kl_raw.item():.1f}, '
                    f'kl_per_sample={kl_per_sample.item():.1f}, '
                    f'kl_weighted={kl_weighted.item():.6f}, '
                    f'kl/recon={kl_recon_ratio:.3f}, '
                    f'total_loss={total_loss.item():.6f}'
                )
        
        # Validation phase
        net.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_weighted = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = net(data)
                recon_loss = criterion(output, target)
                
                # Use same KL computation as training
                kl_raw = net.raw_kl_divergence()
                kl_per_sample = kl_raw / samples_per_epoch
                kl_weighted = beta * kl_per_sample
                
                total_loss = recon_loss + kl_weighted
                
                val_loss += total_loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_weighted += kl_weighted.item()
        
        # Average losses
        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_kl_raw /= len(train_loader)
        train_kl_per_sample /= len(train_loader)
        train_kl_weighted /= len(train_loader)
        val_loss /= len(val_loader)
        val_recon_loss /= len(val_loader)
        val_kl_weighted /= len(val_loader)
        
        logger.info(
            f'Epoch {epoch:03d} | '
            f'train {train_loss:.4f} | '
            f'val {val_loss:.4f} | '
            f'train_recon {train_recon_loss:.4f} | '
            f'train_kl_weighted {train_kl_weighted:.4f} | '
            f'β={beta:.6f}'
        )
        
        # Save best model
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            
            best_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'beta': beta
            }, best_path)
            logger.info(f'New best model saved: val_loss={val_loss:.6f}')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f'Early stopping at epoch {epoch} (patience={patience})')
            break
        
        # Save checkpoint
        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        torch.save({
            'epoch': epoch,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'beta': beta
        }, checkpoint_path)
    
    return output_dir / 'best_model.pth'
