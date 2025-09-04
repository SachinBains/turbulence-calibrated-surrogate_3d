import torch
from src.train.trainer import train_loop as base_train_loop
from src.utils.logging import get_logger

def variational_train_loop(cfg, net, criterion, optimizer, scaler, train_loader, val_loader, 
                          save_dir, logger, resume_path=None, device='cuda'):
    """
    Training loop for variational models that includes KL divergence in the loss.
    """
    
    def variational_loss_fn(pred, target):
        """Compute loss including KL divergence for variational models."""
        # Standard reconstruction loss
        recon_loss = criterion(pred, target)
        
        # KL divergence loss (already weighted by kl_weight in the model)
        kl_loss = net.kl_divergence()
        
        # Total loss
        total_loss = recon_loss + kl_loss
        
        # Log components for monitoring
        if hasattr(variational_loss_fn, 'step_count'):
            variational_loss_fn.step_count += 1
        else:
            variational_loss_fn.step_count = 1
            
        if variational_loss_fn.step_count % 100 == 0:
            logger.info(f"Step {variational_loss_fn.step_count}: recon_loss={recon_loss:.6f}, kl_loss={kl_loss:.6f}")
        
        return total_loss
    
    # Use the base training loop but with our custom loss function
    return base_train_loop(cfg, net, variational_loss_fn, optimizer, scaler, 
                          train_loader, val_loader, save_dir, logger, 
                          resume_path, device)
