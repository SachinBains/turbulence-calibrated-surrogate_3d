#!/usr/bin/env python3
"""
Multi-GPU training script for RunPod 8x A100 SXM configuration.
Optimized for maximum speed on thesis deadline.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' not in os.environ:
        # Single-node multi-GPU setup
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0)

def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

def create_model(config):
    """Create model based on config."""
    model_name = config['model']['name']
    
    if model_name == 'unet3d':
        from models.unet3d import UNet3D
        model = UNet3D(
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels'],
            base_channels=config['model']['base_channels'],
            depth=config['model']['depth']
        )
    elif model_name == 'ensemble':
        from models.ensemble import EnsembleModel
        model = EnsembleModel(config['model'])
    elif model_name == 'mc_dropout':
        from models.mc_dropout import MCDropoutModel
        model = MCDropoutModel(config['model'])
    elif model_name == 'variational':
        from models.variational import VariationalModel
        model = VariationalModel(config['model'])
    elif model_name == 'deep_ensemble':
        from models.deep_ensemble import DeepEnsembleModel
        model = DeepEnsembleModel(config['model'])
    elif model_name == 'evidential':
        from models.evidential import EvidentialModel
        model = EvidentialModel(config['model'])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def train_single_model(config_path, local_rank=0):
    """Train a single model with multi-GPU support."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup distributed training
    if torch.cuda.device_count() > 1:
        setup_distributed()
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda:0')
    
    print(f"Training on device: {device}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    try:
        # Create model
        model = create_model(config)
        model = model.to(device)
        
        # Wrap with DDP if multi-GPU
        if torch.cuda.device_count() > 1 and dist.is_initialized():
            model = DDP(model, device_ids=[local_rank])
            print(f"Model wrapped with DDP on GPU {local_rank}")
        
        # Create dataset
        from dataio.channel_dataset import ChannelDataset
        
        train_dataset = ChannelDataset(config['dataset'], split='train')
        val_dataset = ChannelDataset(config['dataset'], split='val')
        
        # Create samplers for distributed training
        if torch.cuda.device_count() > 1 and dist.is_initialized():
            train_sampler = DistributedSampler(train_dataset)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None
        
        # Create data loaders with increased batch size for multi-GPU
        batch_size = config['training']['batch_size']
        if torch.cuda.device_count() > 1:
            batch_size *= torch.cuda.device_count()  # Scale batch size
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=min(8, os.cpu_count()),
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=min(8, os.cpu_count()),
            pin_memory=True
        )
        
        # Create optimizer with scaled learning rate
        lr = config['training']['learning_rate']
        if torch.cuda.device_count() > 1:
            lr *= torch.cuda.device_count()  # Scale learning rate
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=config['training']['weight_decay']
        )
        
        # Create loss function
        criterion = nn.MSELoss()
        
        # Training loop
        model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        patience = config['training']['early_stopping_patience']
        
        for epoch in range(config['training']['epochs']):
            if train_sampler:
                train_sampler.set_epoch(epoch)
            
            # Training phase
            train_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0 and (local_rank == 0 or torch.cuda.device_count() == 1):
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            if local_rank == 0 or torch.cuda.device_count() == 1:
                print(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # Early stopping and checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save checkpoint (only on main process)
                if local_rank == 0 or torch.cuda.device_count() == 1:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'config': config
                    }
                    
                    # Create results directory
                    results_dir = Path(config['paths']['results_dir'])
                    results_dir.mkdir(parents=True, exist_ok=True)
                    
                    checkpoint_path = results_dir / 'best_model.pth'
                    torch.save(checkpoint, checkpoint_path)
                    print(f'Saved checkpoint: {checkpoint_path}')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
            
            model.train()
        
        print(f'Training completed. Best val loss: {best_val_loss:.6f}')
        
    finally:
        if torch.cuda.device_count() > 1:
            cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU training for thesis models")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    args = parser.parse_args()
    
    train_single_model(args.config, args.local_rank)

if __name__ == "__main__":
    main()
