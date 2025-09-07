#!/usr/bin/env python3
"""
Generate prediction files for baseline model analysis.
"""
import torch
import numpy as np
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.utils.devices import pick_device
from src.dataio.channel_dataset import ChannelDataset
from src.models.unet3d import UNet3D

def find_checkpoint(results_dir):
    """Find checkpoint with multiple naming patterns."""
    checkpoint_patterns = ['best_*.pth', 'best_model.pth', 'model_*.pth', '*.pth']
    
    for pattern in checkpoint_patterns:
        ckpts = sorted(results_dir.glob(pattern))
        if ckpts:
            # Filter for actual model checkpoints
            if pattern == '*.pth':
                ckpts = [f for f in ckpts if any(word in f.name.lower() 
                                               for word in ['best', 'model', 'checkpoint', 'final'])]
            if ckpts:
                return ckpts[-1]  # Return the most recent
    
    return None

def load_model_state(checkpoint_path):
    """Load model state dict handling different checkpoint formats."""
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract model state dict
    if 'model' in state:
        model_state = state['model']
    elif 'swa_model_state_dict' in state:
        model_state = state['swa_model_state_dict']
    elif 'model_state_dict' in state:
        model_state = state['model_state_dict']
    elif 'state_dict' in state:
        model_state = state['state_dict']
    else:
        model_state = state
    
    # Handle DataParallel wrapper (remove 'module.' prefix)
    if any(k.startswith('module.') for k in model_state.keys()):
        model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
    
    return model_state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    exp_id = cfg['experiment_id']
    results_dir = Path(cfg['paths']['results_dir']) / exp_id
    
    # Handle directory structure
    if not results_dir.exists():
        # Try alternative structure
        results_dir = results_dir / exp_id
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    print(f"Looking for checkpoint in: {results_dir}")
    
    # Load dataset
    dataset = ChannelDataset(cfg, args.split, eval_mode=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load model
    device = pick_device(args.cuda)
    mcfg = cfg['model']
    model = UNet3D(mcfg['in_channels'], mcfg['out_channels'], base_ch=mcfg['base_channels'])
    
    # Find and load checkpoint
    ckpt_path = find_checkpoint(results_dir)
    if ckpt_path is None:
        available_files = list(results_dir.glob("*"))
        raise FileNotFoundError(f'No checkpoint found in {results_dir}. Available files: {available_files}')
    
    print(f"Loading checkpoint: {ckpt_path}")
    model_state = load_model_state(ckpt_path)
    
    try:
        model.load_state_dict(model_state, strict=True)
        print("Successfully loaded state dict (strict mode)")
    except RuntimeError as e:
        print(f"Strict loading failed: {e}")
        print("Trying non-strict loading...")
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        if missing_keys:
            print(f"Warning - Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            print(f"Warning - Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        print("Successfully loaded state dict (non-strict mode)")
    
    model = model.to(device)
    model.eval()
    
    # Generate predictions
    predictions = []
    targets = []
    
    print(f"Generating predictions for {len(dataset)} samples...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if isinstance(batch, (list, tuple)):
                x, y = batch
            else:
                x = batch['input']
                y = batch['target']
            
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(dataset)} samples")
    
    # Save predictions
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Save with different naming patterns for compatibility
    pred_file = results_dir / f'pred_{args.split}.npy'
    target_file = results_dir / f'target_{args.split}.npy'
    
    # Also save with det_ prefix for conformal calibration compatibility
    det_pred_file = results_dir / f'det_mean_{args.split}.npy'
    det_target_file = results_dir / f'det_target_{args.split}.npy'
    
    np.save(pred_file, predictions)
    np.save(target_file, targets)
    np.save(det_pred_file, predictions)  # Same as pred for deterministic models
    np.save(det_target_file, targets)
    
    print(f"Saved predictions: {pred_file}")
    print(f"Saved targets: {target_file}")
    print(f"Saved deterministic predictions: {det_pred_file}")
    print(f"Saved deterministic targets: {det_target_file}")
    print(f"Shape: {predictions.shape}")

if __name__ == '__main__':
    main()
