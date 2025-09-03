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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    exp_id = cfg['experiment_id']
    results_dir = Path(cfg['paths']['results_dir']) / exp_id
    
    # Load dataset
    dataset = ChannelDataset(cfg, args.split, eval_mode=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load model
    device = pick_device(args.cuda)
    mcfg = cfg['model']
    model = UNet3D(mcfg['in_channels'], mcfg['out_channels'], base_ch=mcfg['base_channels'])
    
    ckpts = sorted(results_dir.glob('best_*.pth'))
    assert ckpts, f'No checkpoint found in {results_dir}'
    state = torch.load(ckpts[-1], map_location=device)
    model.load_state_dict(state['model'])
    model = model.to(device)
    model.eval()
    
    # Generate predictions
    predictions = []
    targets = []
    
    print(f"Generating predictions for {len(dataset)} samples...")
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(dataset)} samples")
    
    # Save predictions
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    pred_file = results_dir / f'pred_{args.split}.npy'
    target_file = results_dir / f'target_{args.split}.npy'
    
    np.save(pred_file, predictions)
    np.save(target_file, targets)
    
    print(f"Saved predictions: {pred_file}")
    print(f"Saved targets: {target_file}")
    print(f"Shape: {predictions.shape}")

if __name__ == '__main__':
    main()
