#!/usr/bin/env python3
"""
Train deep ensemble for uncertainty quantification.
"""
import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.devices import pick_device
from src.utils.seeding import seed_all
from src.utils.logging import get_logger
from src.utils.manifest import append_manifest_row
from src.dataio.channel_dataset import ChannelDataset
from src.uq.ensembles import train_ensemble

def main():
    parser = argparse.ArgumentParser(description='Train deep ensemble')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--members', type=int, default=None, help='Number of ensemble members')
    parser.add_argument('--seeds', nargs='+', type=int, default=None, help='Seeds for ensemble members')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    device = pick_device(args.cuda)
    logger = get_logger()
    
    # Determine ensemble size and seeds
    n_members = args.members or cfg.get('uq', {}).get('ensemble_members', 5)
    if args.seeds:
        seeds = args.seeds[:n_members]  # Use provided seeds
    else:
        # Generate default seeds
        seeds = [111 + i * 111 for i in range(n_members)]
    
    logger.info(f"Training ensemble with {n_members} members")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Config ensemble_members value: {cfg.get('uq', {}).get('ensemble_members', 'NOT_FOUND')}")
    
    # Setup paths
    exp_id = cfg['experiment_id']
    results_dir = Path(cfg['paths']['results_dir']) / exp_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    train_dataset = ChannelDataset(cfg, 'train')
    val_dataset = ChannelDataset(cfg, 'val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['train']['batch_size'], 
        shuffle=True, 
        num_workers=cfg['train']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=cfg['train']['num_workers']
    )
    
    # Train ensemble
    member_paths = train_ensemble(cfg, seeds, device, train_loader, val_loader, results_dir)
    
    # Save ensemble metadata
    ensemble_info = {
        'n_members': len(member_paths),
        'seeds': seeds,
        'member_paths': [str(p) for p in member_paths],
        'config': args.config
    }
    
    info_path = results_dir / 'ensemble_info.json'
    with open(info_path, 'w') as f:
        json.dump(ensemble_info, f, indent=2)
    
    logger.info(f"Ensemble training complete. Info saved to: {info_path}")
    logger.info(f"Trained {len(member_paths)}/{n_members} members successfully")
    
    # Update manifest
    append_manifest_row(args.config, seeds[0], str(results_dir))

if __name__ == '__main__':
    main()
