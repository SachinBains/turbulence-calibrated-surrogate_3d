#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.dataio.hit_dataset import HITDataset
from src.utils.config import load_config
from src.models.unet3d import UNet3D
from src.uq.mc_dropout import mc_predict
from torch.utils.data import DataLoader

def test_ab_split_functionality():
    print("Testing A->B split functionality...")
    
    # Test E3 config (baseline)
    cfg = load_config('configs/E3_hit_ab_baseline.yaml')
    ds = HITDataset(cfg, 'test')
    print(f"E3 (baseline) test split has {len(ds)} samples")
    
    # Test E4 config (dropout)
    cfg = load_config('configs/E4_hit_ab_dropout.yaml')
    ds = HITDataset(cfg, 'test')
    print(f"E4 (dropout) test split has {len(ds)} samples")
    
    print("A->B split functionality verified successfully!")

def test_model_forward_pass():
    print("\nTesting model forward pass with A->B split...")
    
    # Load E4 config (with dropout)
    cfg = load_config('configs/E4_hit_ab_dropout.yaml')
    ds = HITDataset(cfg, 'test')
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    
    # Create model
    model = UNet3D(3, 1, 32, 0.2)
    model.enable_mc_dropout(0.2)
    model.eval()
    
    # Test forward pass
    with torch.no_grad():
        for i, (X, y) in enumerate(loader):
            if i >= 1:  # Only test first batch
                break
            print(f"Input shape: {X.shape}")
            output = model(X)
            print(f"Output shape: {output.shape}")
            print("Forward pass successful!")
            
    print("Model forward pass verified successfully!")

if __name__ == '__main__':
    test_ab_split_functionality()
    test_model_forward_pass()
