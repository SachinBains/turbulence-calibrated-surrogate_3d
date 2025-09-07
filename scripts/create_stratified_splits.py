#!/usr/bin/env python3
"""
Create stratified train/val/test/calibration splits for JHTDB channel flow dataset.
Ensures all Y+ bands are represented proportionally in each split.
"""

import numpy as np
import json
from pathlib import Path
import re
import argparse
from collections import defaultdict

def extract_yplus_from_filename(filename):
    """Extract Y+ value from JHTDB filename structure."""
    # Assuming filename format: chan96_yplus_XXX_*.h5
    # Adjust this regex based on your actual filename format
    match = re.search(r'yplus_(\d+\.?\d*)', filename)
    if match:
        return float(match.group(1))
    
    # Alternative patterns - adjust based on your actual filenames
    match = re.search(r'y(\d+\.?\d*)', filename)
    if match:
        return float(match.group(1))
    
    # If no Y+ found, try to infer from position in sorted list
    # This is a fallback - ideally Y+ should be in filename
    return None

def assign_yplus_band(yplus_value, bands):
    """Assign Y+ value to appropriate band."""
    for i, (min_y, max_y) in enumerate(bands):
        if min_y <= yplus_value < max_y:
            return i
    # Handle edge case for maximum value
    if yplus_value == bands[-1][1]:
        return len(bands) - 1
    return None

def create_stratified_splits(data_dir, output_dir, seed=42):
    """Create stratified splits for JHTDB channel flow data."""
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all cube files
    cube_files = sorted(list(data_path.glob('chan96_*.h5')))
    if not cube_files:
        # Try alternative pattern
        cube_files = sorted(list(data_path.glob('cube_*.h5')))
    
    if not cube_files:
        raise ValueError(f"No cube files found in {data_dir}")
    
    print(f"Found {len(cube_files)} cube files")
    
    # Y+ bands from thesis (4 bands for 1200 cubes = 300 per band)
    yplus_bands = [
        (0, 30),      # B1: Viscous sublayer and buffer
        (30, 100),    # B2: Lower log region
        (100, 370),   # B3: Upper log region  
        (370, 1000)   # B4: Inner-outer interface
    ]
    
    # Group files by Y+ band
    band_files = defaultdict(list)
    unassigned_files = []
    
    for i, filepath in enumerate(cube_files):
        filename = filepath.name
        yplus = extract_yplus_from_filename(filename)
        
        if yplus is None:
            # Fallback: assume files are ordered by Y+ and distribute evenly
            band_idx = i // (len(cube_files) // len(yplus_bands))
            band_idx = min(band_idx, len(yplus_bands) - 1)
            band_files[band_idx].append(i)
            print(f"Warning: No Y+ found in {filename}, assigned to band {band_idx} by position")
        else:
            band_idx = assign_yplus_band(yplus, yplus_bands)
            if band_idx is not None:
                band_files[band_idx].append(i)
            else:
                unassigned_files.append(i)
                print(f"Warning: Y+ {yplus} in {filename} doesn't fit any band")
    
    # Handle unassigned files by distributing them evenly
    for i, file_idx in enumerate(unassigned_files):
        band_idx = i % len(yplus_bands)
        band_files[band_idx].append(file_idx)
    
    # Print band statistics
    print("\nY+ Band Distribution:")
    for band_idx, (min_y, max_y) in enumerate(yplus_bands):
        count = len(band_files[band_idx])
        print(f"  Band {band_idx+1} [{min_y}-{max_y}): {count} files")
    
    # Split ratios: 60% train, 20% val, 10% test, 10% calibration
    split_ratios = {
        'train': 0.60,
        'val': 0.20, 
        'test': 0.10,
        'cal': 0.10
    }
    
    # Initialize split arrays
    train_idx = []
    val_idx = []
    test_idx = []
    cal_idx = []
    
    np.random.seed(seed)
    
    # Stratified splitting within each band
    for band_idx in range(len(yplus_bands)):
        band_indices = np.array(band_files[band_idx])
        n_band = len(band_indices)
        
        if n_band == 0:
            continue
            
        # Shuffle indices within band
        shuffled = np.random.permutation(band_indices)
        
        # Calculate split points
        n_train = int(n_band * split_ratios['train'])
        n_val = int(n_band * split_ratios['val'])
        n_test = int(n_band * split_ratios['test'])
        # Remaining goes to calibration
        
        # Split the band
        train_idx.extend(shuffled[:n_train])
        val_idx.extend(shuffled[n_train:n_train+n_val])
        test_idx.extend(shuffled[n_train+n_val:n_train+n_val+n_test])
        cal_idx.extend(shuffled[n_train+n_val+n_test:])
        
        print(f"  Band {band_idx+1}: {n_train} train, {n_val} val, {n_test} test, {len(shuffled[n_train+n_val+n_test:])} cal")
    
    # Convert to numpy arrays and sort
    train_idx = np.sort(np.array(train_idx))
    val_idx = np.sort(np.array(val_idx))
    test_idx = np.sort(np.array(test_idx))
    cal_idx = np.sort(np.array(cal_idx))
    
    # Verify no overlap
    all_indices = np.concatenate([train_idx, val_idx, test_idx, cal_idx])
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("Overlapping indices detected in splits!")
    
    # Verify all files are assigned
    if len(all_indices) != len(cube_files):
        raise ValueError(f"Not all files assigned: {len(all_indices)} != {len(cube_files)}")
    
    # Save splits
    np.save(output_path / 'channel_train_idx.npy', train_idx)
    np.save(output_path / 'channel_val_idx.npy', val_idx)
    np.save(output_path / 'channel_test_idx.npy', test_idx)
    np.save(output_path / 'channel_cal_idx.npy', cal_idx)
    
    # Create metadata
    meta = {
        "dataset_name": "channel_flow_3d_primary",
        "data_dir": str(data_dir),
        "total_cubes": len(cube_files),
        "cube_shape": [96, 96, 96],
        "variables": ["u", "v", "w"],
        "yplus_range": [0, 1000],
        "yplus_bands": yplus_bands,
        "splits": {
            "train": {
                "ratio": split_ratios['train'],
                "count": len(train_idx),
                "indices": f"stratified across {len(yplus_bands)} Y+ bands"
            },
            "val": {
                "ratio": split_ratios['val'], 
                "count": len(val_idx),
                "indices": f"stratified across {len(yplus_bands)} Y+ bands"
            },
            "test": {
                "ratio": split_ratios['test'],
                "count": len(test_idx),
                "indices": f"stratified across {len(yplus_bands)} Y+ bands"
            },
            "calibration": {
                "ratio": split_ratios['cal'],
                "count": len(cal_idx),
                "indices": f"stratified across {len(yplus_bands)} Y+ bands"
            }
        },
        "stratification": "Y+ bands ensure all wall layers represented",
        "seed": seed,
        "created_by": "create_stratified_splits.py"
    }
    
    with open(output_path / 'channel_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n✅ Created stratified splits:")
    print(f"  Train: {len(train_idx)} files ({len(train_idx)/len(cube_files)*100:.1f}%)")
    print(f"  Val: {len(val_idx)} files ({len(val_idx)/len(cube_files)*100:.1f}%)")
    print(f"  Test: {len(test_idx)} files ({len(test_idx)/len(cube_files)*100:.1f}%)")
    print(f"  Calibration: {len(cal_idx)} files ({len(cal_idx)/len(cube_files)*100:.1f}%)")
    print(f"  Saved to: {output_path}")
    print(f"  No overlap confirmed ✓")
    print(f"  Y+ stratification across {len(yplus_bands)} bands ✓")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create stratified splits for JHTDB channel flow data")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing chan96_*.h5 files")
    parser.add_argument("--output_dir", type=str, default="splits",
                       help="Output directory for split files")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible splits")
    
    args = parser.parse_args()
    
    create_stratified_splits(args.data_dir, args.output_dir, args.seed)
