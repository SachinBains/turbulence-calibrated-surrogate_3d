#!/usr/bin/env python3
"""
Generate dataset splits for the actual smoke test data that the configs point to.
"""

import os
import numpy as np
from pathlib import Path
import glob
import argparse

def generate_smoke_test_splits(artifacts_root):
    """Generate splits for the actual smoke test dataset"""
    
    # The configs point to this path for smoke test data
    smoke_data_dir = Path("/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/data_3d/channel_flow_smoke")
    
    print(f"Looking for smoke test data in: {smoke_data_dir}")
    
    # Find all cube files in the smoke test directory
    cube_files = []
    
    if smoke_data_dir.exists():
        # Try different patterns
        patterns = ["*.h5", "cube_*.h5", "ret1000_cube_*.h5", "chan64_*.h5"]
        for pattern in patterns:
            files = list(smoke_data_dir.glob(pattern))
            if files:
                cube_files = sorted(files)
                print(f"Found {len(cube_files)} files with pattern: {pattern}")
                break
    
    if not cube_files:
        print(f"❌ No cube files found in {smoke_data_dir}")
        print("Available directories:")
        parent_dir = smoke_data_dir.parent
        if parent_dir.exists():
            for item in parent_dir.iterdir():
                if item.is_dir():
                    print(f"  {item}")
        return False
    
    print(f"Found {len(cube_files)} smoke test cube files")
    
    # Create simple splits for smoke test (no Y+ stratification needed for smoke test)
    n_total = len(cube_files)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = int(0.15 * n_total)
    
    # Ensure we use all files
    if n_train + n_val + n_test < n_total:
        n_test += n_total - (n_train + n_val + n_test)
    
    indices = np.arange(n_total)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:n_train+n_val+n_test]
    cal_idx = val_idx[:len(val_idx)//2]  # Use half of val for calibration
    
    # Create splits directory
    splits_dir = Path(artifacts_root) / "datasets/channel3d/splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    np.save(splits_dir / "channel_train_idx.npy", train_idx)
    np.save(splits_dir / "channel_val_idx.npy", val_idx)
    np.save(splits_dir / "channel_test_idx.npy", test_idx)
    np.save(splits_dir / "channel_cal_idx.npy", cal_idx)
    
    # Save file list for reference
    with open(splits_dir / "smoke_test_files.txt", "w") as f:
        for i, file in enumerate(cube_files):
            f.write(f"{i}: {file}\n")
    
    print(f"\n✅ Smoke test splits created:")
    print(f"  Train: {len(train_idx)} files")
    print(f"  Val: {len(val_idx)} files") 
    print(f"  Test: {len(test_idx)} files")
    print(f"  Cal: {len(cal_idx)} files")
    print(f"  Saved to: {splits_dir}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate splits for smoke test data")
    parser.add_argument("--artifacts_root", required=True, help="Path to artifacts directory")
    args = parser.parse_args()
    
    success = generate_smoke_test_splits(args.artifacts_root)
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
