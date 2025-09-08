#!/usr/bin/env python3
"""
Comprehensive fix for all HPC deployment issues:
1. Fix nested directory structures 
2. Check dataset splits and regenerate if needed
3. Verify checkpoint locations
4. Only run analysis on models that actually exist
"""

import os
import sys
import shutil
import glob
import numpy as np
from pathlib import Path
import argparse

def fix_nested_directories(artifacts_root):
    """Fix all nested directory structures"""
    results_dir = Path(artifacts_root) / "results"
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    print("=== Fixing Nested Directory Structures ===")
    
    # Find all experiment directories
    experiment_dirs = [d for d in results_dir.glob("C3D*") if d.is_dir()]
    
    for exp_dir in experiment_dirs:
        print(f"Processing {exp_dir.name}...")
        
        # Check for nested structure: C3D*/C3D*/
        nested_dir = exp_dir / exp_dir.name
        
        if nested_dir.exists():
            print(f"  Found nested structure: {nested_dir}")
            
            # Move all contents up one level
            for item in nested_dir.iterdir():
                target_item = exp_dir / item.name
                if target_item.exists():
                    if target_item.is_dir():
                        shutil.rmtree(target_item)
                    else:
                        target_item.unlink()
                print(f"  Moving {item} -> {target_item}")
                shutil.move(str(item), str(target_item))
            
            # Remove the now-empty nested directory
            if nested_dir.exists():
                print(f"  Removing nested directory: {nested_dir}")
                shutil.rmtree(nested_dir)
        else:
            print(f"  No nested structure found for {exp_dir.name}")

def check_dataset_files(artifacts_root):
    """Check if dataset files exist and splits are valid"""
    print("\n=== Checking Dataset Files ===")
    
    # Check raw data directory
    data_dir = Path(artifacts_root) / "datasets/channel3d/raw"
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    # Find cube files
    cube_files = list(data_dir.glob("*.h5"))
    if not cube_files:
        # Check batch directories
        batch_dirs = list(data_dir.glob("Re1000_Batch*_Data*"))
        for batch_dir in batch_dirs:
            cube_files.extend(list(batch_dir.glob("*.h5")))
    
    print(f"Found {len(cube_files)} cube files")
    
    if len(cube_files) == 0:
        print("❌ No cube files found!")
        return False
    
    # Check splits
    splits_dir = Path(artifacts_root) / "datasets/channel3d/splits"
    split_files = ["channel_train_idx.npy", "channel_val_idx.npy", "channel_test_idx.npy", "channel_cal_idx.npy"]
    
    splits_valid = True
    for split_file in split_files:
        split_path = splits_dir / split_file
        if split_path.exists():
            indices = np.load(split_path)
            if len(indices) > 0 and max(indices) < len(cube_files):
                print(f"✅ {split_file}: {len(indices)} samples (valid)")
            else:
                print(f"❌ {split_file}: indices out of range (max: {max(indices)}, files: {len(cube_files)})")
                splits_valid = False
        else:
            print(f"❌ {split_file}: missing")
            splits_valid = False
    
    return splits_valid

def check_model_checkpoints(artifacts_root):
    """Check which models have valid checkpoints"""
    print("\n=== Checking Model Checkpoints ===")
    
    results_dir = Path(artifacts_root) / "results"
    models = ["C3D1_channel_baseline_128", "C3D2_channel_mc_dropout_128", "C3D3_channel_ensemble_128", 
              "C3D4_channel_variational_128", "C3D5_channel_swag_128", "C3D6_channel_physics_informed_128"]
    
    valid_models = []
    
    for model in models:
        model_dir = results_dir / model
        if not model_dir.exists():
            print(f"❌ {model}: directory missing")
            continue
        
        # Check for checkpoints
        checkpoints = list(model_dir.glob("best_*.pth"))
        if not checkpoints:
            checkpoints = list(model_dir.glob("*.pth"))
        
        if checkpoints:
            print(f"✅ {model}: {len(checkpoints)} checkpoints found")
            valid_models.append(model)
        else:
            print(f"❌ {model}: no checkpoints found")
    
    return valid_models

def create_minimal_analysis_script(artifacts_root, valid_models):
    """Create analysis script that only runs on valid models"""
    print(f"\n=== Creating Analysis Script for {len(valid_models)} Valid Models ===")
    
    script_content = f"""#!/bin/bash
# MINIMAL ANALYSIS PIPELINE - ONLY VALID MODELS
# Friend's HPC (n63719vm) - Only models with checkpoints

export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/turbulence-calibrated-surrogate_3d:$PYTHONPATH
export ARTIFACTS_ROOT=/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/artifacts_3d

echo "Starting MINIMAL analysis pipeline - {len(valid_models)} valid models..."

# Fix config paths first
echo "=== PHASE 0: Fix Config Paths ==="
python cranfield_analysis/update_paths_for_friend.py

# PHASE 1: Generate Base Predictions (Only valid models)
echo "=== PHASE 1: Base Predictions (GPU) ==="
"""

    for model in valid_models:
        if "ensemble" in model:
            script_content += f"python scripts/run_ensemble_eval.py --config configs/3d/{model}.yaml --cuda\n"
        else:
            script_content += f"python scripts/run_eval.py --config configs/3d/{model}.yaml --cuda\n"
    
    script_content += """
echo "✅ Minimal analysis complete!"
"""
    
    script_path = Path(artifacts_root).parent / "turbulence-calibrated-surrogate_3d/cranfield_analysis/run_minimal_analysis.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"Created: {script_path}")
    return script_path

def main():
    parser = argparse.ArgumentParser(description="Fix all HPC deployment issues")
    parser.add_argument("--artifacts_root", required=True, help="Path to artifacts directory")
    args = parser.parse_args()
    
    artifacts_root = args.artifacts_root
    
    print("🔧 COMPREHENSIVE HPC FIX STARTING...")
    
    # 1. Fix nested directories
    fix_nested_directories(artifacts_root)
    
    # 2. Check dataset
    dataset_valid = check_dataset_files(artifacts_root)
    
    # 3. Check model checkpoints
    valid_models = check_model_checkpoints(artifacts_root)
    
    # 4. Create minimal analysis script
    if valid_models:
        script_path = create_minimal_analysis_script(artifacts_root, valid_models)
        print(f"\n🎉 READY TO RUN: {script_path}")
    else:
        print("\n❌ NO VALID MODELS FOUND - Cannot create analysis script")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Dataset valid: {'✅' if dataset_valid else '❌'}")
    print(f"  Valid models: {len(valid_models)}/6")
    print(f"  Models: {', '.join(valid_models) if valid_models else 'None'}")

if __name__ == "__main__":
    main()
