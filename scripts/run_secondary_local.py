#!/usr/bin/env python3
"""
Run secondary evaluation locally on your machine.
"""

import os
import sys
from pathlib import Path

# Set environment variables for local paths
os.environ['PYTHONPATH'] = str(Path(__file__).parent.parent)
os.environ['ARTIFACTS_ROOT'] = r'c:\Users\Sachi\OneDrive\Desktop\Dissertation\artifacts_3d'

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def main():
    """Run secondary evaluation for available models."""
    
    artifacts_root = Path(os.environ['ARTIFACTS_ROOT'])
    
    # Check which models have checkpoints
    models_to_run = []
    
    # C3D1 - Baseline
    c3d1_checkpoint = artifacts_root / "results/C3D1_channel_baseline_128"
    if any(c3d1_checkpoint.glob("*.pth")):
        models_to_run.append("C3D1")
        print("✓ C3D1: Checkpoint found")
    else:
        print("✗ C3D1: No checkpoint found")
    
    # C3D2 - MC Dropout
    c3d2_checkpoint = artifacts_root / "results/C3D2_channel_mc_dropout_128"
    if any(c3d2_checkpoint.glob("*.pth")):
        models_to_run.append("C3D2")
        print("✓ C3D2: Checkpoint found")
    else:
        print("✗ C3D2: No checkpoint found")
    
    # C3D3 - Ensemble
    c3d3_members = artifacts_root / "results/C3D3_channel_ensemble_128/members"
    if c3d3_members.exists() and any(c3d3_members.glob("*/best_*.pth")):
        models_to_run.append("C3D3")
        print("✓ C3D3: Ensemble members found")
    else:
        print("✗ C3D3: No ensemble members found")
    
    # C3D6 - Physics Informed
    c3d6_checkpoint = artifacts_root / "results/C3D6_channel_physics_informed_128"
    if any(c3d6_checkpoint.glob("*.pth")):
        models_to_run.append("C3D6")
        print("✓ C3D6: Checkpoint found")
    else:
        print("✗ C3D6: No checkpoint found")
    
    print(f"\nRunning secondary evaluation on {len(models_to_run)} models: {models_to_run}")
    
    # Import and run secondary evaluation
    from scripts.run_secondary_evaluation import main as run_secondary
    
    for model in models_to_run:
        config_path = f"configs/3d_secondary/{model}_secondary_5200.yaml"
        print(f"\n--- Running {model} secondary evaluation ---")
        
        # Set up arguments for secondary evaluation
        import argparse
        args = argparse.Namespace()
        args.config = config_path
        
        try:
            # Run secondary evaluation
            sys.argv = ['run_secondary_evaluation.py', '--config', config_path]
            run_secondary()
            print(f"✓ {model} secondary evaluation completed")
        except Exception as e:
            print(f"✗ {model} secondary evaluation failed: {e}")
            continue
    
    print("\n=== SECONDARY EVALUATION COMPLETE ===")
    
    # Count output files
    total_files = 0
    for model in models_to_run:
        results_dir = artifacts_root / f"results/{model}_secondary_5200"
        if results_dir.exists():
            count = len(list(results_dir.glob("*")))
            print(f"{model}_secondary_5200: {count} files")
            total_files += count
    
    print(f"Total secondary evaluation files: {total_files}")

if __name__ == "__main__":
    main()
