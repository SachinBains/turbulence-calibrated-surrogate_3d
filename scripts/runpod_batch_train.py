#!/usr/bin/env python3
"""
Batch training script for RunPod - trains all 5 remaining models sequentially.
Optimized for 8x A100 SXM configuration.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def run_training_job(config_path):
    """Run a single training job."""
    print(f"\n{'='*60}")
    print(f"STARTING: {config_path}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Run multi-GPU training
    cmd = [
        "python", "-m", "torch.distributed.launch",
        "--nproc_per_node=8",  # 8 GPUs
        "--master_port=29500",
        "scripts/train_multigpu.py",
        "--config", config_path
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"COMPLETED: {config_path}")
        print(f"Duration: {duration/3600:.2f} hours")
        print(f"{'='*60}")
        
        return True, duration
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR in {config_path}: {e}")
        return False, 0

def main():
    """Run all training jobs in sequence."""
    
    # List of configs to train (in order of priority)
    configs = [
        "configs/3d_primary_final/C3D2_channel_primary_final_1000.yaml",  # MC Dropout - fastest
        "configs/3d_primary_final/C3D5_channel_primary_final_1000.yaml",  # Deep Ensemble
        "configs/3d_primary_final/C3D6_channel_primary_final_1000.yaml",  # Evidential
        "configs/3d_primary_final/C3D3_channel_primary_final_1000.yaml",  # Ensemble
        "configs/3d_primary_final/C3D4_channel_primary_final_1000.yaml",  # Variational
    ]
    
    print("🚀 RUNPOD BATCH TRAINING - 8x A100 SXM")
    print(f"Total jobs: {len(configs)}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check GPU availability
    gpu_count = os.popen('nvidia-smi -L | wc -l').read().strip()
    print(f"Available GPUs: {gpu_count}")
    
    results = []
    total_start = time.time()
    
    for i, config in enumerate(configs, 1):
        print(f"\n🔥 JOB {i}/{len(configs)}: {Path(config).stem}")
        
        if not Path(config).exists():
            print(f"❌ Config not found: {config}")
            results.append((config, False, 0))
            continue
        
        success, duration = run_training_job(config)
        results.append((config, success, duration))
        
        if not success:
            print(f"❌ Job {i} failed. Continuing with next job...")
        else:
            print(f"✅ Job {i} completed successfully!")
    
    # Final summary
    total_duration = time.time() - total_start
    
    print(f"\n{'='*80}")
    print("🎯 FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total runtime: {total_duration/3600:.2f} hours")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    successful = 0
    total_training_time = 0
    
    for config, success, duration in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        model_name = Path(config).stem
        print(f"{status} {model_name}: {duration/3600:.2f}h")
        
        if success:
            successful += 1
            total_training_time += duration
    
    print(f"\nSuccessful jobs: {successful}/{len(configs)}")
    print(f"Total training time: {total_training_time/3600:.2f} hours")
    print(f"Estimated cost: ${(total_duration/3600) * 11.12:.2f}")  # 8x A100 @ $11.12/hour
    
    if successful == len(configs):
        print("\n🎉 ALL MODELS TRAINED SUCCESSFULLY!")
        print("Ready for secondary evaluation and thesis submission!")
    else:
        print(f"\n⚠️  {len(configs) - successful} jobs failed. Check logs above.")

if __name__ == "__main__":
    main()
