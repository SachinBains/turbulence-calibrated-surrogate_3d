#!/usr/bin/env python3
"""
Streamlined evaluation pipeline for C3D1, C3D2, C3D3, C3D6 primary final models only.
Focuses on core evaluation steps with 20-minute SLURM job limits.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Selected models for streamlined evaluation
SELECTED_MODELS = ['C3D1', 'C3D2', 'C3D3', 'C3D6']
CONFIG_DIR = 'configs/3d_primary_final'

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ SUCCESS: {description}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: {description}")
        print(f"ERROR: {e}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        return False

def stage1_core_evaluation():
    """Stage 1: Core evaluation - base predictions and ensemble."""
    print("\n" + "="*80)
    print("STAGE 1: CORE EVALUATION")
    print("="*80)
    
    success_count = 0
    total_count = 0
    
    # Step 1: Base predictions for C3D1, C3D2, C3D6
    for model in ['C3D1', 'C3D2', 'C3D6']:
        config_path = f"{CONFIG_DIR}/{model}_channel_primary_final_1000.yaml"
        cmd = f"python scripts/run_eval.py --config {config_path}"
        if run_command(cmd, f"Base evaluation for {model}"):
            success_count += 1
        total_count += 1
    
    # Step 1b: Ensemble evaluation for C3D3
    config_path = f"{CONFIG_DIR}/C3D3_channel_primary_final_1000.yaml"
    cmd = f"python scripts/run_ensemble_eval.py --config {config_path}"
    if run_command(cmd, "Ensemble evaluation for C3D3"):
        success_count += 1
    total_count += 1
    
    print(f"\nSTAGE 1 SUMMARY: {success_count}/{total_count} successful")
    return success_count == total_count

def stage2_method_predictions():
    """Stage 2: Method-specific predictions."""
    print("\n" + "="*80)
    print("STAGE 2: METHOD-SPECIFIC PREDICTIONS")
    print("="*80)
    
    success_count = 0
    total_count = 0
    
    # C3D1 baseline predictions
    for split in ['val', 'test']:
        config_path = f"{CONFIG_DIR}/C3D1_channel_primary_final_1000.yaml"
        cmd = f"python scripts/generate_baseline_predictions.py --config {config_path} --split {split}"
        if run_command(cmd, f"C3D1 baseline predictions ({split})"):
            success_count += 1
        total_count += 1
    
    # C3D2 MC dropout predictions
    for split in ['val', 'test']:
        config_path = f"{CONFIG_DIR}/C3D2_channel_primary_final_1000.yaml"
        cmd = f"python scripts/predict_mc.py --config {config_path} --split {split}"
        if run_command(cmd, f"C3D2 MC dropout predictions ({split})"):
            success_count += 1
        total_count += 1
    
    # C3D3 ensemble predictions
    for split in ['val', 'test']:
        config_path = f"{CONFIG_DIR}/C3D3_channel_primary_final_1000.yaml"
        cmd = f"python scripts/predict_ens.py --config {config_path} --split {split}"
        if run_command(cmd, f"C3D3 ensemble predictions ({split})"):
            success_count += 1
        total_count += 1
    
    print(f"\nSTAGE 2 SUMMARY: {success_count}/{total_count} successful")
    return success_count == total_count

def stage3_conformal_calibration():
    """Stage 3: Conformal calibration for uncertainty methods."""
    print("\n" + "="*80)
    print("STAGE 3: CONFORMAL CALIBRATION")
    print("="*80)
    
    success_count = 0
    total_count = 0
    
    # C3D2 conformal calibration
    config_path = f"{CONFIG_DIR}/C3D2_channel_primary_final_1000.yaml"
    cmd = f"python scripts/calibrate_conformal.py --config {config_path} --mode scaled --base mc"
    if run_command(cmd, "C3D2 conformal calibration"):
        success_count += 1
    total_count += 1
    
    # C3D3 conformal calibration
    config_path = f"{CONFIG_DIR}/C3D3_channel_primary_final_1000.yaml"
    cmd = f"python scripts/calibrate_conformal.py --config {config_path} --mode scaled --base ens"
    if run_command(cmd, "C3D3 conformal calibration"):
        success_count += 1
    total_count += 1
    
    print(f"\nSTAGE 3 SUMMARY: {success_count}/{total_count} successful")
    return success_count == total_count

def stage4_uncertainty_calibration():
    """Stage 4: Uncertainty calibration analysis."""
    print("\n" + "="*80)
    print("STAGE 4: UNCERTAINTY CALIBRATION")
    print("="*80)
    
    success_count = 0
    total_count = 0
    
    for model in SELECTED_MODELS:
        config_path = f"{CONFIG_DIR}/{model}_channel_primary_final_1000.yaml"
        cmd = f"python scripts/run_uncertainty_calibration.py --config {config_path}"
        if run_command(cmd, f"Uncertainty calibration for {model}"):
            success_count += 1
        total_count += 1
    
    print(f"\nSTAGE 4 SUMMARY: {success_count}/{total_count} successful")
    return success_count == total_count

def stage5_physics_validation():
    """Stage 5: Physics validation."""
    print("\n" + "="*80)
    print("STAGE 5: PHYSICS VALIDATION")
    print("="*80)
    
    success_count = 0
    total_count = 0
    
    for model in SELECTED_MODELS:
        config_path = f"{CONFIG_DIR}/{model}_channel_primary_final_1000.yaml"
        cmd = f"python scripts/validate_physics.py --config {config_path}"
        if run_command(cmd, f"Physics validation for {model}"):
            success_count += 1
        total_count += 1
    
    print(f"\nSTAGE 5 SUMMARY: {success_count}/{total_count} successful")
    return success_count == total_count

def stage6_visualization():
    """Stage 6: Core visualization and reporting."""
    print("\n" + "="*80)
    print("STAGE 6: VISUALIZATION AND REPORTING")
    print("="*80)
    
    success_count = 0
    total_count = 0
    
    # Generate figures for each model
    for model in SELECTED_MODELS:
        config_path = f"{CONFIG_DIR}/{model}_channel_primary_final_1000.yaml"
        
        # Make figures
        cmd = f"python scripts/make_figures.py --config {config_path}"
        if run_command(cmd, f"Generate figures for {model}"):
            success_count += 1
        total_count += 1
        
        # Generate report
        cmd = f"python scripts/generate_report.py --config {config_path}"
        if run_command(cmd, f"Generate report for {model}"):
            success_count += 1
        total_count += 1
    
    print(f"\nSTAGE 6 SUMMARY: {success_count}/{total_count} successful")
    return success_count == total_count

def stage7_comparison():
    """Stage 7: Global comparison and aggregation."""
    print("\n" + "="*80)
    print("STAGE 7: GLOBAL COMPARISON")
    print("="*80)
    
    success_count = 0
    total_count = 0
    
    # Compare UQ methods
    cmd = f"python scripts/compare_uq.py --results_dir $ARTIFACTS_ROOT/results"
    if run_command(cmd, "Compare UQ methods"):
        success_count += 1
    total_count += 1
    
    # Aggregate results
    cmd = f"python scripts/step9_aggregate_results.py --results_dir $ARTIFACTS_ROOT/results"
    if run_command(cmd, "Aggregate results"):
        success_count += 1
    total_count += 1
    
    print(f"\nSTAGE 7 SUMMARY: {success_count}/{total_count} successful")
    return success_count == total_count

def main():
    parser = argparse.ArgumentParser(description='Streamlined evaluation pipeline')
    parser.add_argument('--stage', type=int, choices=range(1, 8), 
                       help='Run specific stage (1-7), or all stages if not specified')
    parser.add_argument('--models', nargs='+', choices=SELECTED_MODELS, 
                       default=SELECTED_MODELS, help='Models to evaluate')
    
    args = parser.parse_args()
    
    print("STREAMLINED EVALUATION PIPELINE")
    print(f"Selected models: {args.models}")
    print(f"Stage: {'All' if args.stage is None else args.stage}")
    
    # Update global models list if specified
    global SELECTED_MODELS
    SELECTED_MODELS = args.models
    
    stages = {
        1: stage1_core_evaluation,
        2: stage2_method_predictions,
        3: stage3_conformal_calibration,
        4: stage4_uncertainty_calibration,
        5: stage5_physics_validation,
        6: stage6_visualization,
        7: stage7_comparison
    }
    
    if args.stage:
        # Run specific stage
        success = stages[args.stage]()
        print(f"\n{'='*80}")
        print(f"STAGE {args.stage} {'COMPLETED SUCCESSFULLY' if success else 'FAILED'}")
        print(f"{'='*80}")
    else:
        # Run all stages
        overall_success = True
        for stage_num, stage_func in stages.items():
            success = stage_func()
            if not success:
                overall_success = False
                print(f"\n⚠️  Stage {stage_num} failed, but continuing...")
        
        print(f"\n{'='*80}")
        print(f"PIPELINE {'COMPLETED' if overall_success else 'COMPLETED WITH ERRORS'}")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()
