#!/usr/bin/env python3
"""
FIXED Comprehensive Analysis Pipeline for C3D1-C3D6 Models
All issues identified and corrected
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis_fixed.log'),
        logging.StreamHandler()
    ]
)

def run_script(script_name, args, description=""):
    """Run a script with error handling and logging"""
    # Set artifacts directory and activate virtual environment
    artifacts_dir = "/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d"
    cmd = f"source ~/venvs/turbml/bin/activate && export PYTHONPATH=$PWD:$PYTHONPATH && export ARTIFACTS_DIR={artifacts_dir} && python scripts/{script_name} {args}"
    logging.info(f"Running: {description}")
    logging.info(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=".", executable='/bin/bash')
        if result.returncode == 0:
            logging.info(f"✓ SUCCESS: {description}")
            if result.stdout:
                logging.info(f"Output: {result.stdout}")
        else:
            logging.error(f"✗ FAILED: {description}")
            logging.error(f"Error: {result.stderr}")
            return False
    except Exception as e:
        logging.error(f"✗ EXCEPTION in {description}: {str(e)}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Run FIXED comprehensive analysis for C3D models')
    parser.add_argument('--models', nargs='+', 
                       default=['C3D1_channel_baseline_128', 'C3D2_channel_mc_dropout_128'],
                       help='Models to analyze')
    
    args = parser.parse_args()
    
    logging.info("="*80)
    logging.info("STARTING FIXED COMPREHENSIVE ANALYSIS PIPELINE")
    logging.info("="*80)
    
    # Phase 1: Basic Evaluation (WORKING MODELS ONLY)
    logging.info("\n" + "="*50)
    logging.info("PHASE 1: BASIC EVALUATION")
    logging.info("="*50)
    
    for model in args.models:
        config_path = f"configs/3d/{model}.yaml"
        
        # Run evaluation (works for C3D1, C3D2)
        run_script("run_eval.py", 
                  f"--config {config_path}",
                  f"Basic evaluation for {model}")
        
        # MC Dropout predictions (only for C3D2)
        if "mc_dropout" in model:
            run_script("predict_mc.py",
                      f"--config {config_path}",
                      f"MC Dropout predictions for {model}")
    
    # Phase 2: Visualizations (FIXED PARAMETERS)
    logging.info("\n" + "="*50)
    logging.info("PHASE 2: VISUALIZATIONS")
    logging.info("="*50)
    
    for model in args.models:
        config_path = f"configs/3d/{model}.yaml"
        artifacts_dir = "/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d"
        results_dir = f"{artifacts_dir}/results/{model}"
        
        # Slice maps (no --results_dir parameter)
        run_script("make_slice_maps.py",
                  f"--config {config_path}",
                  f"Slice maps for {model}")
        
        # Training figures (only --results_dir parameter)
        run_script("make_figures.py",
                  f"--results_dir {results_dir}",
                  f"Training figures for {model}")
    
    # Phase 3: Interpretability (FIXED PARAMETERS)
    logging.info("\n" + "="*50)
    logging.info("PHASE 3: INTERPRETABILITY ANALYSIS")
    logging.info("="*50)
    
    for model in args.models:
        config_path = f"configs/3d/{model}.yaml"
        
        # Global explanations (add required --target parameter)
        run_script("explain_global.py",
                  f"--config {config_path} --target error",
                  f"Global explanations for {model}")
        
        # Local explanations (add required --method parameter)
        run_script("explain_local.py",
                  f"--config {config_path} --method ig",
                  f"Local explanations for {model}")
    
    # Phase 4: Error Analysis (FIXED PATHS)
    logging.info("\n" + "="*50)
    logging.info("PHASE 4: ERROR ANALYSIS")
    logging.info("="*50)
    
    for model in args.models:
        config_path = f"configs/3d/{model}.yaml"
        artifacts_dir = "/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d"
        results_dir = f"{artifacts_dir}/results/{model}"
        
        # Error analysis (only --results_dir parameter)
        run_script("run_error_analysis.py",
                  f"--config {config_path} --results_dir {results_dir}",
                  f"Error analysis for {model}")
    
    logging.info("\n" + "="*80)
    logging.info("FIXED ANALYSIS PIPELINE COMPLETED!")
    logging.info("="*80)
    logging.info("Check artifacts_3d/ for generated results")

if __name__ == "__main__":
    main()
