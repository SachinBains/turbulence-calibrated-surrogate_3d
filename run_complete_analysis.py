#!/usr/bin/env python3
"""
Comprehensive Analysis Pipeline for C3D1-C3D6 Models
Runs all evaluation, visualization, interpretability, and physics validation scripts
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
        logging.FileHandler('complete_analysis.log'),
        logging.StreamHandler()
    ]
)

def run_script(script_name, args, description=""):
    """Run a script with error handling and logging"""
    cmd = f"python scripts/{script_name} {args}"
    logging.info(f"Running: {description}")
    logging.info(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=".")
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
    parser = argparse.ArgumentParser(description='Run comprehensive analysis for all C3D models')
    parser.add_argument('--models', nargs='+', 
                       default=['C3D1_channel_baseline_128', 'C3D2_channel_mc_dropout_128', 
                               'C3D3_channel_ensemble_128', 'C3D4_channel_variational_128',
                               'C3D5_channel_swag_128', 'C3D6_channel_physics_informed_128'],
                       help='Models to analyze')
    parser.add_argument('--skip-eval', action='store_true', help='Skip basic evaluation')
    parser.add_argument('--skip-viz', action='store_true', help='Skip visualizations')
    parser.add_argument('--skip-interp', action='store_true', help='Skip interpretability')
    parser.add_argument('--skip-physics', action='store_true', help='Skip physics validation')
    parser.add_argument('--skip-uq', action='store_true', help='Skip UQ analysis')
    
    args = parser.parse_args()
    
    logging.info("="*80)
    logging.info("STARTING COMPREHENSIVE ANALYSIS PIPELINE")
    logging.info("="*80)
    
    # Phase 1: Basic Evaluation for each model
    if not args.skip_eval:
        logging.info("\n" + "="*50)
        logging.info("PHASE 1: BASIC EVALUATION")
        logging.info("="*50)
        
        for model in args.models:
            config_path = f"configs/3d/{model}.yaml"
            
            # Run evaluation
            run_script("run_eval.py", 
                      f"--config {config_path}",
                      f"Basic evaluation for {model}")
            
            # Generate predictions for ensemble/MC methods
            if "ensemble" in model:
                run_script("predict_ens.py",
                          f"--config {config_path}",
                          f"Ensemble predictions for {model}")
            elif "mc_dropout" in model:
                run_script("predict_mc.py",
                          f"--config {config_path}",
                          f"MC Dropout predictions for {model}")
            
            # Error analysis
            run_script("run_error_analysis.py",
                      f"--config {config_path}",
                      f"Error analysis for {model}")
    
    # Phase 2: Visualizations
    if not args.skip_viz:
        logging.info("\n" + "="*50)
        logging.info("PHASE 2: VISUALIZATIONS")
        logging.info("="*50)
        
        for model in args.models:
            config_path = f"configs/3d/{model}.yaml"
            
            # Slice maps
            run_script("make_slice_maps.py",
                      f"--config {config_path}",
                      f"Slice maps for {model}")
            
            # Training figures
            run_script("make_figures.py",
                      f"--config {config_path}",
                      f"Training figures for {model}")
            
            # Error and uncertainty maps
            run_script("step10_error_uncertainty_maps.py",
                      f"--config {config_path}",
                      f"Error/uncertainty maps for {model}")
    
    # Phase 3: Interpretability Analysis
    if not args.skip_interp:
        logging.info("\n" + "="*50)
        logging.info("PHASE 3: INTERPRETABILITY ANALYSIS")
        logging.info("="*50)
        
        for model in args.models:
            config_path = f"configs/3d/{model}.yaml"
            
            # Global explanations
            run_script("explain_global.py",
                      f"--config {config_path}",
                      f"Global explanations for {model}")
            
            # Local explanations
            run_script("explain_local.py",
                      f"--config {config_path}",
                      f"Local explanations for {model}")
            
            # Uncertainty explanations
            run_script("explain_uncertainty.py",
                      f"--config {config_path}",
                      f"Uncertainty explanations for {model}")
            
            # Faithfulness analysis
            run_script("faithfulness.py",
                      f"--config {config_path}",
                      f"Faithfulness analysis for {model}")
            
            # Comprehensive interpretability
            run_script("step13_interpretability_analysis.py",
                      f"--config {config_path}",
                      f"Comprehensive interpretability for {model}")
    
    # Phase 4: Physics Validation
    if not args.skip_physics:
        logging.info("\n" + "="*50)
        logging.info("PHASE 4: PHYSICS VALIDATION")
        logging.info("="*50)
        
        for model in args.models:
            config_path = f"configs/3d/{model}.yaml"
            
            # Q-criterion analysis
            run_script("run_q_criterion.py",
                      f"--config {config_path}",
                      f"Q-criterion analysis for {model}")
            
            # Physics validation
            run_script("validate_physics.py",
                      f"--config {config_path}",
                      f"Physics validation for {model}")
            
            # Multiscale physics
            run_script("run_multiscale_physics.py",
                      f"--config {config_path}",
                      f"Multiscale physics for {model}")
            
            # Temporal consistency
            run_script("run_temporal_consistency.py",
                      f"--config {config_path}",
                      f"Temporal consistency for {model}")
            
            # Comprehensive physics validation
            run_script("step12_physics_validation.py",
                      f"--config {config_path}",
                      f"Comprehensive physics validation for {model}")
    
    # Phase 5: UQ and Calibration Analysis
    if not args.skip_uq:
        logging.info("\n" + "="*50)
        logging.info("PHASE 5: UQ AND CALIBRATION ANALYSIS")
        logging.info("="*50)
        
        # Cross-model UQ comparison
        config_list = " ".join([f"configs/3d/{model}.yaml" for model in args.models])
        
        run_script("compare_uq.py",
                  f"--configs {config_list}",
                  "Cross-model UQ comparison")
        
        # Calibration analysis for each model
        for model in args.models:
            config_path = f"configs/3d/{model}.yaml"
            
            # Uncertainty calibration
            run_script("run_uncertainty_calibration.py",
                      f"--config {config_path}",
                      f"Uncertainty calibration for {model}")
            
            # Conformal prediction
            run_script("calibrate_conformal.py",
                      f"--config {config_path}",
                      f"Conformal prediction for {model}")
            
            # Calibration plots
            run_script("plot_calibration.py",
                      f"--config {config_path}",
                      f"Calibration plots for {model}")
        
        # Ensemble diversity analysis
        if any("ensemble" in model for model in args.models):
            ensemble_configs = [f"configs/3d/{model}.yaml" for model in args.models if "ensemble" in model]
            for config in ensemble_configs:
                run_script("run_ensemble_diversity.py",
                          f"--config {config}",
                          f"Ensemble diversity analysis")
    
    # Phase 6: Robustness Testing
    logging.info("\n" + "="*50)
    logging.info("PHASE 6: ROBUSTNESS TESTING")
    logging.info("="*50)
    
    for model in args.models:
        config_path = f"configs/3d/{model}.yaml"
        
        # Adversarial robustness
        run_script("run_adversarial_robustness.py",
                  f"--config {config_path}",
                  f"Adversarial robustness for {model}")
        
        # Distribution shift analysis
        run_script("run_distribution_shift.py",
                  f"--config {config_path}",
                  f"Distribution shift analysis for {model}")
    
    # Phase 7: Cross-Validation and Advanced Analysis
    logging.info("\n" + "="*50)
    logging.info("PHASE 7: ADVANCED ANALYSIS")
    logging.info("="*50)
    
    for model in args.models:
        config_path = f"configs/3d/{model}.yaml"
        
        # Cross-validation
        run_script("run_cross_validation.py",
                  f"--config {config_path}",
                  f"Cross-validation for {model}")
        
        # SINDy analysis
        run_script("run_sindy.py",
                  f"--config {config_path}",
                  f"SINDy analysis for {model}")
        
        # GPR comparison
        run_script("run_gpr.py",
                  f"--config {config_path}",
                  f"GPR comparison for {model}")
    
    # Phase 8: Aggregate Results and Reports
    logging.info("\n" + "="*50)
    logging.info("PHASE 8: AGGREGATE RESULTS AND REPORTS")
    logging.info("="*50)
    
    # Aggregate all results
    run_script("step9_aggregate_results.py",
              "--all-models",
              "Aggregate all model results")
    
    # Quantitative comparison
    run_script("step11_quantitative_comparison.py",
              "--all-models",
              "Quantitative model comparison")
    
    # Generate comprehensive report
    run_script("step14_summary_report.py",
              "--all-models",
              "Generate comprehensive summary report")
    
    # Final report generation
    run_script("generate_report.py",
              "--all-models",
              "Generate final analysis report")
    
    logging.info("\n" + "="*80)
    logging.info("COMPREHENSIVE ANALYSIS PIPELINE COMPLETED!")
    logging.info("="*80)
    logging.info("Check artifacts_3d/ for all generated results, figures, and analysis")

if __name__ == "__main__":
    main()
