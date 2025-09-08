#!/usr/bin/env python3
"""
Step 9: Aggregate and analyze all experiment results
Parses metrics, logs, and checkpoints from all experiments (E1-E6)
"""

import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob

def load_json_safe(filepath):
    """Safely load JSON file, return empty dict if fails"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return {}

def parse_experiment_results(results_dir):
    """Parse all experiment results from step9_metrics folder"""
    results_path = Path(results_dir)
    
    # Initialize aggregated results
    summary_data = []
    
    # Find all experiment directories
    exp_dirs = [d for d in results_path.glob("E*") if d.is_dir()]
    
    for exp_dir in sorted(exp_dirs):
        exp_name = exp_dir.name
        print(f"Processing {exp_name}...")
        
        # Initialize experiment record
        exp_record = {
            'experiment': exp_name,
            'method': get_method_type(exp_name),
            'domain': get_domain_type(exp_name)
        }
        
        # Load basic metrics (baseline experiments)
        test_metrics_file = exp_dir / "test_metrics.json"
        val_metrics_file = exp_dir / "val_metrics.json"
        
        test_metrics = load_json_safe(test_metrics_file)
        val_metrics = load_json_safe(val_metrics_file)
        
        if test_metrics:
            exp_record.update({
                'test_rmse': test_metrics.get('rmse'),
                'test_mae': test_metrics.get('mae'),
                'test_n_samples': test_metrics.get('n')
            })
        
        if val_metrics:
            exp_record.update({
                'val_rmse': val_metrics.get('rmse'),
                'val_mae': val_metrics.get('mae'),
                'val_n_samples': val_metrics.get('n')
            })
        
        # Load MC Dropout metrics
        mc_test_file = exp_dir / "mc_metrics_test.json"
        mc_val_file = exp_dir / "mc_metrics_val.json"
        
        mc_test = load_json_safe(mc_test_file)
        mc_val = load_json_safe(mc_val_file)
        
        if mc_test:
            exp_record.update({
                'mc_test_rmse': mc_test.get('rmse_vs_mu'),
                'mc_test_nll': mc_test.get('nll'),
                'mc_test_avg_sigma': mc_test.get('avg_sigma'),
                'mc_test_cov80': mc_test.get('cov80'),
                'mc_test_cov90': mc_test.get('cov90'),
                'mc_test_cov95': mc_test.get('cov95'),
                'mc_conformal_coverage': mc_test.get('conformal_coverage'),
                'mc_conformal_width': mc_test.get('conformal_width')
            })
        
        if mc_val:
            exp_record.update({
                'mc_val_rmse': mc_val.get('rmse_vs_mu'),
                'mc_val_nll': mc_val.get('nll'),
                'mc_val_avg_sigma': mc_val.get('avg_sigma'),
                'mc_val_cov80': mc_val.get('cov80'),
                'mc_val_cov90': mc_val.get('cov90'),
                'mc_val_cov95': mc_val.get('cov95')
            })
        
        # Load Ensemble metrics
        ens_test_file = exp_dir / "ens_metrics_test.json"
        ens_val_file = exp_dir / "ens_metrics_val.json"
        
        ens_test = load_json_safe(ens_test_file)
        ens_val = load_json_safe(ens_val_file)
        
        if ens_test:
            overall = ens_test.get('overall', {})
            exp_record.update({
                'ens_test_rmse': overall.get('rmse'),
                'ens_test_mae': overall.get('mae'),
                'ens_test_n_samples': overall.get('n_samples'),
                'ens_conformal_coverage': overall.get('conformal_coverage'),
                'ens_conformal_width': overall.get('conformal_width')
            })
        
        if ens_val:
            overall = ens_val.get('overall', {})
            exp_record.update({
                'ens_val_rmse': overall.get('rmse'),
                'ens_val_mae': overall.get('mae'),
                'ens_val_n_samples': overall.get('n_samples')
            })
        
        # Load run info (training details)
        run_info_file = exp_dir / "run_info.json"
        run_info = load_json_safe(run_info_file)
        
        if run_info:
            exp_record.update({
                'best_epoch': run_info.get('best_epoch'),
                'best_val_loss': run_info.get('best_val_loss'),
                'total_epochs': run_info.get('total_epochs'),
                'early_stopped': run_info.get('early_stopped')
            })
        
        # Load ensemble info (if applicable)
        ens_info_file = exp_dir / "ensemble_info.json"
        ens_info = load_json_safe(ens_info_file)
        
        if ens_info:
            exp_record.update({
                'n_ensemble_members': ens_info.get('n_members'),
                'ensemble_seeds': ens_info.get('seeds')
            })
        
        summary_data.append(exp_record)
    
    return pd.DataFrame(summary_data)

def get_method_type(exp_name):
    """Determine UQ method from experiment name"""
    if 'baseline' in exp_name.lower():
        return 'Baseline'
    elif 'bayes' in exp_name.lower() or 'dropout' in exp_name.lower():
        return 'MC Dropout'
    elif 'ens' in exp_name.lower():
        return 'Ensemble'
    else:
        return 'Unknown'

def get_domain_type(exp_name):
    """Determine domain scenario from experiment name"""
    if 'ab' in exp_name.lower():
        return 'A->B (Domain Shift)'
    else:
        return 'ID (In-Domain)'

def analyze_training_logs(logs_dir):
    """Parse SLURM logs for training insights"""
    logs_path = Path(logs_dir) / "slurm"
    log_files = list(logs_path.glob("*.out"))
    
    log_summary = []
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Extract key info from logs
            lines = content.split('\n')
            
            # Find final metrics
            final_metrics = {}
            for line in reversed(lines):
                if "TEST RMSE:" in line:
                    parts = line.split()
                    final_metrics['final_test_rmse'] = float(parts[2].rstrip(','))
                    final_metrics['final_test_mae'] = float(parts[4])
                    break
                elif "VAL RMSE:" in line:
                    parts = line.split()
                    final_metrics['final_val_rmse'] = float(parts[2].rstrip(','))
                    final_metrics['final_val_mae'] = float(parts[4])
            
            log_summary.append({
                'log_file': log_file.name,
                'experiment': log_file.name.split('-')[0],
                **final_metrics
            })
            
        except Exception as e:
            print(f"Warning: Could not parse {log_file}: {e}")
    
    return pd.DataFrame(log_summary)

def main():
    """Main aggregation function for Step 9"""
    print("=== Step 9: Aggregating All Experiment Results ===\n")
    
    # Define paths
    step9_dir = Path("C:/Users/Sachi/OneDrive/Desktop/Dissertation/step9_metrics")
    results_dir = step9_dir / "results"
    logs_dir = step9_dir / "logs"
    
    # Parse experiment results
    print("1. Parsing experiment metrics...")
    df_results = parse_experiment_results(results_dir)
    
    # Parse training logs
    print("2. Parsing training logs...")
    df_logs = analyze_training_logs(logs_dir)
    
    # Save aggregated results
    output_dir = Path("C:/Users/Sachi/OneDrive/Desktop/Dissertation/turbulence-calibrated-surrogate_full/step9_analysis")
    output_dir.mkdir(exist_ok=True)
    
    df_results.to_csv(output_dir / "aggregated_metrics.csv", index=False)
    df_logs.to_csv(output_dir / "training_logs_summary.csv", index=False)
    
    print(f"3. Results saved to {output_dir}/")
    
    # Display summary
    print("\n=== EXPERIMENT SUMMARY ===")
    print(df_results[['experiment', 'method', 'domain', 'test_rmse', 'test_mae', 'best_epoch']].to_string(index=False))
    
    print("\n=== UQ METHODS COMPARISON ===")
    uq_cols = ['experiment', 'method', 'mc_test_rmse', 'mc_test_cov80', 'mc_test_cov90', 'ens_test_rmse', 'ens_conformal_coverage']
    uq_data = df_results[uq_cols].dropna(how='all', subset=uq_cols[2:])
    if not uq_data.empty:
        print(uq_data.to_string(index=False))
    
    print("\n=== DOMAIN SHIFT ANALYSIS ===")
    domain_comparison = df_results.groupby(['method', 'domain'])['test_rmse'].mean().unstack(fill_value=np.nan)
    print(domain_comparison)
    
    print(f"\nStep 9 Complete: All results aggregated and saved to {output_dir}/")
    print("Next: Step 10 - Generate visualization and error maps")
    
    return df_results, df_logs

if __name__ == "__main__":
    df_results, df_logs = main()
