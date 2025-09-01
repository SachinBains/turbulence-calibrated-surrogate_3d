#!/usr/bin/env python3
"""
Step 11: Compare UQ methods quantitatively (tables, plots, coverage)
Combines results from Step 9 aggregation and Step 10 visualization data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def load_aggregated_results():
    """Load results from Step 9 aggregation"""
    step9_dir = Path("step9_analysis")
    
    # Load aggregated metrics
    metrics_df = pd.read_csv(step9_dir / "aggregated_metrics.csv")
    
    return metrics_df

def create_performance_comparison_table(df):
    """Create comprehensive performance comparison table"""
    
    # Select key metrics for comparison
    comparison_cols = [
        'experiment', 'method', 'domain',
        'test_rmse', 'test_mae', 
        'mc_test_rmse', 'mc_test_nll', 'mc_test_cov80', 'mc_test_cov90',
        'ens_test_rmse', 'ens_conformal_coverage'
    ]
    
    # Create clean comparison table
    comp_df = df[comparison_cols].copy()
    
    # Round numerical values
    numeric_cols = comp_df.select_dtypes(include=[np.number]).columns
    comp_df[numeric_cols] = comp_df[numeric_cols].round(4)
    
    return comp_df

def create_uncertainty_quality_metrics(df):
    """Calculate uncertainty quality metrics"""
    
    uq_metrics = []
    
    for _, row in df.iterrows():
        record = {
            'experiment': row['experiment'],
            'method': row['method'], 
            'domain': row['domain']
        }
        
        # Coverage metrics (how well calibrated)
        if not pd.isna(row['mc_test_cov80']):
            record['coverage_80_error'] = abs(row['mc_test_cov80'] - 0.80)
            record['coverage_90_error'] = abs(row['mc_test_cov90'] - 0.90)
            record['avg_coverage_error'] = (record['coverage_80_error'] + record['coverage_90_error']) / 2
        
        # Conformal coverage
        if not pd.isna(row['mc_conformal_coverage']):
            record['conformal_coverage'] = row['mc_conformal_coverage']
        if not pd.isna(row['ens_conformal_coverage']):
            record['conformal_coverage'] = row['ens_conformal_coverage']
        
        # Uncertainty magnitude
        if not pd.isna(row['mc_test_avg_sigma']):
            record['avg_uncertainty'] = row['mc_test_avg_sigma']
        
        uq_metrics.append(record)
    
    return pd.DataFrame(uq_metrics)

def create_method_comparison_plots(df, output_dir):
    """Create comprehensive comparison plots"""
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. RMSE Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # RMSE comparison across methods
    ax1 = axes[0, 0]
    methods_data = []
    
    for _, row in df.iterrows():
        if not pd.isna(row['test_rmse']):
            methods_data.append({'Method': 'Baseline', 'Domain': row['domain'], 'RMSE': row['test_rmse']})
        if not pd.isna(row['mc_test_rmse']):
            methods_data.append({'Method': 'MC Dropout', 'Domain': row['domain'], 'RMSE': row['mc_test_rmse']})
        if not pd.isna(row['ens_test_rmse']):
            methods_data.append({'Method': 'Ensemble', 'Domain': row['domain'], 'RMSE': row['ens_test_rmse']})
    
    methods_df = pd.DataFrame(methods_data)
    if not methods_df.empty:
        sns.barplot(data=methods_df, x='Method', y='RMSE', hue='Domain', ax=ax1)
        ax1.set_title('RMSE Comparison Across Methods')
        ax1.grid(True, alpha=0.3)
    
    # 2. Coverage Comparison
    ax2 = axes[0, 1]
    coverage_data = []
    
    for _, row in df.iterrows():
        if not pd.isna(row['mc_test_cov80']):
            coverage_data.append({'Method': 'MC Dropout', 'Domain': row['domain'], 
                                'Coverage_80': row['mc_test_cov80'], 'Coverage_90': row['mc_test_cov90']})
    
    if coverage_data:
        cov_df = pd.DataFrame(coverage_data)
        x_pos = np.arange(len(cov_df))
        width = 0.35
        
        ax2.bar(x_pos - width/2, cov_df['Coverage_80'], width, label='80% Coverage', alpha=0.8)
        ax2.bar(x_pos + width/2, cov_df['Coverage_90'], width, label='90% Coverage', alpha=0.8)
        ax2.axhline(y=0.80, color='red', linestyle='--', alpha=0.7, label='Nominal 80%')
        ax2.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='Nominal 90%')
        
        ax2.set_xlabel('Experiment')
        ax2.set_ylabel('Coverage')
        ax2.set_title('Coverage Calibration')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"{row['Method']}\n{row['Domain']}" for _, row in cov_df.iterrows()])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Conformal Coverage Comparison
    ax3 = axes[1, 0]
    conformal_data = []
    
    for _, row in df.iterrows():
        if not pd.isna(row['mc_conformal_coverage']):
            conformal_data.append({'Method': 'MC Dropout', 'Domain': row['domain'], 
                                 'Coverage': row['mc_conformal_coverage']})
        if not pd.isna(row['ens_conformal_coverage']):
            conformal_data.append({'Method': 'Ensemble', 'Domain': row['domain'], 
                                 'Coverage': row['ens_conformal_coverage']})
    
    if conformal_data:
        conf_df = pd.DataFrame(conformal_data)
        sns.barplot(data=conf_df, x='Method', y='Coverage', hue='Domain', ax=ax3)
        ax3.axhline(y=0.90, color='red', linestyle='--', alpha=0.7, label='Target 90%')
        ax3.set_title('Conformal Prediction Coverage')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Domain Shift Impact
    ax4 = axes[1, 1]
    domain_comparison = []
    
    # Compare ID vs A->B for each method
    baseline_id = df[df['experiment'] == 'E1_hit_baseline']['test_rmse'].iloc[0] if len(df[df['experiment'] == 'E1_hit_baseline']) > 0 else np.nan
    baseline_ab = df[df['experiment'] == 'E3_hit_ab_baseline']['test_rmse'].iloc[0] if len(df[df['experiment'] == 'E3_hit_ab_baseline']) > 0 else np.nan
    
    mc_id = df[df['experiment'] == 'E2_hit_bayes']['mc_test_rmse'].iloc[0] if len(df[df['experiment'] == 'E2_hit_bayes']) > 0 else np.nan
    mc_ab = df[df['experiment'] == 'E4_hit_ab_dropout']['mc_test_rmse'].iloc[0] if len(df[df['experiment'] == 'E4_hit_ab_dropout']) > 0 else np.nan
    
    ens_id = df[df['experiment'] == 'E5_hit_ens']['ens_test_rmse'].iloc[0] if len(df[df['experiment'] == 'E5_hit_ens']) > 0 else np.nan
    ens_ab = df[df['experiment'] == 'E6_hit_ab_ens']['ens_test_rmse'].iloc[0] if len(df[df['experiment'] == 'E6_hit_ab_ens']) > 0 else np.nan
    
    methods = ['Baseline', 'MC Dropout', 'Ensemble']
    id_values = [baseline_id, mc_id, ens_id]
    ab_values = [baseline_ab, mc_ab, ens_ab]
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    # Filter out NaN values
    valid_methods = []
    valid_id = []
    valid_ab = []
    
    for i, (method, id_val, ab_val) in enumerate(zip(methods, id_values, ab_values)):
        if not (pd.isna(id_val) and pd.isna(ab_val)):
            valid_methods.append(method)
            valid_id.append(id_val if not pd.isna(id_val) else 0)
            valid_ab.append(ab_val if not pd.isna(ab_val) else 0)
    
    if valid_methods:
        x_pos = np.arange(len(valid_methods))
        ax4.bar(x_pos - width/2, valid_id, width, label='ID', alpha=0.8)
        ax4.bar(x_pos + width/2, valid_ab, width, label='A->B', alpha=0.8)
        
        ax4.set_xlabel('Method')
        ax4.set_ylabel('Test RMSE')
        ax4.set_title('Domain Shift Impact')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(valid_methods)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Quantitative UQ Method Comparison')
    plt.tight_layout()
    plt.savefig(output_dir / 'quantitative_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_latex_summary_table(df, uq_df, output_dir):
    """Create LaTeX table for dissertation"""
    
    latex_content = """
\\begin{table}[h]
\\centering
\\caption{Quantitative Comparison of UQ Methods}
\\label{tab:uq_comparison}
\\begin{tabular}{llcccc}
\\toprule
Method & Domain & RMSE & Coverage 80\\% & Coverage 90\\% & Conf. Coverage \\\\
\\midrule
"""
    
    for _, row in df.iterrows():
        method = row['method']
        domain = row['domain']
        
        # Get RMSE (prefer UQ method over baseline)
        rmse = row['mc_test_rmse'] if not pd.isna(row['mc_test_rmse']) else row['ens_test_rmse']
        if pd.isna(rmse):
            rmse = row['test_rmse']
        
        rmse_str = f"{rmse:.3f}" if not pd.isna(rmse) else "--"
        
        # Coverage metrics
        cov80_str = f"{row['mc_test_cov80']:.3f}" if not pd.isna(row['mc_test_cov80']) else "--"
        cov90_str = f"{row['mc_test_cov90']:.3f}" if not pd.isna(row['mc_test_cov90']) else "--"
        
        # Conformal coverage
        conf_cov = row['mc_conformal_coverage'] if not pd.isna(row['mc_conformal_coverage']) else row['ens_conformal_coverage']
        conf_str = f"{conf_cov:.3f}" if not pd.isna(conf_cov) else "--"
        
        latex_content += f"{method} & {domain} & {rmse_str} & {cov80_str} & {cov90_str} & {conf_str} \\\\\n"
    
    latex_content += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open(output_dir / "uq_comparison_table.tex", 'w') as f:
        f.write(latex_content)
    
    return latex_content

def main():
    """Main function for Step 11"""
    print("=== Step 11: Quantitative UQ Method Comparison ===\n")
    
    # Setup output directory
    output_dir = Path("step11_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Load aggregated results from Step 9
    print("1. Loading aggregated results...")
    df = load_aggregated_results()
    
    # Create performance comparison table
    print("2. Creating performance comparison table...")
    comp_table = create_performance_comparison_table(df)
    comp_table.to_csv(output_dir / "performance_comparison_table.csv", index=False)
    
    # Calculate uncertainty quality metrics
    print("3. Calculating uncertainty quality metrics...")
    uq_df = create_uncertainty_quality_metrics(df)
    uq_df.to_csv(output_dir / "uncertainty_quality_metrics.csv", index=False)
    
    # Create comparison plots
    print("4. Creating quantitative comparison plots...")
    create_method_comparison_plots(df, output_dir)
    
    # Create LaTeX table
    print("5. Generating LaTeX summary table...")
    latex_table = create_latex_summary_table(df, uq_df, output_dir)
    
    # Print summary
    print("\n=== QUANTITATIVE COMPARISON SUMMARY ===")
    print("\n** Performance (RMSE) **")
    perf_summary = df[['experiment', 'method', 'domain', 'test_rmse', 'mc_test_rmse', 'ens_test_rmse']].fillna('--')
    print(perf_summary.to_string(index=False))
    
    print("\n** Uncertainty Calibration **")
    cal_summary = df[['experiment', 'method', 'mc_test_cov80', 'mc_test_cov90', 'mc_conformal_coverage', 'ens_conformal_coverage']].fillna('--')
    print(cal_summary.to_string(index=False))
    
    print(f"\nStep 11 Complete: Quantitative analysis saved to {output_dir}/")
    print("Generated files:")
    print("  - performance_comparison_table.csv")
    print("  - uncertainty_quality_metrics.csv") 
    print("  - quantitative_comparison.png")
    print("  - uq_comparison_table.tex")
    print("\nNext: Step 12 - Physics/consistency checks")

if __name__ == "__main__":
    main()
