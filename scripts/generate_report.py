#!/usr/bin/env python3
"""
Generate comprehensive HTML report for UQ experiments.
"""
import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import base64

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def encode_image_to_base64(image_path: Path) -> str:
    """Encode image to base64 for embedding in HTML."""
    if not image_path.exists():
        return ""
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    return base64.b64encode(image_data).decode('utf-8')

def load_experiment_metadata(results_dir: Path) -> Dict:
    """Load experiment configuration and metadata."""
    metadata = {
        'experiment_id': results_dir.name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results_dir': str(results_dir)
    }
    
    # Try to load config files
    config_candidates = list(results_dir.glob('*.yaml')) + list(results_dir.glob('*.json'))
    if config_candidates:
        metadata['config_file'] = str(config_candidates[0])
    
    return metadata

def load_all_results(results_dir: Path, figures_dir: Path, methods: List[str]) -> Dict:
    """Load all analysis results."""
    results = {}
    
    for method in methods:
        results[method] = {}
        
        # Load basic metrics
        for split in ['val', 'test']:
            metrics_file = results_dir / f'{method}_metrics_{split}.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    results[method][f'metrics_{split}'] = json.load(f)
        
        # Load calibration results
        for split in ['val', 'test']:
            cal_file = figures_dir / f'calibration_metrics_{method}_{split}.json'
            if cal_file.exists():
                with open(cal_file, 'r') as f:
                    results[method][f'calibration_{split}'] = json.load(f)
        
        # Load uncertainty analysis
        for split in ['val', 'test']:
            unc_file = figures_dir / f'uncertainty_analysis_{method}_{split}.json'
            if unc_file.exists():
                with open(unc_file, 'r') as f:
                    results[method][f'uncertainty_{split}'] = json.load(f)
        
        # Load physics validation
        for split in ['val', 'test']:
            physics_file = figures_dir / f'physics_validation_{method}_{split}.json'
            if physics_file.exists():
                with open(physics_file, 'r') as f:
                    results[method][f'physics_{split}'] = json.load(f)
    
    return results

def create_metrics_table(results: Dict, methods: List[str]) -> str:
    """Create HTML table of key metrics."""
    html = """
    <table class="metrics-table">
        <thead>
            <tr>
                <th>Metric</th>
                <th>Split</th>
    """
    
    for method in methods:
        html += f"<th>{method.upper()}</th>"
    
    html += """
            </tr>
        </thead>
        <tbody>
    """
    
    # Key metrics to display
    metrics_info = [
        ('rmse_vs_mu', 'RMSE', 'metrics'),
        ('nll', 'NLL', 'metrics'),
        ('coverage_1sigma', 'Coverage 1σ', 'calibration'),
        ('coverage_2sigma', 'Coverage 2σ', 'calibration'),
        ('ece', 'ECE', 'calibration'),
        ('mce', 'MCE', 'calibration'),
        ('sharpness', 'Sharpness', 'calibration')
    ]
    
    for metric_key, metric_name, source in metrics_info:
        for split in ['val', 'test']:
            html += f"""
            <tr>
                <td>{metric_name}</td>
                <td>{split}</td>
            """
            
            for method in methods:
                value = results.get(method, {}).get(f'{source}_{split}', {}).get(metric_key, 'N/A')
                if isinstance(value, (int, float)) and not np.isnan(value):
                    html += f"<td>{value:.4f}</td>"
                else:
                    html += "<td>N/A</td>"
            
            html += "</tr>"
    
    html += """
        </tbody>
    </table>
    """
    
    return html

def create_figure_gallery(figures_dir: Path, methods: List[str]) -> str:
    """Create HTML gallery of generated figures."""
    html = """
    <div class="figure-gallery">
    """
    
    # Figure categories and their files
    figure_categories = [
        ('Performance Comparison', ['performance_comparison.png']),
        ('Calibration Analysis', ['calibration_summary.png'] + 
         [f'reliability_{method}_test.png' for method in methods] +
         [f'calibration_scatter_{method}_test.png' for method in methods]),
        ('Uncertainty Analysis', ['uncertainty_correlation_summary.png'] +
         [f'uncertainty_error_scatter_{method}_test.png' for method in methods] +
         [f'uncertainty_error_bins_{method}_test.png' for method in methods]),
        ('Physics Validation', ['physics_validation_summary.png'] +
         [f'spectrum_comparison_{method}_test.png' for method in methods]),
        ('Spatial Analysis', [f'spatial_uncertainty_{method}_test.png' for method in methods])
    ]
    
    for category, figure_files in figure_categories:
        html += f"""
        <div class="figure-category">
            <h3>{category}</h3>
            <div class="figure-grid">
        """
        
        for fig_file in figure_files:
            fig_path = figures_dir / fig_file
            if fig_path.exists():
                img_b64 = encode_image_to_base64(fig_path)
                if img_b64:
                    html += f"""
                    <div class="figure-item">
                        <img src="data:image/png;base64,{img_b64}" alt="{fig_file}" />
                        <p class="figure-caption">{fig_file}</p>
                    </div>
                    """
        
        html += """
            </div>
        </div>
        """
    
    html += """
    </div>
    """
    
    return html

def create_summary_section(results: Dict, methods: List[str]) -> str:
    """Create executive summary section."""
    html = """
    <div class="summary-section">
        <h2>Executive Summary</h2>
    """
    
    # Find best performing method for key metrics
    best_methods = {}
    key_metrics = ['rmse_vs_mu', 'nll', 'ece']
    
    for metric in key_metrics:
        best_value = float('inf')
        best_method = None
        
        for method in methods:
            value = results.get(method, {}).get('metrics_test', {}).get(metric)
            if value is None:
                value = results.get(method, {}).get('calibration_test', {}).get(metric)
            
            if value is not None and not np.isnan(value) and value < best_value:
                best_value = value
                best_method = method
        
        if best_method:
            best_methods[metric] = (best_method, best_value)
    
    html += "<ul>"
    for metric, (method, value) in best_methods.items():
        html += f"<li><strong>{metric.upper()}</strong>: {method.upper()} achieves best performance ({value:.4f})</li>"
    
    # Coverage analysis
    for method in methods:
        cov_1sigma = results.get(method, {}).get('calibration_test', {}).get('coverage_1sigma')
        if cov_1sigma is not None:
            diff_from_ideal = abs(cov_1sigma - 0.683) * 100
            html += f"<li><strong>{method.upper()} Calibration</strong>: 1σ coverage is {cov_1sigma:.1%} (deviation: {diff_from_ideal:.1f}%)</li>"
    
    html += "</ul>"
    html += "</div>"
    
    return html

def generate_html_report(metadata: Dict, results: Dict, figures_dir: Path, methods: List[str]) -> str:
    """Generate complete HTML report."""
    
    # CSS styles
    css = """
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; line-height: 1.6; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
        .header h1 { margin: 0; font-size: 2.5em; }
        .header p { margin: 10px 0 0 0; opacity: 0.9; }
        .section { margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #667eea; }
        .metrics-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .metrics-table th, .metrics-table td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        .metrics-table th { background-color: #667eea; color: white; font-weight: bold; }
        .metrics-table tr:nth-child(even) { background-color: #f2f2f2; }
        .figure-gallery { margin: 20px 0; }
        .figure-category { margin: 30px 0; }
        .figure-category h3 { color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 5px; }
        .figure-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }
        .figure-item { text-align: center; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .figure-item img { max-width: 100%; height: auto; border-radius: 5px; }
        .figure-caption { margin: 10px 0 0 0; font-size: 0.9em; color: #666; font-style: italic; }
        .summary-section { background: #e8f4f8; padding: 20px; border-radius: 8px; border-left: 4px solid #17a2b8; }
        .summary-section ul { list-style-type: none; padding: 0; }
        .summary-section li { margin: 10px 0; padding: 10px; background: white; border-radius: 5px; }
        .metadata { background: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .metadata strong { color: #856404; }
    </style>
    """
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>UQ Analysis Report - {metadata['experiment_id']}</title>
        {css}
    </head>
    <body>
        <div class="header">
            <h1>Uncertainty Quantification Analysis Report</h1>
            <p>Experiment: {metadata['experiment_id']} | Generated: {metadata['timestamp']}</p>
        </div>
        
        <div class="metadata">
            <strong>Experiment Details:</strong><br>
            Results Directory: {metadata['results_dir']}<br>
            Methods Analyzed: {', '.join([m.upper() for m in methods])}<br>
            Report Generated: {metadata['timestamp']}
        </div>
        
        {create_summary_section(results, methods)}
        
        <div class="section">
            <h2>Performance Metrics</h2>
            {create_metrics_table(results, methods)}
        </div>
        
        <div class="section">
            <h2>Analysis Results</h2>
            {create_figure_gallery(figures_dir, methods)}
        </div>
        
        <div class="section">
            <h2>Detailed Results</h2>
            <p>For detailed numerical results, please refer to the JSON files in the results directory:</p>
            <ul>
                <li><strong>Basic Metrics:</strong> *_metrics_*.json</li>
                <li><strong>Calibration Analysis:</strong> calibration_metrics_*.json</li>
                <li><strong>Uncertainty Analysis:</strong> uncertainty_analysis_*.json</li>
                <li><strong>Physics Validation:</strong> physics_validation_*.json</li>
            </ul>
        </div>
        
        <footer style="margin-top: 50px; padding: 20px; text-align: center; color: #666; border-top: 1px solid #ddd;">
            <p>Generated by Turbulence UQ Analysis Suite</p>
        </footer>
    </body>
    </html>
    """
    
    return html

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive HTML report')
    parser.add_argument('--results_dir', required=True, help='Results directory')
    parser.add_argument('--methods', nargs='+', default=['mc', 'ens'], 
                       help='UQ methods to include')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory (default: figures/{exp_id})')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        exp_id = results_dir.name
        # Load config to get artifacts_root
        from src.utils.config import load_config
        config_path = results_dir.parent.parent / 'configs' / '3d' / f'{exp_id}.yaml'
        if config_path.exists():
            cfg = load_config(str(config_path))
            output_dir = Path(cfg['paths']['artifacts_root']) / 'figures' / exp_id
        else:
            output_dir = Path('figures') / exp_id
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load experiment metadata
    metadata = load_experiment_metadata(results_dir)
    
    # Load all results
    print(f"Loading results from {results_dir}")
    results = load_all_results(results_dir, output_dir, args.methods)
    
    # Generate HTML report
    print("Generating HTML report...")
    html_content = generate_html_report(metadata, results, output_dir, args.methods)
    
    # Save report
    report_file = output_dir / 'analysis_report.html'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {report_file}")
    print(f"Open in browser: file://{report_file.absolute()}")

if __name__ == '__main__':
    main()
