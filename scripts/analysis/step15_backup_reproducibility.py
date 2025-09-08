#!/usr/bin/env python3
"""
Step 15: Code/Data/Results Backup and Reproducibility Check
Ensures all code, data, and results are properly backed up and the analysis 
pipeline is reproducible by creating checksums, dependency lists, and validation scripts.
"""

import os
import sys
import json
import hashlib
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd

def calculate_file_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file"""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception as e:
        return f"ERROR: {str(e)}"

def create_file_manifest(base_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Create manifest of all important files with checksums"""
    
    print("1. Creating file manifest with checksums...")
    
    manifest_data = []
    
    # Important directories to include
    important_dirs = [
        'scripts/',
        'src/',
        'configs/',
        'step9_analysis/',
        'step10_analysis/', 
        'step11_analysis/',
        'step12_analysis/',
        'step13_analysis/',
        'step14_summary/',
        'splits/',
        'requirements.txt',
        'README.md'
    ]
    
    for dir_pattern in important_dirs:
        if dir_pattern.endswith('/'):
            # Directory - scan all files
            dir_path = base_dir / dir_pattern.rstrip('/')
            if dir_path.exists():
                for file_path in dir_path.rglob('*'):
                    if file_path.is_file() and not file_path.name.startswith('.'):
                        rel_path = file_path.relative_to(base_dir)
                        checksum = calculate_file_checksum(file_path)
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        
                        manifest_data.append({
                            'file_path': str(rel_path),
                            'size_mb': f"{size_mb:.3f}",
                            'checksum': checksum,
                            'category': dir_pattern.rstrip('/'),
                            'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                        })
        else:
            # Single file
            file_path = base_dir / dir_pattern
            if file_path.exists():
                checksum = calculate_file_checksum(file_path)
                size_mb = file_path.stat().st_size / (1024 * 1024)
                
                manifest_data.append({
                    'file_path': dir_pattern,
                    'size_mb': f"{size_mb:.3f}",
                    'checksum': checksum,
                    'category': 'root',
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
    
    manifest_df = pd.DataFrame(manifest_data)
    manifest_df.to_csv(output_dir / 'file_manifest.csv', index=False)
    
    print(f"  Created manifest for {len(manifest_df)} files")
    return manifest_df

def check_dependencies(base_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Check and document all dependencies"""
    
    print("2. Checking dependencies and environment...")
    
    deps_info = {
        'python_version': sys.version,
        'platform': sys.platform,
        'timestamp': datetime.now().isoformat()
    }
    
    # Check if requirements.txt exists
    req_file = base_dir / 'requirements.txt'
    if req_file.exists():
        with open(req_file, 'r') as f:
            deps_info['requirements_txt'] = f.read().strip().split('\n')
    else:
        deps_info['requirements_txt'] = []
    
    # Try to get installed package versions
    try:
        import pkg_resources
        installed_packages = [str(d) for d in pkg_resources.working_set]
        deps_info['installed_packages'] = sorted(installed_packages)
    except:
        deps_info['installed_packages'] = []
    
    # Check critical packages
    critical_packages = ['numpy', 'pandas', 'matplotlib', 'scipy', 'torch', 'seaborn']
    package_versions = {}
    
    for package in critical_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            package_versions[package] = version
        except ImportError:
            package_versions[package] = 'not_installed'
    
    deps_info['critical_packages'] = package_versions
    
    # Save dependency information
    with open(output_dir / 'dependencies.json', 'w') as f:
        json.dump(deps_info, f, indent=2)
    
    return deps_info

def validate_analysis_pipeline(base_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Validate that the analysis pipeline can be reproduced"""
    
    print("3. Validating analysis pipeline reproducibility...")
    
    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'pipeline_steps': {},
        'missing_files': [],
        'script_validation': {}
    }
    
    # Check each pipeline step
    pipeline_steps = {
        'step9': {
            'script': 'scripts/step9_aggregate_results.py',
            'output_dir': 'step9_analysis/',
            'key_outputs': ['aggregated_metrics.csv', 'training_logs_summary.csv']
        },
        'step10': {
            'script': 'scripts/step10_error_uncertainty_maps.py',
            'output_dir': 'step10_analysis/',
            'key_outputs': ['method_statistics_summary.csv']
        },
        'step11': {
            'script': 'scripts/step11_quantitative_comparison.py',
            'output_dir': 'step11_analysis/',
            'key_outputs': ['performance_comparison_table.csv', 'uncertainty_quality_metrics.csv']
        },
        'step12': {
            'script': 'scripts/step12_physics_validation.py',
            'output_dir': 'step12_analysis/',
            'key_outputs': ['physics_properties_summary.csv', 'detailed_physics_results.json']
        },
        'step13': {
            'script': 'scripts/step13_interpretability_analysis.py',
            'output_dir': 'step13_analysis/',
            'key_outputs': ['interpretability_summary.csv', 'detailed_interpretability_results.json']
        },
        'step14': {
            'script': 'scripts/step14_summary_report.py',
            'output_dir': 'step14_summary/',
            'key_outputs': ['turbulence_analysis_report.md', 'summary_table.tex']
        }
    }
    
    for step_name, step_info in pipeline_steps.items():
        step_validation = {
            'script_exists': False,
            'output_dir_exists': False,
            'outputs_exist': [],
            'missing_outputs': []
        }
        
        # Check script exists
        script_path = base_dir / step_info['script']
        step_validation['script_exists'] = script_path.exists()
        
        # Check output directory exists
        output_path = base_dir / step_info['output_dir']
        step_validation['output_dir_exists'] = output_path.exists()
        
        # Check key outputs exist
        if output_path.exists():
            for output_file in step_info['key_outputs']:
                file_path = output_path / output_file
                if file_path.exists():
                    step_validation['outputs_exist'].append(output_file)
                else:
                    step_validation['missing_outputs'].append(output_file)
                    validation_results['missing_files'].append(str(file_path))
        
        validation_results['pipeline_steps'][step_name] = step_validation
    
    # Validate script syntax
    for step_name, step_info in pipeline_steps.items():
        script_path = base_dir / step_info['script']
        if script_path.exists():
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    script_content = f.read()
                
                # Try to compile the script
                compile(script_content, str(script_path), 'exec')
                validation_results['script_validation'][step_name] = 'valid'
            except SyntaxError as e:
                validation_results['script_validation'][step_name] = f'syntax_error: {str(e)}'
            except Exception as e:
                validation_results['script_validation'][step_name] = f'error: {str(e)}'
    
    # Save validation results
    with open(output_dir / 'pipeline_validation.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    return validation_results

def create_reproducibility_guide(base_dir: Path, output_dir: Path, deps_info: Dict[str, Any]):
    """Create step-by-step reproducibility guide"""
    
    print("4. Creating reproducibility guide...")
    
    guide = []
    guide.append("# Turbulence Surrogate Analysis - Reproducibility Guide")
    guide.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    guide.append("")
    
    guide.append("## Environment Setup")
    guide.append("")
    guide.append("### Python Environment")
    guide.append(f"- **Python Version**: {deps_info['python_version'].split()[0]}")
    guide.append(f"- **Platform**: {deps_info['platform']}")
    guide.append("")
    
    guide.append("### Required Packages")
    guide.append("Install dependencies using:")
    guide.append("```bash")
    guide.append("pip install -r requirements.txt")
    guide.append("```")
    guide.append("")
    
    if deps_info['critical_packages']:
        guide.append("**Critical Package Versions:**")
        for pkg, version in deps_info['critical_packages'].items():
            guide.append(f"- {pkg}: {version}")
        guide.append("")
    
    guide.append("## Data Requirements")
    guide.append("")
    guide.append("### CSF3 Data Access")
    guide.append("1. Access to CSF3 HPC system with trained model artifacts")
    guide.append("2. Download prediction arrays from CSF3:")
    guide.append("   ```bash")
    guide.append("   # On CSF3:")
    guide.append("   cd /path/to/artifacts")
    guide.append("   tar -czf predictions.tar.gz results/*/mc_*_test.npy results/*/ens_*_test.npy")
    guide.append("   scp predictions.tar.gz local_machine:/path/to/step10_visualization/")
    guide.append("   ```")
    guide.append("")
    
    guide.append("### Local Data Structure")
    guide.append("Ensure the following directory structure:")
    guide.append("```")
    guide.append("turbulence-calibrated-surrogate_full/")
    guide.append("├── scripts/           # Analysis scripts")
    guide.append("├── src/              # Source code")
    guide.append("├── configs/          # Experiment configurations")
    guide.append("├── step9_analysis/   # Aggregated metrics")
    guide.append("├── step10_analysis/  # Visualization results")
    guide.append("├── step11_analysis/  # Quantitative comparison")
    guide.append("├── step12_analysis/  # Physics validation")
    guide.append("├── step13_analysis/  # Interpretability analysis")
    guide.append("└── step14_summary/   # Final reports")
    guide.append("```")
    guide.append("")
    
    guide.append("## Execution Pipeline")
    guide.append("")
    guide.append("Run the analysis pipeline in order:")
    guide.append("")
    
    steps = [
        ("Step 9", "python scripts/step9_aggregate_results.py", "Aggregate experiment metrics and logs"),
        ("Step 10", "python scripts/step10_error_uncertainty_maps.py", "Generate error and uncertainty visualizations"),
        ("Step 11", "python scripts/step11_quantitative_comparison.py", "Quantitative UQ method comparison"),
        ("Step 12", "python scripts/step12_physics_validation.py", "Physics consistency validation"),
        ("Step 13", "python scripts/step13_interpretability_analysis.py", "Interpretability and feature analysis"),
        ("Step 14", "python scripts/step14_summary_report.py", "Generate comprehensive summary report")
    ]
    
    for i, (step_name, command, description) in enumerate(steps, 1):
        guide.append(f"### {step_name}: {description}")
        guide.append("```bash")
        guide.append(command)
        guide.append("```")
        guide.append("")
    
    guide.append("## Expected Outputs")
    guide.append("")
    guide.append("After successful execution, you should have:")
    guide.append("- **CSV files**: Aggregated metrics and summary tables")
    guide.append("- **PNG files**: Visualization plots and comparison figures")
    guide.append("- **JSON files**: Detailed analysis results")
    guide.append("- **TEX files**: LaTeX tables for publication")
    guide.append("- **MD/HTML files**: Comprehensive analysis reports")
    guide.append("")
    
    guide.append("## Validation")
    guide.append("")
    guide.append("To validate successful reproduction:")
    guide.append("1. Check that all output directories contain expected files")
    guide.append("2. Verify file checksums match the provided manifest")
    guide.append("3. Review generated plots for consistency")
    guide.append("4. Compare summary metrics with reference values")
    guide.append("")
    
    guide.append("## Troubleshooting")
    guide.append("")
    guide.append("### Common Issues")
    guide.append("- **Missing prediction files**: Ensure CSF3 data is properly downloaded")
    guide.append("- **Unicode errors**: Use UTF-8 encoding for all text files")
    guide.append("- **Memory issues**: Process data in batches if needed")
    guide.append("- **Package conflicts**: Use virtual environment with exact versions")
    guide.append("")
    
    guide.append("### Contact Information")
    guide.append("For questions about reproduction, refer to:")
    guide.append("- Original experiment configurations in `configs/`")
    guide.append("- Detailed logs in CSF3 artifacts")
    guide.append("- Pipeline validation results in `step15_backup/`")
    guide.append("")
    
    # Save reproducibility guide
    with open(output_dir / 'reproducibility_guide.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(guide))
    
    return guide

def create_backup_archive(base_dir: Path, output_dir: Path) -> str:
    """Create compressed backup archive of all important files"""
    
    print("5. Creating backup archive...")
    
    # Create temporary directory for backup staging
    backup_staging = output_dir / 'backup_staging'
    backup_staging.mkdir(exist_ok=True)
    
    # Copy important directories and files
    important_items = [
        'scripts/',
        'src/',
        'configs/',
        'step9_analysis/',
        'step10_analysis/',
        'step11_analysis/',
        'step12_analysis/',
        'step13_analysis/',
        'step14_summary/',
        'splits/',
        'requirements.txt',
        'README.md'
    ]
    
    for item in important_items:
        src_path = base_dir / item
        dst_path = backup_staging / item
        
        if src_path.exists():
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
    
    # Create archive
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_name = f'turbulence_analysis_backup_{timestamp}'
    
    try:
        # Try to create tar.gz archive
        shutil.make_archive(
            str(output_dir / archive_name),
            'gztar',
            str(backup_staging)
        )
        archive_path = f"{archive_name}.tar.gz"
        print(f"  Created backup archive: {archive_path}")
    except:
        # Fallback to zip
        shutil.make_archive(
            str(output_dir / archive_name),
            'zip',
            str(backup_staging)
        )
        archive_path = f"{archive_name}.zip"
        print(f"  Created backup archive: {archive_path}")
    
    # Clean up staging directory
    shutil.rmtree(backup_staging)
    
    return archive_path

def create_validation_script(base_dir: Path, output_dir: Path):
    """Create script to validate reproduction"""
    
    print("6. Creating validation script...")
    
    validation_script = '''#!/usr/bin/env python3
"""
Validation script to check pipeline reproduction
"""

import sys
from pathlib import Path
import pandas as pd
import json

def validate_reproduction():
    """Validate that all pipeline steps completed successfully"""
    
    base_dir = Path.cwd()
    
    # Expected outputs for each step
    expected_outputs = {
        'step9_analysis': ['aggregated_metrics.csv', 'training_logs_summary.csv'],
        'step10_analysis': ['method_statistics_summary.csv'],
        'step11_analysis': ['performance_comparison_table.csv', 'uncertainty_quality_metrics.csv'],
        'step12_analysis': ['physics_properties_summary.csv', 'detailed_physics_results.json'],
        'step13_analysis': ['interpretability_summary.csv', 'detailed_interpretability_results.json'],
        'step14_summary': ['turbulence_analysis_report.md', 'summary_table.tex']
    }
    
    print("=== Pipeline Validation ===\\n")
    
    all_valid = True
    
    for step_dir, expected_files in expected_outputs.items():
        step_path = base_dir / step_dir
        print(f"Checking {step_dir}...")
        
        if not step_path.exists():
            print(f"  ERROR: Directory {step_dir} does not exist")
            all_valid = False
            continue
        
        missing_files = []
        for expected_file in expected_files:
            file_path = step_path / expected_file
            if file_path.exists():
                print(f"  ✓ {expected_file}")
            else:
                print(f"  ✗ {expected_file} (missing)")
                missing_files.append(expected_file)
                all_valid = False
        
        if missing_files:
            print(f"  Missing files: {missing_files}")
        print()
    
    if all_valid:
        print("🎉 All pipeline steps validated successfully!")
        print("\\nReproduction appears to be complete.")
    else:
        print("⚠️  Some validation checks failed.")
        print("\\nPlease check missing files and re-run failed steps.")
    
    return all_valid

if __name__ == "__main__":
    validate_reproduction()
'''
    
    with open(output_dir / 'validate_reproduction.py', 'w', encoding='utf-8') as f:
        f.write(validation_script)

def generate_backup_summary(
    manifest_df: pd.DataFrame,
    deps_info: Dict[str, Any],
    validation_results: Dict[str, Any],
    archive_path: str,
    output_dir: Path
):
    """Generate summary of backup and reproducibility status"""
    
    print("7. Generating backup summary...")
    
    summary = []
    summary.append("# Backup and Reproducibility Summary")
    summary.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("")
    
    # File statistics
    total_files = len(manifest_df)
    total_size_mb = sum(float(size) for size in manifest_df['size_mb'])
    
    summary.append("## Backup Statistics")
    summary.append(f"- **Total Files**: {total_files}")
    summary.append(f"- **Total Size**: {total_size_mb:.1f} MB")
    summary.append(f"- **Archive**: {archive_path}")
    summary.append("")
    
    # File breakdown by category
    summary.append("### File Breakdown by Category")
    category_stats = manifest_df.groupby('category').agg({
        'file_path': 'count',
        'size_mb': lambda x: sum(float(s) for s in x)
    }).round(2)
    
    for category, stats in category_stats.iterrows():
        summary.append(f"- **{category}**: {stats['file_path']} files, {stats['size_mb']:.1f} MB")
    summary.append("")
    
    # Pipeline validation status
    summary.append("## Pipeline Validation Status")
    
    total_steps = len(validation_results['pipeline_steps'])
    valid_steps = sum(1 for step_info in validation_results['pipeline_steps'].values() 
                     if step_info['script_exists'] and step_info['output_dir_exists'])
    
    summary.append(f"- **Pipeline Steps**: {valid_steps}/{total_steps} validated")
    summary.append(f"- **Missing Files**: {len(validation_results['missing_files'])}")
    summary.append("")
    
    # Critical package versions
    summary.append("## Environment Information")
    summary.append(f"- **Python**: {deps_info['python_version'].split()[0]}")
    summary.append("- **Platform**: {deps_info['platform']}")
    summary.append("")
    
    summary.append("### Critical Packages")
    for pkg, version in deps_info['critical_packages'].items():
        status = "✓" if version != 'not_installed' else "✗"
        summary.append(f"- {status} **{pkg}**: {version}")
    summary.append("")
    
    # Recommendations
    summary.append("## Recommendations")
    
    if validation_results['missing_files']:
        summary.append("### Action Required")
        summary.append("- Some output files are missing - re-run corresponding pipeline steps")
        summary.append("- Check CSF3 data availability for prediction arrays")
    else:
        summary.append("### Status: Ready for Publication")
        summary.append("- All pipeline steps completed successfully")
        summary.append("- Backup archive created and validated")
        summary.append("- Environment documented for reproduction")
    
    summary.append("")
    summary.append("## Next Steps")
    summary.append("1. **Archive Storage**: Store backup archive in secure location")
    summary.append("2. **Documentation**: Include reproducibility guide in dissertation appendix")
    summary.append("3. **Code Repository**: Consider publishing code repository for transparency")
    summary.append("4. **Data Sharing**: Prepare anonymized datasets for research community")
    summary.append("")
    
    # Save summary
    with open(output_dir / 'backup_summary.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    
    return summary

def main():
    """Main function for Step 15: Backup and Reproducibility Check"""
    
    print("=== Step 15: Code/Data/Results Backup and Reproducibility Check ===\n")
    
    # Setup paths
    base_dir = Path.cwd()
    output_dir = base_dir / 'step15_backup'
    output_dir.mkdir(exist_ok=True)
    
    # Create file manifest
    manifest_df = create_file_manifest(base_dir, output_dir)
    
    # Check dependencies
    deps_info = check_dependencies(base_dir, output_dir)
    
    # Validate pipeline
    validation_results = validate_analysis_pipeline(base_dir, output_dir)
    
    # Create reproducibility guide
    repro_guide = create_reproducibility_guide(base_dir, output_dir, deps_info)
    
    # Create backup archive
    archive_path = create_backup_archive(base_dir, output_dir)
    
    # Create validation script
    create_validation_script(base_dir, output_dir)
    
    # Generate summary
    backup_summary = generate_backup_summary(
        manifest_df, deps_info, validation_results, archive_path, output_dir
    )
    
    # Print final status
    print("\n=== BACKUP AND REPRODUCIBILITY CHECK COMPLETE ===")
    print(f"Backup saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - file_manifest.csv (File inventory with checksums)")
    print("  - dependencies.json (Environment and package information)")
    print("  - pipeline_validation.json (Pipeline validation results)")
    print("  - reproducibility_guide.md (Step-by-step reproduction guide)")
    print("  - backup_summary.md (Backup status summary)")
    print("  - validate_reproduction.py (Validation script)")
    print(f"  - {archive_path} (Complete backup archive)")
    
    # Check overall status
    total_steps = len(validation_results['pipeline_steps'])
    valid_steps = sum(1 for step_info in validation_results['pipeline_steps'].values() 
                     if step_info['script_exists'] and step_info['output_dir_exists'])
    
    if valid_steps == total_steps and not validation_results['missing_files']:
        print("\nPIPELINE FULLY VALIDATED AND BACKED UP")
        print("Ready for Step 16: Dissertation write-up")
    else:
        print(f"\nValidation: {valid_steps}/{total_steps} steps complete")
        print("Some files may be missing - check validation results")
    
    print(f"\nStep 15 Complete: Backup and reproducibility check finished")

if __name__ == "__main__":
    main()
