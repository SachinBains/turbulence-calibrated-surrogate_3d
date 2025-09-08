#!/usr/bin/env python3
"""
Package minimal reproduction kits for experiments.
"""
import os
import sys
import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Set
import yaml

def identify_essential_files(config_path: Path, results_dir: Path) -> Dict[str, List[Path]]:
    """Identify essential files for reproduction."""
    essential_files = {
        'configs': [],
        'models': [],
        'data_splits': [],
        'scripts': [],
        'results': [],
        'requirements': []
    }
    
    # Config file
    if config_path.exists():
        essential_files['configs'].append(config_path)
    
    # Load config to understand experiment
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Data splits
        splits_dir = Path('splits')
        if splits_dir.exists():
            essential_files['data_splits'].extend(splits_dir.glob('*.json'))
    
    # Model checkpoints
    if results_dir.exists():
        # Best models
        essential_files['models'].extend(results_dir.glob('best_*.pth'))
        essential_files['models'].extend(results_dir.glob('final_*.pth'))
        
        # Key results
        essential_files['results'].extend(results_dir.glob('*_metrics_*.json'))
        essential_files['results'].extend(results_dir.glob('*_config.json'))
        essential_files['results'].extend(results_dir.glob('training_log.json'))
    
    # Core scripts
    script_dir = Path('scripts')
    core_scripts = [
        'run_train.py',
        'predict_mc.py',
        'predict_ens.py',
        'run_train_ens.py',
        'run_train_swa.py',
        'calibrate_conformal.py',
        'plot_calibration.py',
        'validate_physics.py',
        'explain_uncertainty.py'
    ]
    
    for script in core_scripts:
        script_path = script_dir / script
        if script_path.exists():
            essential_files['scripts'].append(script_path)
    
    # Requirements
    req_files = ['requirements.txt', 'environment.yml', 'pyproject.toml']
    for req_file in req_files:
        req_path = Path(req_file)
        if req_path.exists():
            essential_files['requirements'].append(req_path)
    
    return essential_files

def identify_source_dependencies(script_files: List[Path]) -> Set[Path]:
    """Identify source code dependencies by parsing imports."""
    dependencies = set()
    
    # Always include core modules
    src_dir = Path('src')
    if src_dir.exists():
        core_modules = [
            'src/utils',
            'src/models',
            'src/dataio',
            'src/train',
            'src/eval',
            'src/uq',
            'src/metrics',
            'src/physics',
            'src/interp'
        ]
        
        for module in core_modules:
            module_path = Path(module)
            if module_path.exists():
                # Add all Python files in module
                dependencies.update(module_path.rglob('*.py'))
    
    return dependencies

def create_reproduction_readme(essential_files: Dict, experiment_id: str, 
                             config_path: Path) -> str:
    """Create README for reproduction kit."""
    
    readme_content = f"""# Reproduction Kit: {experiment_id}

This package contains the minimal files needed to reproduce the experiment results.

## Contents

### Configuration
- `{config_path.name}`: Main experiment configuration

### Models
"""
    
    if essential_files['models']:
        for model_file in essential_files['models']:
            readme_content += f"- `{model_file.name}`: Trained model checkpoint\n"
    else:
        readme_content += "- No pre-trained models included (train from scratch)\n"
    
    readme_content += """
### Data Splits
"""
    
    if essential_files['data_splits']:
        for split_file in essential_files['data_splits']:
            readme_content += f"- `{split_file.name}`: Dataset split definition\n"
    
    readme_content += """
### Source Code
- `src/`: Core source code modules
- `scripts/`: Training and evaluation scripts

### Results
"""
    
    if essential_files['results']:
        for result_file in essential_files['results']:
            readme_content += f"- `{result_file.name}`: Experiment results\n"
    
    readme_content += """
## Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Or use conda if environment.yml is provided
conda env create -f environment.yml
conda activate turbml
```

### 2. Prepare Data
Place your HIT turbulence data in the `data/raw/hit/` directory according to the dataset configuration.

### 3. Training (if no pre-trained models)
```bash
# Basic training
python scripts/run_train.py --config {config_path.name}

# Ensemble training
python scripts/run_train_ens.py --config {config_path.name}

# SWA training (requires pre-trained model)
python scripts/run_train_swa.py --config {config_path.name} --pretrained results/{experiment_id}/best_model.pth
```

### 4. Evaluation
```bash
# MC Dropout prediction
python scripts/predict_mc.py --config {config_path.name} --split test

# Ensemble prediction
python scripts/predict_ens.py --config {config_path.name} --split test

# Conformal calibration
python scripts/calibrate_conformal.py --config {config_path.name} --method mc
python scripts/predict_mc.py --config {config_path.name} --split test --conformal absolute
```

### 5. Analysis
```bash
# Calibration analysis
python scripts/plot_calibration.py --results_dir results/{experiment_id} --method mc

# Physics validation
python scripts/validate_physics.py --results_dir results/{experiment_id} --method mc

# Uncertainty analysis
python scripts/explain_uncertainty.py --results_dir results/{experiment_id} --method mc
```

## Expected Results

The reproduction should yield similar metrics to the original experiment:
"""
    
    # Add key metrics if available
    if essential_files['results']:
        readme_content += """
Key performance metrics will be saved in JSON format in the results directory.
Compare your results with the included reference results.
"""
    
    readme_content += """
## Troubleshooting

### Common Issues
1. **CUDA/GPU Issues**: Add `--cuda` flag only if GPU is available and properly configured
2. **Memory Issues**: Reduce batch size in config file if encountering OOM errors
3. **Data Path Issues**: Ensure data is placed in correct directory structure
4. **Missing Dependencies**: Install additional packages as needed

### Support
For questions about reproduction, please refer to the main repository documentation.

## File Structure
```
reproduction_kit/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── {config_path.name}       # Experiment configuration
├── src/                     # Source code
├── scripts/                 # Training/evaluation scripts
├── splits/                  # Dataset splits
├── results/                 # Reference results
└── models/                  # Pre-trained models (if included)
```

Generated: {experiment_id}
"""
    
    return readme_content

def create_requirements_file(base_requirements: Path) -> str:
    """Create minimal requirements file."""
    
    # Essential packages for reproduction
    minimal_requirements = [
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "pyyaml>=5.4.0",
        "h5py>=3.3.0",
        "tqdm>=4.62.0"
    ]
    
    # Try to read existing requirements
    if base_requirements.exists():
        with open(base_requirements, 'r') as f:
            existing_reqs = f.read().strip().split('\n')
        
        # Filter to essential packages
        filtered_reqs = []
        for req in existing_reqs:
            req = req.strip()
            if req and not req.startswith('#'):
                # Keep if it's in our essential list or looks important
                package_name = req.split('>=')[0].split('==')[0].split('[')[0]
                if any(essential.split('>=')[0] == package_name for essential in minimal_requirements):
                    filtered_reqs.append(req)
        
        # Add any missing essential packages
        existing_packages = [req.split('>=')[0].split('==')[0] for req in filtered_reqs]
        for essential in minimal_requirements:
            essential_name = essential.split('>=')[0]
            if essential_name not in existing_packages:
                filtered_reqs.append(essential)
        
        return '\n'.join(filtered_reqs)
    else:
        return '\n'.join(minimal_requirements)

def package_reproduction_kit(experiment_id: str, config_path: Path, 
                           results_dir: Path, output_dir: Path,
                           include_models: bool = True,
                           include_results: bool = True) -> Path:
    """Package complete reproduction kit."""
    
    # Create temporary directory for packaging
    temp_dir = output_dir / f"{experiment_id}_reproduction_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Identify essential files
        essential_files = identify_essential_files(config_path, results_dir)
        
        # Copy configuration
        if essential_files['configs']:
            config_dest = temp_dir / config_path.name
            shutil.copy2(config_path, config_dest)
        
        # Copy source code
        src_dependencies = identify_source_dependencies(essential_files['scripts'])
        if src_dependencies:
            src_dest = temp_dir / 'src'
            src_dest.mkdir(exist_ok=True)
            
            # Copy source files maintaining structure
            for src_file in src_dependencies:
                rel_path = src_file.relative_to(Path('src'))
                dest_file = src_dest / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dest_file)
        
        # Copy scripts
        if essential_files['scripts']:
            scripts_dest = temp_dir / 'scripts'
            scripts_dest.mkdir(exist_ok=True)
            
            for script in essential_files['scripts']:
                shutil.copy2(script, scripts_dest / script.name)
        
        # Copy data splits
        if essential_files['data_splits']:
            splits_dest = temp_dir / 'splits'
            splits_dest.mkdir(exist_ok=True)
            
            for split_file in essential_files['data_splits']:
                shutil.copy2(split_file, splits_dest / split_file.name)
        
        # Copy models (optional)
        if include_models and essential_files['models']:
            models_dest = temp_dir / 'models'
            models_dest.mkdir(exist_ok=True)
            
            for model_file in essential_files['models']:
                shutil.copy2(model_file, models_dest / model_file.name)
        
        # Copy results (optional)
        if include_results and essential_files['results']:
            results_dest = temp_dir / 'results' / experiment_id
            results_dest.mkdir(parents=True, exist_ok=True)
            
            for result_file in essential_files['results']:
                shutil.copy2(result_file, results_dest / result_file.name)
        
        # Create requirements file
        req_content = create_requirements_file(Path('requirements.txt'))
        with open(temp_dir / 'requirements.txt', 'w') as f:
            f.write(req_content)
        
        # Create README
        readme_content = create_reproduction_readme(essential_files, experiment_id, config_path)
        with open(temp_dir / 'README.md', 'w') as f:
            f.write(readme_content)
        
        # Create zip archive
        zip_path = output_dir / f"{experiment_id}_reproduction_kit.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in temp_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(temp_dir)
                    zipf.write(file_path, arcname)
        
        return zip_path
    
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(description='Package minimal reproduction kit')
    parser.add_argument('--config', required=True, help='Experiment config file')
    parser.add_argument('--results_dir', required=True, help='Results directory')
    parser.add_argument('--output_dir', default='reproduction_kits', 
                       help='Output directory for packages')
    parser.add_argument('--no-models', action='store_true', 
                       help='Exclude model checkpoints (smaller package)')
    parser.add_argument('--no-results', action='store_true',
                       help='Exclude result files')
    args = parser.parse_args()
    
    config_path = Path(args.config)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract experiment ID from results directory
    experiment_id = results_dir.name
    
    print(f"Packaging reproduction kit for experiment: {experiment_id}")
    print(f"Config: {config_path}")
    print(f"Results: {results_dir}")
    print(f"Include models: {not args.no_models}")
    print(f"Include results: {not args.no_results}")
    
    # Package the kit
    zip_path = package_reproduction_kit(
        experiment_id, config_path, results_dir, output_dir,
        include_models=not args.no_models,
        include_results=not args.no_results
    )
    
    print(f"\nReproduction kit created: {zip_path}")
    print(f"Package size: {zip_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Print contents summary
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        file_count = len(zipf.namelist())
        print(f"Files included: {file_count}")
        
        # Show structure
        print("\nPackage structure:")
        dirs = set()
        for name in zipf.namelist():
            if '/' in name:
                dirs.add(name.split('/')[0])
        
        for dir_name in sorted(dirs):
            files_in_dir = [n for n in zipf.namelist() if n.startswith(dir_name + '/')]
            print(f"  {dir_name}/: {len(files_in_dir)} files")

if __name__ == '__main__':
    main()
