# Turbulence-Calibrated-Surrogate (3D Track)

A comprehensive 3D turbulence surrogate model with advanced uncertainty quantification, physics validation, and interpretability analysis.

> **Note**: This repository is the **3D development track** for complex JHTDB datasets and advanced 3D cube analysis. The baseline 2D HIT work is maintained in the original repository: [turbulence-calibrated-surrogate_full](https://github.com/SachinBains/turbulence-calibrated-surrogate_full).

## 3D Track Overview

This repository extends the baseline turbulence surrogate framework to handle:
- **3D JHTDB datasets** with complex boundary conditions
- **Y+ band analysis** for wall-bounded turbulence
- **Advanced 3D cube processing** with higher resolution
- **Separate artifact management** under `/artifacts_3d/` structure

## Features

- **Multiple UQ Methods**: MC Dropout, Deep Ensembles, Stochastic Weight Averaging (SWA)
- **Conformal Prediction**: Distribution-free uncertainty intervals with coverage guarantees
- **Physics Validation**: Turbulence-specific metrics (energy spectra, Reynolds stress, incompressibility)
- **Calibration Analysis**: Reliability diagrams, ECE/MCE metrics, coverage assessment
- **Interpretability Suite**: Local attributions, global feature importance, uncertainty analysis
- **Automated Reporting**: Publication-ready figures and comprehensive HTML reports
- **Cluster Support**: Slurm job scripts for high-performance computing
- **Reproducibility**: Minimal reproduction kits for easy sharing

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Training
```bash
# MC Dropout training
python scripts/run_train.py --config configs/E1_hit_baseline.yaml --cuda

# Deep Ensemble training  
python scripts/run_train_ens.py --config configs/E5_hit_ens.yaml --cuda

# SWA fine-tuning (requires pretrained model)
python scripts/run_train_swa.py --config configs/E1_hit_baseline.yaml --pretrained results/E1_hit_baseline/best_model.pth --cuda
```

### Prediction with UQ
```bash
# MC Dropout prediction
python scripts/predict_mc.py --config configs/E1_hit_baseline.yaml --split test --cuda

# Ensemble prediction
python scripts/predict_ens.py --config configs/E5_hit_ens.yaml --split test --cuda
```

### Conformal Prediction
```bash
# Calibrate conformal quantiles
python scripts/calibrate_conformal.py --config configs/E1_hit_baseline.yaml --method mc

# Predict with conformal intervals
python scripts/predict_mc.py --config configs/E1_hit_baseline.yaml --split test --conformal absolute --cuda
```

## Analysis & Validation

### Calibration Analysis
```bash
# Generate reliability diagrams and calibration metrics
python scripts/plot_calibration.py --results_dir results/E1_hit_baseline --method mc --split test

# Comprehensive calibration analysis for ensembles
python scripts/plot_calibration.py --results_dir results/E5_hit_ens --method ens --split test
```

### Physics Validation
```bash
# Validate turbulence physics (energy spectra, incompressibility, Reynolds stress)
python scripts/validate_physics.py --results_dir results/E1_hit_baseline --method mc --split test

# Physics validation with custom parameters
python scripts/validate_physics.py --results_dir results/E5_hit_ens --method ens --split test --dx 1.0 --nu 1e-4
```

### Uncertainty Analysis
```bash
# Analyze uncertainty-error correlations and spatial patterns
python scripts/explain_uncertainty.py --results_dir results/E1_hit_baseline --method mc --split test

# Binning analysis with custom bins
python scripts/explain_uncertainty.py --results_dir results/E5_hit_ens --method ens --split test --n_bins 15
```

## Interpretability Suite

### Local Attributions
Generate per-sample saliency maps using Integrated Gradients, GradientSHAP, or 3D occlusion:

```bash
# Integrated Gradients for baseline model
python scripts/explain_local.py --config configs/E1_hit_baseline.yaml --split val --method ig --n 2

# 3D Occlusion for MC dropout model  
python scripts/explain_local.py --config configs/E4_hit_ab_dropout.yaml --split test --method occlusion --n 2

# GradientSHAP
python scripts/explain_local.py --config configs/E2_hit_bayes.yaml --split val --method gradshap --n 4
```

### Faithfulness Evaluation
Measure attribution quality via top-k ablation curves:

```bash
# Test faithfulness of Integrated Gradients
python scripts/faithfulness.py --config configs/E1_hit_baseline.yaml --split val --method ig --k_list 0.05 0.1 0.2

# Test occlusion faithfulness
python scripts/faithfulness.py --config configs/E4_hit_ab_dropout.yaml --split test --method occlusion --k_list 0.1 0.2 0.3
```

### Global Feature Importance
Analyze which turbulence features correlate with model error or uncertainty:

```bash
# Feature importance for prediction error (baseline models)
python scripts/explain_global.py --config configs/E1_hit_baseline.yaml --split test --target error

# Feature importance for uncertainty (MC models)
python scripts/explain_global.py --config configs/E4_hit_ab_dropout.yaml --split test --target sigma
```

## Report Generation

### Publication Figures
```bash
# Generate publication-ready comparison figures
python scripts/make_figures.py --results_dir results/E1_hit_baseline --methods mc ens

# Custom output directory
python scripts/make_figures.py --results_dir results/E5_hit_ens --methods mc ens --output_dir custom_figures/
```

### HTML Reports
```bash
# Generate comprehensive HTML report with embedded figures
python scripts/generate_report.py --results_dir results/E1_hit_baseline --methods mc ens

# Multi-method comparison report
python scripts/generate_report.py --results_dir results/E5_hit_ens --methods mc ens --output_dir reports/
```

### Reproduction Kits
```bash
# Package minimal reproduction kit
python scripts/package_minimal.py --config configs/E1_hit_baseline.yaml --results_dir results/E1_hit_baseline

# Exclude models for smaller package
python scripts/package_minimal.py --config configs/E5_hit_ens.yaml --results_dir results/E5_hit_ens --no-models
```

## Cluster Usage (CSF3)

### Job Submission
```bash
# Make submission script executable
chmod +x job/submit_jobs.sh

# Submit training job
./job/submit_jobs.sh train configs/E1_hit_baseline.yaml mc

# Submit ensemble training
./job/submit_jobs.sh train configs/E5_hit_ens.yaml ensemble

# Submit evaluation pipeline
./job/submit_jobs.sh eval configs/E1_hit_baseline.yaml mc test

# Run batch experiments
./job/submit_jobs.sh batch
```

### Job Management
```bash
# Check job status
./job/submit_jobs.sh status [JOB_ID]

# View job logs
./job/submit_jobs.sh logs JOB_ID

# Cancel job
./job/submit_jobs.sh cancel JOB_ID

# Run complete pipeline (training + evaluation)
./job/submit_jobs.sh pipeline configs/E1_hit_baseline.yaml mc
```

## Configuration Files

- **`E1_hit_baseline.yaml`**: Baseline deterministic model
- **`E2_hit_bayes.yaml`**: Bayesian (MC Dropout) model
- **`E5_hit_ens.yaml`**: Deep ensemble (5 members)
- **`E6_hit_ab_ens.yaml`**: Cross-domain ensemble (A→B)

## Device Usage

All scripts default to CPU. Add `--cuda` flag to use GPU:

```bash
# Run on CPU (default)
python scripts/run_train.py --config configs/E1_hit_baseline.yaml

# Run on GPU
python scripts/run_train.py --config configs/E1_hit_baseline.yaml --cuda
```

## Results Structure

```
results/
├── {experiment_id}/
│   ├── best_model.pth              # Best trained model
│   ├── training_log.json           # Training metrics
│   ├── {method}_metrics_{split}.json    # Prediction metrics
│   ├── {method}_mean_{split}.npy        # Mean predictions
│   ├── {method}_var_{split}.npy         # Prediction variance
│   └── conformal_{method}_{mode}.json   # Conformal calibration

figures/
├── {experiment_id}/
│   ├── reliability_{method}_{split}.png      # Reliability diagrams
│   ├── calibration_scatter_{method}_{split}.png
│   ├── uncertainty_error_scatter_{method}_{split}.png
│   ├── physics_validation_summary.png
│   ├── performance_comparison.png
│   └── analysis_report.html              # Comprehensive report
```

## Troubleshooting

### Common Issues
- **CUDA out of memory**: Reduce batch size in config or use `--no-cuda`
- **Missing data**: Ensure HIT data is in `data/raw/hit/` directory
- **Import errors**: Check Python path and virtual environment activation
- **Slurm issues**: Verify module loading and queue availability on cluster

### Performance Tips
- Use `--cuda` for GPU acceleration when available
- Increase `num_workers` in configs for faster data loading
- Use ensemble prediction in batches for memory efficiency
- Enable mixed precision training for larger models
