# TCS_3D: 3D Turbulence-Calibrated Surrogate Model Pipeline

## Overview
This repository contains a clean, production-ready pipeline for training 3D turbulence surrogate models using JHTDB channel flow data. The pipeline has been fully migrated from the original HIT dataset to work with 3D velocity cubes.

## Dataset
- **Data Location**: `/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/data_3d/channel_flow_smoke/`
- **Format**: 200 velocity cubes (64³ resolution) stored as `cube_64_*.h5` files
- **Variables**: 3 velocity components (u, v, w)
- **Splits**: 140 train / 30 val / 30 test

## Experiments

### C3D1: Baseline Model
- **Config**: `configs/3d/C3D1_channel_baseline_128.yaml`
- **Job**: `job/train_C3D1.slurm`
- **Method**: Deterministic U-Net3D
- **Purpose**: Baseline performance reference

### C3D2: MC Dropout
- **Config**: `configs/3d/C3D2_channel_mc_dropout_128.yaml`
- **Job**: `job/train_C3D2.slurm`
- **Method**: Monte Carlo Dropout for uncertainty quantification
- **Purpose**: Aleatoric uncertainty estimation

### C3D3: Deep Ensemble
- **Config**: `configs/3d/C3D3_channel_ensemble_128.yaml`
- **Job**: `job/train_C3D3.slurm`
- **Method**: 5-member ensemble
- **Purpose**: Epistemic uncertainty quantification

### C3D4: Variational Inference
- **Config**: `configs/3d/C3D4_channel_variational_128.yaml`
- **Job**: `job/train_C3D4.slurm`
- **Method**: Variational Bayesian neural network
- **Purpose**: Full Bayesian uncertainty quantification

### C3D5: SWAG
- **Config**: `configs/3d/C3D5_channel_swag_128.yaml`
- **Job**: `job/train_C3D5.slurm`
- **Method**: Stochastic Weight Averaging Gaussian
- **Purpose**: Posterior approximation via weight averaging

### C3D6: Physics-Informed
- **Config**: `configs/3d/C3D6_channel_physics_informed_128.yaml`
- **Job**: `job/train_C3D6.slurm`
- **Method**: Physics-informed neural network
- **Purpose**: Physics-constrained learning

## Quick Start

### 1. Submit All Experiments
```bash
cd /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_3d
bash job/submit_all_experiments.sh
```

### 2. Monitor Jobs
```bash
squeue -u p78669sb
```

### 3. Check Data Loading
```bash
python scripts/check_loader.py --config configs/3d/C3D1_channel_baseline_128.yaml --split train
```

### 4. Individual Job Submission
```bash
sbatch job/train_C3D1.slurm  # Baseline
sbatch job/train_C3D2.slurm  # MC Dropout
# etc.
```

## Key Files Fixed
- ✅ All configs updated to use correct data paths and 3 channels
- ✅ All scripts migrated from HITDataset to ChannelDataset
- ✅ SLURM jobs configured for CSF3 with correct modules
- ✅ Experiment manifest updated with correct experiment definitions
- ✅ Demo app updated for 3D channel flow data
- ✅ Splits metadata aligned with actual dataset structure

## Directory Structure
```
├── configs/3d/           # Experiment configurations
├── job/                  # SLURM job scripts
├── scripts/              # Training and evaluation scripts
├── src/                  # Source code modules
├── experiments/3d/       # Experiment tracking
├── splits/               # Dataset split metadata
└── demo/                 # Gradio demo application
```

## Environment
- **Cluster**: CSF3
- **Partition**: gpuV
- **Modules**: python/3.9, cuda/11.8, gcc/9.3.0
- **Environment**: turbml virtual environment
- **Resources**: 4 CPUs, 1 GPU, 24-32GB RAM per job

## Status
🟢 **READY FOR TRAINING** - All components cleaned and aligned for 3D channel flow data.
