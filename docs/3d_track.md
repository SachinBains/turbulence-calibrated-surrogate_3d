# 3D Track Development Guide

## Overview

This repository is the **3D development track** for turbulence surrogate modeling, focusing on complex JHTDB datasets and advanced 3D analysis capabilities. It maintains complete separation from the baseline 2D HIT work.

## Goals

### Primary Objectives
- **3D JHTDB Integration**: Support for channel flow, boundary layer, and other complex 3D datasets
- **Y+ Band Analysis**: Wall-bounded turbulence with proper y+ scaling and analysis
- **Advanced UQ Methods**: Enhanced Monte Carlo and ensemble approaches for 3D data
- **High-Resolution Processing**: Support for larger 3D cubes with efficient memory management

### Technical Targets
- **Dataset Support**: Channel3D, BoundaryLayer3D, MixingLayer3D from JHTDB
- **Resolution Scaling**: 128³, 256³, and 512³ cube processing
- **Memory Optimization**: Efficient 3D convolution and gradient computation
- **Physics Validation**: 3D-specific turbulence metrics and validation

## Experiment ID Scheme

### Reserved IDs for 3D Track
- **C3D1_***: Channel flow baseline experiments
- **C3D2_***: Channel flow MC Dropout experiments  
- **C3D3_***: Channel flow ensemble experiments
- **C3D4_***: Boundary layer experiments
- **C3D5_***: Mixing layer experiments
- **C3D6_***: Cross-dataset transfer experiments

### Naming Convention
```
C3D{dataset_id}_{method}_{domain}_{variant}
```

Examples:
- `C3D1_channel_baseline_128`: Channel flow baseline at 128³ resolution
- `C3D2_channel_mc_dropout_256`: Channel flow MC Dropout at 256³ resolution
- `C3D3_channel_ensemble_wall`: Channel flow ensemble focusing on wall region

## Artifacts Structure

### CSF3 Artifacts Root
```
/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/
├── datasets/
│   ├── channel3d/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── splits/
│   ├── boundary_layer3d/
│   └── mixing_layer3d/
├── results/
│   ├── C3D1_*/
│   ├── C3D2_*/
│   └── C3D3_*/
├── checkpoints/
│   ├── C3D1_*/
│   └── C3D2_*/
├── logs/
│   ├── training/
│   └── slurm/
└── cache/
    ├── preprocessed/
    └── features/
```

### Local Results Symlinks
The local repository maintains symlinks to CSF3 artifacts:
```
results/summary/ -> /path/to/artifacts_3d/results/summary/
figures/3d/ -> /path/to/artifacts_3d/figures/
```

## Dataset Specifications

### Channel3D Dataset
- **Source**: JHTDB Channel Flow
- **Resolution**: 2048 × 512 × 1536 (streamwise × wall-normal × spanwise)
- **Re_τ**: 1000
- **Y+ Range**: 0.5 to 1000
- **Variables**: u, v, w, p (velocity components + pressure)

### Processing Pipeline
1. **Download**: JHTDB API integration for 3D cube extraction
2. **Preprocessing**: Y+ scaling, normalization, cube extraction
3. **Splitting**: Train/val/test with proper temporal separation
4. **Caching**: Efficient HDF5 storage for large 3D arrays

## Memory Management

### 3D Cube Processing
- **Batch Size**: Adaptive based on GPU memory (typically 1-4 for 128³)
- **Gradient Checkpointing**: For deeper networks with 3D convolutions
- **Mixed Precision**: FP16 training for memory efficiency
- **Streaming**: On-demand loading for large datasets

### Optimization Strategies
- **Patch-based Training**: Train on smaller patches, evaluate on full cubes
- **Progressive Resolution**: Start with 64³, scale to 128³, then 256³
- **Distributed Training**: Multi-GPU support for ensemble training

## Analysis Extensions

### 3D-Specific Metrics
- **Wall Distance Analysis**: Y+ band performance evaluation
- **Anisotropy Metrics**: Reynolds stress anisotropy tensor analysis
- **3D Energy Spectra**: Full 3D k-space analysis
- **Coherent Structure Detection**: Q-criterion, λ₂-criterion analysis

### Visualization Enhancements
- **3D Isosurfaces**: Vorticity and Q-criterion visualization
- **Wall-Normal Profiles**: Y+ scaling and comparison with DNS
- **Cross-Sectional Analysis**: Streamwise and spanwise plane analysis
- **Interactive 3D Plots**: Web-based visualization for exploration

## Development Workflow

### Phase 1: Infrastructure Setup
1. JHTDB API integration and data pipeline
2. 3D dataset loaders and preprocessing
3. Memory-efficient 3D model architectures
4. Basic training and evaluation scripts

### Phase 2: UQ Implementation
1. 3D MC Dropout with proper scaling
2. 3D ensemble training with distributed support
3. Conformal prediction for 3D outputs
4. Uncertainty visualization for 3D fields

### Phase 3: Physics Validation
1. 3D turbulence metrics implementation
2. Wall-bounded flow validation
3. Cross-dataset transfer evaluation
4. Physics-informed loss functions

### Phase 4: Advanced Analysis
1. 3D interpretability methods
2. Coherent structure analysis
3. Multi-scale uncertainty quantification
4. Production deployment optimization

## Safety and Separation

### Isolation Guarantees
- **No Cross-Contamination**: Zero references to original `/artifacts/` path
- **Independent Configs**: All configs point to `/artifacts_3d/` structure
- **Separate Workspaces**: Distinct Cascade workspace for 3D development
- **Version Control**: Upstream tracking for selective cherry-picking

### Validation Checks
- **Path Validation**: Automated checks for artifact path separation
- **Config Validation**: Ensure no references to original experiment IDs (E1-E6)
- **Workspace Isolation**: Verify independent operation of both tracks
- **Data Integrity**: Checksums and validation for 3D datasets

## Getting Started

### Prerequisites
- CSF3 access with `/artifacts_3d/` directory setup
- JHTDB account and API credentials
- GPU nodes with sufficient memory (≥32GB for 128³ cubes)
- Python environment with 3D-specific dependencies

### Quick Start
```bash
# Clone and setup 3D track
git clone https://github.com/SachinBains/turbulence-calibrated-surrogate_3d.git
cd turbulence-calibrated-surrogate_3d

# Setup artifacts structure on CSF3
./scripts/setup_3d_artifacts.sh

# Download and preprocess channel3d dataset
python scripts/download_jhtdb.py --dataset channel3d --resolution 128

# Train first 3D baseline model
python scripts/run_train.py --config configs/C3D1_channel_baseline_128.yaml --cuda
```

### Next Steps
1. Review experiment configs in `configs/3d/`
2. Check dataset availability in `/artifacts_3d/datasets/`
3. Run validation pipeline with `scripts/validate_3d_setup.py`
4. Submit first training job with `job/submit_3d_jobs.sh`
