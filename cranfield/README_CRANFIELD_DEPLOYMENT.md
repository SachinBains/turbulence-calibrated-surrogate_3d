# C3D4 Variational Model - Cranfield HPC Deployment Package

## Complete File Dependencies

This package contains ALL files needed to run C3D4 variational training on Cranfield HPC.

### Files Included:
- `C3D4_cranfield.yaml` - Main config file with Cranfield paths
- `train_C3D4_cranfield.slurm` - SLURM job script
- `requirements.txt` - Python dependencies
- `setup_cranfield.sh` - Automated setup script
- `data_transfer_guide.md` - Data transfer instructions

### Source Code Dependencies (already in main repo):
- `src/models/variational_unet3d.py` - Variational U-Net implementation
- `src/train/variational_trainer.py` - ELBO training loop with β warm-up
- `src/train/losses.py` - Variational ELBO loss function
- `src/dataio/channel_dataset.py` - Dataset loader
- `scripts/run_train.py` - Training script (routes to variational trainer)

## Quick Deployment Steps:

1. **Transfer data package** (see `data_transfer_guide.md`)
2. **Clone repository** on Cranfield HPC
3. **Run setup script**: `./cranfield/setup_cranfield.sh`
4. **Submit job**: `sbatch cranfield/train_C3D4_cranfield.slurm`

## Expected Training:
- **Duration**: ~18-25 hours (200 epochs)
- **GPU Memory**: ~8-12GB
- **Output**: `/home/[username]/artifacts_3d/results/C3D4_channel_primary_final_1000/`

## Key Features:
- ✅ Variational Bayesian U-Net with full ELBO loss
- ✅ KL divergence β warm-up (1e-7 → 6e-7 over 30 epochs)
- ✅ Early stopping (patience=20)
- ✅ Mixed precision training
- ✅ Automatic checkpointing

All paths are automatically updated by the setup script.
