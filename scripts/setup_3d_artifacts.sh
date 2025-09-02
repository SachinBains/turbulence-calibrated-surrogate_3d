#!/bin/bash
# Setup 3D Track Artifacts Directory Structure on CSF3
# This script creates the complete artifacts_3d directory structure

set -e

# Define artifacts root
ARTIFACTS_ROOT="/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d"

echo "Setting up 3D Track artifacts structure at: $ARTIFACTS_ROOT"

# Create main directory structure
mkdir -p "$ARTIFACTS_ROOT"/{datasets,results,checkpoints,logs,cache,figures}

# Dataset directories
mkdir -p "$ARTIFACTS_ROOT"/datasets/{channel3d,boundary_layer3d,mixing_layer3d}
mkdir -p "$ARTIFACTS_ROOT"/datasets/channel3d/{raw,processed,splits}
mkdir -p "$ARTIFACTS_ROOT"/datasets/boundary_layer3d/{raw,processed,splits}
mkdir -p "$ARTIFACTS_ROOT"/datasets/mixing_layer3d/{raw,processed,splits}

# Results directories (placeholder for C3D experiments)
mkdir -p "$ARTIFACTS_ROOT"/results/{C3D1_channel_baseline_128,C3D2_channel_mc_dropout_128,C3D3_channel_ensemble_128}
mkdir -p "$ARTIFACTS_ROOT"/results/summary

# Checkpoints directories
mkdir -p "$ARTIFACTS_ROOT"/checkpoints/{C3D1_channel_baseline_128,C3D2_channel_mc_dropout_128,C3D3_channel_ensemble_128}

# Logs directories
mkdir -p "$ARTIFACTS_ROOT"/logs/{training,slurm,analysis}

# Cache directories
mkdir -p "$ARTIFACTS_ROOT"/cache/{preprocessed,features,temp}

# Figures directories
mkdir -p "$ARTIFACTS_ROOT"/figures/{3d,analysis,reports}

# Set permissions
chmod -R 755 "$ARTIFACTS_ROOT"

echo "3D Track artifacts structure created successfully!"
echo ""
echo "Directory structure:"
tree "$ARTIFACTS_ROOT" -L 3 || ls -la "$ARTIFACTS_ROOT"

echo ""
echo "Next steps:"
echo "1. Download JHTDB datasets to datasets/ directories"
echo "2. Update local repo configs to point to this artifacts root"
echo "3. Create symlinks in local repo: results/summary -> $ARTIFACTS_ROOT/results/summary"
echo "4. Verify path separation from original artifacts directory"
