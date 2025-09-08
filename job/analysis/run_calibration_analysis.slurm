#!/bin/bash
#SBATCH --job-name=calibration_analysis
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/logs/slurm/calibration_analysis_%j.out
#SBATCH --error=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/logs/slurm/calibration_analysis_%j.err

# CALIBRATION ANALYSIS FOR ALL PRIMARY MODELS
echo "Starting Calibration Analysis for C3D1-C3D6"
echo "Job ID: $SLURM_JOB_ID"

# Environment setup
module load apps/anaconda3/2022.10
source activate tcs3d

# Set environment variables
export ARTIFACTS_ROOT=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d
export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_3d:$PYTHONPATH

cd /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_3d

echo "=== Generating Calibration Plots for All Primary Models ==="
python scripts/generate_all_calibration_plots.py \
    --artifacts_root $ARTIFACTS_ROOT \
    --output_dir $ARTIFACTS_ROOT/figures/calibration_primary \
    --split test

echo "=== Calibration Analysis Completed at: $(date) ==="
