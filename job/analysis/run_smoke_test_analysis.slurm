#!/bin/bash
#SBATCH --job-name=smoke_analysis
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/logs/slurm/smoke_analysis_%j.out
#SBATCH --error=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/logs/slurm/smoke_analysis_%j.err

# Load modules
module load apps/anaconda3/2022.05
source activate turbulence_3d

# Set environment variables
export ARTIFACTS_ROOT=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d
export DATA_DIR=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/data_3d/channel_flow_smoke

# Navigate to project directory
cd /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_3d

echo "Starting comprehensive smoke test analysis..."
echo "Date: $(date)"
echo "Node: $(hostname)"

# Create analysis output directory
mkdir -p $ARTIFACTS_ROOT/analysis/smoke_test_complete

# Run complete analysis pipeline for all smoke test experiments
python scripts/run_complete_analysis.py \
    --base_dir $ARTIFACTS_ROOT \
    --output_dir $ARTIFACTS_ROOT/analysis/smoke_test_complete \
    --experiments C3D1_channel_baseline_128 C3D2_channel_mc_dropout_128 C3D3_channel_ensemble_128 C3D4_channel_variational_128 C3D5_channel_swag_128 C3D6_channel_physics_informed_128 \
    --data_config configs/channel_flow_3d_smoke.yaml \
    --cpu_only

echo "Smoke test analysis completed at $(date)"
echo "Results saved to: $ARTIFACTS_ROOT/analysis/smoke_test_complete"
