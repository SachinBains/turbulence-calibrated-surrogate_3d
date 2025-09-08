#!/bin/bash
#SBATCH --job-name=stage2_secondary
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/logs/slurm/stage2_secondary_%j.out
#SBATCH --error=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/logs/slurm/stage2_secondary_%j.err

# STAGE 2: Domain Shift Validation - Zero-Shot Secondary Evaluation
# Re_τ=1000 → Re_τ=5200 domain shift testing with GPU acceleration

echo "Starting STAGE 2: Domain Shift Validation (Re_τ=1000 → Re_τ=5200)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Environment setup
module load apps/anaconda3/2022.10
source activate tcs3d

export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_3d:$PYTHONPATH
export ARTIFACTS_ROOT=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d

cd /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_3d

# Fix paths and directory structure
echo "=== PHASE 0: Setup ==="
python scripts/fix_ensemble_structure.py --artifacts_dir $ARTIFACTS_ROOT

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Check which models have checkpoints before running
echo "=== STEP 4: Zero-Shot Secondary Evaluation ==="

models_to_run=()

# C3D1 - Baseline
if find $ARTIFACTS_ROOT/results/C3D1_channel_baseline_128 -name "*.pth" | head -1 | grep -q .; then
    models_to_run+=("C3D1")
    echo "✓ C3D1: Checkpoint found"
else
    echo "✗ C3D1: No checkpoint found"
fi

# C3D2 - MC Dropout
if find $ARTIFACTS_ROOT/results/C3D2_channel_mc_dropout_128 -name "*.pth" | head -1 | grep -q .; then
    models_to_run+=("C3D2")
    echo "✓ C3D2: Checkpoint found"
else
    echo "✗ C3D2: No checkpoint found"
fi

# C3D3 - Ensemble
if find $ARTIFACTS_ROOT/results/C3D3_channel_ensemble_128/members -name "*.pth" | head -1 | grep -q .; then
    models_to_run+=("C3D3")
    echo "✓ C3D3: Ensemble members found"
else
    echo "✗ C3D3: No ensemble members found"
fi

# C3D6 - Physics Informed
if find $ARTIFACTS_ROOT/results/C3D6_channel_physics_informed_128 -name "*.pth" | head -1 | grep -q .; then
    models_to_run+=("C3D6")
    echo "✓ C3D6: Checkpoint found"
else
    echo "✗ C3D6: No checkpoint found"
fi

echo "Running secondary evaluation on ${#models_to_run[@]} models: ${models_to_run[*]}"

# Run secondary evaluation for available models with GPU
for model in "${models_to_run[@]}"; do
    echo "--- Running $model secondary evaluation ---"
    python scripts/run_secondary_evaluation.py --config configs/3d_secondary/${model}_secondary_5200.yaml --cuda
    
    if [ $? -eq 0 ]; then
        echo "✓ $model secondary evaluation completed"
    else
        echo "✗ $model secondary evaluation failed"
    fi
done

# STEP 5: Basic physics validation on secondary
echo "=== STEP 5: Physics Validation on Secondary ==="
for model in "${models_to_run[@]}"; do
    echo "--- Running $model physics validation ---"
    python scripts/validate_physics.py --config configs/3d_secondary/${model}_secondary_5200.yaml --cuda
    
    if [ $? -eq 0 ]; then
        echo "✓ $model physics validation completed"
    else
        echo "✗ $model physics validation failed"
    fi
done

echo "=== DOMAIN SHIFT VALIDATION COMPLETE ==="
echo "Results saved to: $ARTIFACTS_ROOT/results/*_secondary_5200/"

# Count output files
echo "=== OUTPUT SUMMARY ==="
total_files=$(find $ARTIFACTS_ROOT/results -name "*secondary*" -type f | wc -l)
echo "Total secondary evaluation files: $total_files"

for model in "${models_to_run[@]}"; do
    if [ -d "$ARTIFACTS_ROOT/results/${model}_secondary_5200" ]; then
        count=$(find $ARTIFACTS_ROOT/results/${model}_secondary_5200 -type f | wc -l)
        echo "${model}_secondary_5200: $count files"
    fi
done

echo "Job completed at: $(date)"
