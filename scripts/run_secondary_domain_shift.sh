#!/bin/bash
# STAGE 2: Domain Shift Validation - Zero-Shot Secondary Evaluation
# Re_τ=1000 → Re_τ=5200 domain shift testing with existing trained models

export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/turbulence-calibrated-surrogate_3d:$PYTHONPATH
export ARTIFACTS_ROOT=/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/artifacts_3d

echo "Starting STAGE 2: Domain Shift Validation (Re_τ=1000 → Re_τ=5200)"

# Fix paths for friend's environment
echo "=== PHASE 0: Setup ==="
python cranfield_analysis/update_paths_for_friend.py
python scripts/fix_ensemble_structure.py --artifacts_dir $ARTIFACTS_ROOT

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# STEP 4: Deploy primary models on secondary dataset (Re_τ=5200)
echo "=== STEP 4: Zero-Shot Secondary Evaluation ==="

# Check which models have checkpoints before running
models_to_run=()
for model in C3D1 C3D2 C3D3 C3D6; do
    if [ "$model" = "C3D3" ]; then
        # Check ensemble members
        if find $ARTIFACTS_ROOT/results/C3D3_channel_ensemble_128/members -name "*.pth" | head -1 | grep -q .; then
            models_to_run+=("$model")
            echo "✓ $model: Ensemble members found"
        else
            echo "✗ $model: No ensemble members found"
        fi
    else
        # Check regular checkpoints
        if find $ARTIFACTS_ROOT/results/${model}_channel_*_128 -name "*.pth" | head -1 | grep -q .; then
            models_to_run+=("$model")
            echo "✓ $model: Checkpoint found"
        else
            echo "✗ $model: No checkpoint found"
        fi
    fi
done

echo "Running secondary evaluation on ${#models_to_run[@]} models: ${models_to_run[*]}"

# Run secondary evaluation for available models
for model in "${models_to_run[@]}"; do
    echo "--- Running $model secondary evaluation ---"
    python scripts/run_secondary_evaluation.py --config configs/3d_secondary/${model}_secondary_5200.yaml --cuda
done

# STEP 5: Basic physics validation on secondary
echo "=== STEP 5: Physics Validation on Secondary ==="
for model in "${models_to_run[@]}"; do
    echo "--- Running $model physics validation ---"
    python scripts/validate_physics.py --config configs/3d_secondary/${model}_secondary_5200.yaml --cuda
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
