#!/bin/bash
# SMOKE TEST ANALYSIS - FIXED FOR EXISTING MODELS
# Friend's HPC (n63719vm) - Run analysis on smoke test data with existing C3D1-C3D6 models

export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/turbulence-calibrated-surrogate_3d:$PYTHONPATH
export ARTIFACTS_ROOT=/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/artifacts_3d

echo "Starting smoke test analysis with existing models..."

# PHASE 0: Fix all issues first
echo "=== PHASE 0: Fix Issues ==="
python cranfield_analysis/update_paths_for_friend.py
python scripts/fix_ensemble_structure.py --artifacts_dir $ARTIFACTS_ROOT
python scripts/fix_smoke_test_splits.py --artifacts_root $ARTIFACTS_ROOT

# Clear Python cache to avoid old path issues
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "=== Verifying Models ==="
for exp in C3D1_channel_baseline_128 C3D2_channel_mc_dropout_128 C3D3_channel_ensemble_128 C3D4_channel_variational_128 C3D5_channel_swag_128 C3D6_channel_physics_informed_128; do
    echo "Checking $exp..."
    find $ARTIFACTS_ROOT/results/$exp -name "best_*.pth" | head -1
done

# PHASE 1: Generate Base Predictions (Only existing models)
echo "=== PHASE 1: Base Predictions (GPU) ==="
python scripts/run_eval.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda
python scripts/run_eval.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda
python scripts/run_ensemble_eval.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda

# Only run C3D4-C3D6 if checkpoints exist
if find $ARTIFACTS_ROOT/results/C3D4_channel_variational_128 -name "best_*.pth" | head -1 | grep -q .; then
    python scripts/run_eval.py --config configs/3d/C3D4_channel_variational_128.yaml --cuda
else
    echo "Skipping C3D4 - no checkpoints found"
fi

if find $ARTIFACTS_ROOT/results/C3D5_channel_swag_128 -name "best_*.pth" | head -1 | grep -q .; then
    python scripts/run_eval.py --config configs/3d/C3D5_channel_swag_128.yaml --cuda
else
    echo "Skipping C3D5 - no checkpoints found"
fi

if find $ARTIFACTS_ROOT/results/C3D6_channel_physics_informed_128 -name "best_*.pth" | head -1 | grep -q .; then
    python scripts/run_eval.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --cuda
else
    echo "Skipping C3D6 - no checkpoints found"
fi

echo "✅ Smoke test analysis complete!"
