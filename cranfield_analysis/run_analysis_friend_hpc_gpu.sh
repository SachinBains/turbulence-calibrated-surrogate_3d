#!/bin/bash
# GPU-optimized analysis pipeline for friend's HPC (n63719vm)
# Utilizes 2-GPU allocation efficiently

export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/turbulence-calibrated-surrogate_3d:$PYTHONPATH
export ARTIFACTS_ROOT=/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/artifacts_3d

# Set CUDA environment for 2 GPUs
export CUDA_VISIBLE_DEVICES=0,1

echo "Starting GPU-optimized analysis pipeline on friend's HPC..."
echo "Available GPUs: $(nvidia-smi --list-gpus | wc -l)"

# PHASE 0: Verify Essential Files Only
echo "=== PHASE 0: File Verification ==="
echo "Checking for essential files (best_*.pth only)..."
for exp in C3D1_channel_baseline_128 C3D2_channel_mc_dropout_128 C3D3_channel_ensemble_128 C3D4_channel_variational_128 C3D5_channel_swag_128 C3D6_channel_physics_informed_128; do
    echo "Checking $exp..."
    ls -la $ARTIFACTS_ROOT/results/$exp/best_*.pth 2>/dev/null || echo "  WARNING: No best_*.pth found for $exp"
    ls -la $ARTIFACTS_ROOT/results/$exp/*_metrics.json 2>/dev/null || echo "  WARNING: No metrics found for $exp"
done

# PHASE 1: Generate Base Predictions (GPU Accelerated)
echo "=== PHASE 1: Base Predictions (GPU) ==="
# Run on GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/run_eval.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda &
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda &
wait

CUDA_VISIBLE_DEVICES=0 python scripts/run_eval.py --config configs/3d/C3D4_channel_variational_128.yaml --cuda &
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval.py --config configs/3d/C3D5_channel_swag_128.yaml --cuda &
wait

CUDA_VISIBLE_DEVICES=0 python scripts/run_eval.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --cuda &
CUDA_VISIBLE_DEVICES=1 python scripts/run_ensemble_eval.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda &
wait

# PHASE 2: Method-Specific Predictions (Parallel GPU)
echo "=== PHASE 2: Method-Specific Predictions (Parallel GPU) ==="
# Baseline predictions
CUDA_VISIBLE_DEVICES=0 python scripts/generate_baseline_predictions.py --config configs/3d/C3D1_channel_baseline_128.yaml --split val --cuda &
CUDA_VISIBLE_DEVICES=1 python scripts/generate_baseline_predictions.py --config configs/3d/C3D1_channel_baseline_128.yaml --split test --cuda &
wait

# MC Dropout predictions
CUDA_VISIBLE_DEVICES=0 python scripts/predict_mc.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --split val --cuda &
CUDA_VISIBLE_DEVICES=1 python scripts/predict_mc.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --split test --cuda &
wait

# Ensemble predictions
CUDA_VISIBLE_DEVICES=0 python scripts/predict_ens.py --config configs/3d/C3D3_channel_ensemble_128.yaml --split val --cuda &
CUDA_VISIBLE_DEVICES=1 python scripts/predict_ens.py --config configs/3d/C3D3_channel_ensemble_128.yaml --split test --cuda &
wait

# PHASE 3: Calibration and Uncertainty Analysis (GPU)
echo "=== PHASE 3: Calibration Analysis (GPU) ==="
# Conformal calibration - parallel execution
CUDA_VISIBLE_DEVICES=0 python scripts/calibrate_conformal.py --config configs/3d/C3D1_channel_baseline_128.yaml --mode scaled --base det --cuda &
CUDA_VISIBLE_DEVICES=1 python scripts/calibrate_conformal.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --mode scaled --base mc --cuda &
wait

CUDA_VISIBLE_DEVICES=0 python scripts/calibrate_conformal.py --config configs/3d/C3D3_channel_ensemble_128.yaml --mode scaled --base ens --cuda &
CUDA_VISIBLE_DEVICES=1 python scripts/calibrate_conformal.py --config configs/3d/C3D4_channel_variational_128.yaml --mode scaled --base var --cuda &
wait

CUDA_VISIBLE_DEVICES=0 python scripts/calibrate_conformal.py --config configs/3d/C3D5_channel_swag_128.yaml --mode scaled --base swag --cuda &
CUDA_VISIBLE_DEVICES=1 python scripts/calibrate_conformal.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --mode scaled --base det --cuda &
wait

# Uncertainty calibration - parallel execution
CUDA_VISIBLE_DEVICES=0 python scripts/run_uncertainty_calibration.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda &
CUDA_VISIBLE_DEVICES=1 python scripts/run_uncertainty_calibration.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda &
wait

CUDA_VISIBLE_DEVICES=0 python scripts/run_uncertainty_calibration.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda &
CUDA_VISIBLE_DEVICES=1 python scripts/run_uncertainty_calibration.py --config configs/3d/C3D4_channel_variational_128.yaml --cuda &
wait

CUDA_VISIBLE_DEVICES=0 python scripts/run_uncertainty_calibration.py --config configs/3d/C3D5_channel_swag_128.yaml --cuda &
CUDA_VISIBLE_DEVICES=1 python scripts/run_uncertainty_calibration.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --cuda &
wait

# PHASE 4: Visualization and Analysis (CPU - less GPU intensive)
echo "=== PHASE 4: Visualization and Analysis ==="
python scripts/make_slice_maps.py --config configs/3d/C3D1_channel_baseline_128.yaml &
python scripts/make_slice_maps.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml &
wait

python scripts/make_slice_maps.py --config configs/3d/C3D3_channel_ensemble_128.yaml &
python scripts/make_slice_maps.py --config configs/3d/C3D4_channel_variational_128.yaml &
wait

python scripts/make_slice_maps.py --config configs/3d/C3D5_channel_swag_128.yaml &
python scripts/make_slice_maps.py --config configs/3d/C3D6_channel_physics_informed_128.yaml &
wait

# PHASE 5: Physics Validation (GPU for model inference)
echo "=== PHASE 5: Physics Validation (GPU) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/validate_physics.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda &
CUDA_VISIBLE_DEVICES=1 python scripts/validate_physics.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda &
wait

CUDA_VISIBLE_DEVICES=0 python scripts/validate_physics.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda &
CUDA_VISIBLE_DEVICES=1 python scripts/validate_physics.py --config configs/3d/C3D4_channel_variational_128.yaml --cuda &
wait

CUDA_VISIBLE_DEVICES=0 python scripts/validate_physics.py --config configs/3d/C3D5_channel_swag_128.yaml --cuda &
CUDA_VISIBLE_DEVICES=1 python scripts/validate_physics.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --cuda &
wait

# PHASE 6: Report Generation (CPU)
echo "=== PHASE 6: Report Generation ==="
python scripts/generate_report.py --config configs/3d/C3D1_channel_baseline_128.yaml &
python scripts/generate_report.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml &
wait

python scripts/generate_report.py --config configs/3d/C3D3_channel_ensemble_128.yaml &
python scripts/generate_report.py --config configs/3d/C3D4_channel_variational_128.yaml &
wait

python scripts/generate_report.py --config configs/3d/C3D5_channel_swag_128.yaml &
python scripts/generate_report.py --config configs/3d/C3D6_channel_physics_informed_128.yaml &
wait

# PHASE 7: Global Analysis (CPU)
echo "=== PHASE 7: Global Analysis ==="
python scripts/compare_uq.py --results_dir $ARTIFACTS_ROOT/results
python scripts/step9_aggregate_results.py --results_dir $ARTIFACTS_ROOT/results
python scripts/step11_quantitative_comparison.py --artifacts_dir $ARTIFACTS_ROOT
python scripts/step12_physics_validation.py --artifacts_dir $ARTIFACTS_ROOT
python scripts/step14_summary_report.py --artifacts_dir $ARTIFACTS_ROOT

echo "✅ GPU-optimized analysis pipeline finished!"
echo "Results available in: $ARTIFACTS_ROOT/results/"
echo "GPU utilization logs: nvidia-smi"
