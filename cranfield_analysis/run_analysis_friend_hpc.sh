#!/bin/bash
# Complete analysis pipeline for friend's HPC (n63719vm)
# Run this after setup_friend_hpc.sh and SELECTIVE data transfer

export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/turbulence-calibrated-surrogate_3d:$PYTHONPATH
export ARTIFACTS_ROOT=/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/artifacts_3d

echo "Starting complete analysis pipeline on friend's HPC..."

# PHASE 0: Verify Essential Files Only
echo "=== PHASE 0: File Verification ==="
echo "Checking for essential files (best_*.pth only)..."
for exp in C3D1_channel_baseline_128 C3D2_channel_mc_dropout_128 C3D3_channel_ensemble_128 C3D4_channel_variational_128 C3D5_channel_swag_128 C3D6_channel_physics_informed_128; do
    echo "Checking $exp..."
    ls -la $ARTIFACTS_ROOT/results/$exp/best_*.pth 2>/dev/null || echo "  WARNING: No best_*.pth found for $exp"
    ls -la $ARTIFACTS_ROOT/results/$exp/*_metrics.json 2>/dev/null || echo "  WARNING: No metrics found for $exp"
done

# PHASE 1: Generate Base Predictions (GPU)
echo "=== PHASE 1: Base Predictions (GPU) ==="
python scripts/run_eval.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda
python scripts/run_eval.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda
python scripts/run_eval.py --config configs/3d/C3D4_channel_variational_128.yaml --cuda
python scripts/run_eval.py --config configs/3d/C3D5_channel_swag_128.yaml --cuda
python scripts/run_eval.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --cuda
python scripts/run_ensemble_eval.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda

# PHASE 2: Method-Specific Predictions (GPU)
echo "=== PHASE 2: Method-Specific Predictions (GPU) ==="
python scripts/generate_baseline_predictions.py --config configs/3d/C3D1_channel_baseline_128.yaml --split val --cuda
python scripts/generate_baseline_predictions.py --config configs/3d/C3D1_channel_baseline_128.yaml --split test --cuda
python scripts/predict_mc.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --split val --cuda
python scripts/predict_mc.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --split test --cuda
python scripts/predict_ens.py --config configs/3d/C3D3_channel_ensemble_128.yaml --split val --cuda
python scripts/predict_ens.py --config configs/3d/C3D3_channel_ensemble_128.yaml --split test --cuda

# PHASE 3: Calibration and Uncertainty Analysis (GPU)
echo "=== PHASE 3: Calibration Analysis (GPU) ==="
python scripts/calibrate_conformal.py --config configs/3d/C3D1_channel_baseline_128.yaml --mode scaled --base det --cuda
python scripts/calibrate_conformal.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --mode scaled --base mc --cuda
python scripts/calibrate_conformal.py --config configs/3d/C3D3_channel_ensemble_128.yaml --mode scaled --base ens --cuda
python scripts/calibrate_conformal.py --config configs/3d/C3D4_channel_variational_128.yaml --mode scaled --base var --cuda
python scripts/calibrate_conformal.py --config configs/3d/C3D5_channel_swag_128.yaml --mode scaled --base swag --cuda
python scripts/calibrate_conformal.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --mode scaled --base det --cuda

python scripts/run_uncertainty_calibration.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda
python scripts/run_uncertainty_calibration.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda
python scripts/run_uncertainty_calibration.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda
python scripts/run_uncertainty_calibration.py --config configs/3d/C3D4_channel_variational_128.yaml --cuda
python scripts/run_uncertainty_calibration.py --config configs/3d/C3D5_channel_swag_128.yaml --cuda
python scripts/run_uncertainty_calibration.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --cuda

# PHASE 4: Robustness Analysis
echo "=== PHASE 4: Robustness Analysis ==="
python scripts/run_cross_validation.py --config configs/3d/C3D1_channel_baseline_128.yaml
python scripts/run_cross_validation.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml
python scripts/run_cross_validation.py --config configs/3d/C3D3_channel_ensemble_128.yaml
python scripts/run_cross_validation.py --config configs/3d/C3D4_channel_variational_128.yaml
python scripts/run_cross_validation.py --config configs/3d/C3D5_channel_swag_128.yaml
python scripts/run_cross_validation.py --config configs/3d/C3D6_channel_physics_informed_128.yaml

python scripts/run_adversarial_robustness.py --config configs/3d/C3D1_channel_baseline_128.yaml
python scripts/run_adversarial_robustness.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml
python scripts/run_adversarial_robustness.py --config configs/3d/C3D3_channel_ensemble_128.yaml
python scripts/run_adversarial_robustness.py --config configs/3d/C3D4_channel_variational_128.yaml
python scripts/run_adversarial_robustness.py --config configs/3d/C3D5_channel_swag_128.yaml
python scripts/run_adversarial_robustness.py --config configs/3d/C3D6_channel_physics_informed_128.yaml

# PHASE 5: Physics Validation
echo "=== PHASE 5: Physics Validation ==="
python scripts/validate_physics.py --config configs/3d/C3D1_channel_baseline_128.yaml
python scripts/validate_physics.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml
python scripts/validate_physics.py --config configs/3d/C3D3_channel_ensemble_128.yaml
python scripts/validate_physics.py --config configs/3d/C3D4_channel_variational_128.yaml
python scripts/validate_physics.py --config configs/3d/C3D5_channel_swag_128.yaml
python scripts/validate_physics.py --config configs/3d/C3D6_channel_physics_informed_128.yaml

# PHASE 6: Visualization and Reporting
echo "=== PHASE 6: Visualization ==="
python scripts/make_slice_maps.py --config configs/3d/C3D1_channel_baseline_128.yaml
python scripts/make_slice_maps.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml
python scripts/make_slice_maps.py --config configs/3d/C3D3_channel_ensemble_128.yaml
python scripts/make_slice_maps.py --config configs/3d/C3D4_channel_variational_128.yaml
python scripts/make_slice_maps.py --config configs/3d/C3D5_channel_swag_128.yaml
python scripts/make_slice_maps.py --config configs/3d/C3D6_channel_physics_informed_128.yaml

python scripts/generate_report.py --config configs/3d/C3D1_channel_baseline_128.yaml
python scripts/generate_report.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml
python scripts/generate_report.py --config configs/3d/C3D3_channel_ensemble_128.yaml
python scripts/generate_report.py --config configs/3d/C3D4_channel_variational_128.yaml
python scripts/generate_report.py --config configs/3d/C3D5_channel_swag_128.yaml
python scripts/generate_report.py --config configs/3d/C3D6_channel_physics_informed_128.yaml

# PHASE 7: Global Analysis
echo "=== PHASE 7: Global Analysis ==="
python scripts/compare_uq.py --results_dir $ARTIFACTS_ROOT/results
python scripts/step9_aggregate_results.py --results_dir $ARTIFACTS_ROOT/results
python scripts/step11_quantitative_comparison.py --artifacts_dir $ARTIFACTS_ROOT
python scripts/step12_physics_validation.py --artifacts_dir $ARTIFACTS_ROOT
python scripts/step14_summary_report.py --artifacts_dir $ARTIFACTS_ROOT

echo "✅ Complete analysis pipeline finished!"
echo "Results available in: $ARTIFACTS_ROOT/results/"
