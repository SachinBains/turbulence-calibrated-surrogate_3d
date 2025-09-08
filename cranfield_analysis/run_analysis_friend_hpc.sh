#!/bin/bash
# Complete analysis pipeline for friend's HPC (n63719vm)
# Run this after setup_friend_hpc.sh and data transfer

export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/turbulence-calibrated-surrogate_3d:$PYTHONPATH
export ARTIFACTS_ROOT=/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/artifacts_3d

echo "Starting complete analysis pipeline on friend's HPC..."

# PHASE 1: Generate Base Predictions
echo "=== PHASE 1: Base Predictions ==="
python scripts/run_eval.py --config configs/3d/C3D1_channel_baseline_128.yaml
python scripts/run_eval.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml
python scripts/run_eval.py --config configs/3d/C3D4_channel_variational_128.yaml
python scripts/run_eval.py --config configs/3d/C3D5_channel_swag_128.yaml
python scripts/run_eval.py --config configs/3d/C3D6_channel_physics_informed_128.yaml
python scripts/run_ensemble_eval.py --config configs/3d/C3D3_channel_ensemble_128.yaml

# PHASE 2: Method-Specific Predictions
echo "=== PHASE 2: Method-Specific Predictions ==="
python scripts/generate_baseline_predictions.py --config configs/3d/C3D1_channel_baseline_128.yaml --split val
python scripts/generate_baseline_predictions.py --config configs/3d/C3D1_channel_baseline_128.yaml --split test
python scripts/predict_mc.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --split val
python scripts/predict_mc.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --split test
python scripts/predict_ens.py --config configs/3d/C3D3_channel_ensemble_128.yaml --split val
python scripts/predict_ens.py --config configs/3d/C3D3_channel_ensemble_128.yaml --split test

# PHASE 3: Calibration and Uncertainty Analysis
echo "=== PHASE 3: Calibration Analysis ==="
python scripts/calibrate_conformal.py --config configs/3d/C3D1_channel_baseline_128.yaml --mode scaled --base det
python scripts/calibrate_conformal.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --mode scaled --base mc
python scripts/calibrate_conformal.py --config configs/3d/C3D3_channel_ensemble_128.yaml --mode scaled --base ens
python scripts/calibrate_conformal.py --config configs/3d/C3D4_channel_variational_128.yaml --mode scaled --base var
python scripts/calibrate_conformal.py --config configs/3d/C3D5_channel_swag_128.yaml --mode scaled --base swag
python scripts/calibrate_conformal.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --mode scaled --base det

python scripts/run_uncertainty_calibration.py --config configs/3d/C3D1_channel_baseline_128.yaml
python scripts/run_uncertainty_calibration.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml
python scripts/run_uncertainty_calibration.py --config configs/3d/C3D3_channel_ensemble_128.yaml
python scripts/run_uncertainty_calibration.py --config configs/3d/C3D4_channel_variational_128.yaml
python scripts/run_uncertainty_calibration.py --config configs/3d/C3D5_channel_swag_128.yaml
python scripts/run_uncertainty_calibration.py --config configs/3d/C3D6_channel_physics_informed_128.yaml

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
