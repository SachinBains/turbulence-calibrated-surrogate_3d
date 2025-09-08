#!/bin/bash
# COMPLETE ANALYSIS PIPELINE - ALL PHASES FOR DISSERTATION RESULTS
# Friend's HPC (n63719vm) - Generate ALL analysis outputs

export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/turbulence-calibrated-surrogate_3d:$PYTHONPATH
export ARTIFACTS_ROOT=/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/artifacts_3d

echo "Starting COMPLETE analysis pipeline for dissertation results..."

# PHASE 0: Setup and Fixes
echo "=== PHASE 0: Setup ==="
python cranfield_analysis/update_paths_for_friend.py
python scripts/fix_ensemble_structure.py --artifacts_dir $ARTIFACTS_ROOT

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# PHASE 1: Base Model Evaluations (Generate predictions and metrics)
echo "=== PHASE 1: Base Model Evaluations ==="
python scripts/run_eval.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda
python scripts/run_eval.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda
python scripts/run_ensemble_eval.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda
python scripts/run_eval.py --config configs/3d/C3D4_channel_variational_128.yaml --cuda
python scripts/run_eval.py --config configs/3d/C3D5_channel_swag_128.yaml --cuda
python scripts/run_eval.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --cuda

# PHASE 2: Method-Specific Predictions (For uncertainty analysis)
echo "=== PHASE 2: Method-Specific Predictions ==="
python scripts/generate_baseline_predictions.py --config configs/3d/C3D1_channel_baseline_128.yaml --split val --cuda
python scripts/generate_baseline_predictions.py --config configs/3d/C3D1_channel_baseline_128.yaml --split test --cuda
python scripts/predict_mc.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --split val --cuda
python scripts/predict_mc.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --split test --cuda
python scripts/predict_ens.py --config configs/3d/C3D3_channel_ensemble_128.yaml --split val --cuda
python scripts/predict_ens.py --config configs/3d/C3D3_channel_ensemble_128.yaml --split test --cuda

# PHASE 3: Conformal Calibration
echo "=== PHASE 3: Conformal Calibration ==="
python scripts/calibrate_conformal.py --config configs/3d/C3D1_channel_baseline_128.yaml --mode scaled --base det --cuda
python scripts/calibrate_conformal.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --mode scaled --base mc --cuda
python scripts/calibrate_conformal.py --config configs/3d/C3D3_channel_ensemble_128.yaml --mode scaled --base ens --cuda

# PHASE 4: Physics Validation
echo "=== PHASE 4: Physics Validation ==="
python scripts/validate_physics.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda
python scripts/validate_physics.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda
python scripts/validate_physics.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda

# PHASE 5: Band-wise Analysis (Y+ stratified evaluation)
echo "=== PHASE 5: Band-wise Analysis ==="
python scripts/band_evaluation.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda
python scripts/band_evaluation.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda
python scripts/band_evaluation.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda

# PHASE 6: Calibration Analysis
echo "=== PHASE 6: Calibration Analysis ==="
python scripts/analyze_calibration.py --config configs/3d/C3D1_channel_baseline_128.yaml
python scripts/analyze_calibration.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml
python scripts/analyze_calibration.py --config configs/3d/C3D3_channel_ensemble_128.yaml

# PHASE 7: Adversarial Robustness
echo "=== PHASE 7: Adversarial Robustness ==="
python scripts/adversarial_robustness.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda
python scripts/adversarial_robustness.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda
python scripts/adversarial_robustness.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda

# PHASE 8: Generate Reports
echo "=== PHASE 8: Generate Reports ==="
python scripts/generate_report.py --config configs/3d/C3D1_channel_baseline_128.yaml
python scripts/generate_report.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml
python scripts/generate_report.py --config configs/3d/C3D3_channel_ensemble_128.yaml

# PHASE 9: Aggregate Results
echo "=== PHASE 9: Aggregate Results ==="
python scripts/step9_aggregate_results.py

echo "=== ANALYSIS COMPLETE ==="
echo "Results saved to: $ARTIFACTS_ROOT/results/"
echo "Reports saved to: $ARTIFACTS_ROOT/figures/"

# Count output files
echo "=== OUTPUT SUMMARY ==="
total_files=$(find $ARTIFACTS_ROOT/results -name "*.json" -o -name "*.npy" -o -name "*.h5" | wc -l)
echo "Total output files: $total_files"

for exp in C3D1_channel_baseline_128 C3D2_channel_mc_dropout_128 C3D3_channel_ensemble_128 C3D4_channel_variational_128 C3D5_channel_swag_128 C3D6_channel_physics_informed_128; do
    if [ -d "$ARTIFACTS_ROOT/results/$exp" ]; then
        count=$(find $ARTIFACTS_ROOT/results/$exp -name "*.json" -o -name "*.npy" -o -name "*.h5" | wc -l)
        echo "$exp: $count files"
    fi
done
