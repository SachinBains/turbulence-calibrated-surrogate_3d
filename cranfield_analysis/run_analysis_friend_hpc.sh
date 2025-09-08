#!/bin/bash
# COMPLETE SYSTEMATIC ANALYSIS PIPELINE - ALL 210 FILES
# Friend's HPC (n63719vm) - ALL PHASES 1-9 EXECUTION

export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/turbulence-calibrated-surrogate_3d:$PYTHONPATH
export ARTIFACTS_ROOT=/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/artifacts_3d

echo "Starting COMPLETE analysis pipeline - ALL 210 files generation..."

# Fix config paths first (always run after git pull)
echo "=== PHASE 0: Fix Config Paths ==="
python cranfield_analysis/update_paths_for_friend.py

# PHASE 0: Fix Ensemble Directory Structure
echo "=== PHASE 0: Fix Ensemble Structure ==="
python scripts/fix_ensemble_structure.py --artifacts_dir $ARTIFACTS_ROOT

# Verify nested directory structure
echo "=== File Verification ==="
for exp in C3D1_channel_baseline_128 C3D2_channel_mc_dropout_128 C3D3_channel_ensemble_128 C3D4_channel_variational_128 C3D5_channel_swag_128 C3D6_channel_physics_informed_128; do
    echo "Checking $exp..."
    # Check nested structure
    find $ARTIFACTS_ROOT/results/$exp -name "best_*.pth" | head -3
    find $ARTIFACTS_ROOT/results/$exp -name "*_metrics.json" | head -3
done

# PHASE 1: Generate Base Predictions (Required for all subsequent steps)
echo "=== PHASE 1: Base Predictions (GPU) ==="
python scripts/run_eval.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda
python scripts/run_eval.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda
python scripts/run_eval.py --config configs/3d/C3D4_channel_variational_128.yaml --cuda
python scripts/run_eval.py --config configs/3d/C3D5_channel_swag_128.yaml --cuda
python scripts/run_eval.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --cuda
python scripts/run_ensemble_eval.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda

# PHASE 2: Generate Method-Specific Predictions (Required for conformal calibration)
echo "=== PHASE 2: Method-Specific Predictions (GPU) ==="
python scripts/generate_baseline_predictions.py --config configs/3d/C3D1_channel_baseline_128.yaml --split val --cuda
python scripts/generate_baseline_predictions.py --config configs/3d/C3D1_channel_baseline_128.yaml --split test --cuda
python scripts/predict_mc.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --split val --cuda
python scripts/predict_mc.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --split test --cuda
python scripts/predict_ens.py --config configs/3d/C3D3_channel_ensemble_128.yaml --split val --cuda
python scripts/predict_ens.py --config configs/3d/C3D3_channel_ensemble_128.yaml --split test --cuda

# PHASE 3: Calibration and Uncertainty Analysis (Requires prediction arrays from Phase 2)
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

# PHASE 4: Robustness and Validation Analysis (Requires model + predictions)
echo "=== PHASE 4: Robustness Analysis ==="
python scripts/run_cross_validation.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda
python scripts/run_cross_validation.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda
python scripts/run_cross_validation.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda
python scripts/run_cross_validation.py --config configs/3d/C3D4_channel_variational_128.yaml --cuda
python scripts/run_cross_validation.py --config configs/3d/C3D5_channel_swag_128.yaml --cuda
python scripts/run_cross_validation.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --cuda

python scripts/run_adversarial_robustness.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda
python scripts/run_adversarial_robustness.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda
python scripts/run_adversarial_robustness.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda
python scripts/run_adversarial_robustness.py --config configs/3d/C3D4_channel_variational_128.yaml --cuda
python scripts/run_adversarial_robustness.py --config configs/3d/C3D5_channel_swag_128.yaml --cuda
python scripts/run_adversarial_robustness.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --cuda

python scripts/run_distribution_shift.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda
python scripts/run_distribution_shift.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda
python scripts/run_distribution_shift.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda
python scripts/run_distribution_shift.py --config configs/3d/C3D4_channel_variational_128.yaml --cuda
python scripts/run_distribution_shift.py --config configs/3d/C3D5_channel_swag_128.yaml --cuda
python scripts/run_distribution_shift.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --cuda

python scripts/run_ensemble_diversity.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda

python scripts/run_error_analysis.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda
python scripts/run_error_analysis.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda
python scripts/run_error_analysis.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda
python scripts/run_error_analysis.py --config configs/3d/C3D4_channel_variational_128.yaml --cuda
python scripts/run_error_analysis.py --config configs/3d/C3D5_channel_swag_128.yaml --cuda
python scripts/run_error_analysis.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --cuda

python scripts/run_temporal_consistency.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda
python scripts/run_temporal_consistency.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda
python scripts/run_temporal_consistency.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda
python scripts/run_temporal_consistency.py --config configs/3d/C3D4_channel_variational_128.yaml --cuda
python scripts/run_temporal_consistency.py --config configs/3d/C3D5_channel_swag_128.yaml --cuda
python scripts/run_temporal_consistency.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --cuda

# PHASE 5: Physics Validation (Requires prediction arrays from previous phases)
echo "=== PHASE 5: Physics Validation ==="
python scripts/validate_physics.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda
python scripts/validate_physics.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda
python scripts/validate_physics.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda
python scripts/validate_physics.py --config configs/3d/C3D4_channel_variational_128.yaml --cuda
python scripts/validate_physics.py --config configs/3d/C3D5_channel_swag_128.yaml --cuda
python scripts/validate_physics.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --cuda

python scripts/run_q_criterion.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda
python scripts/run_q_criterion.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda
python scripts/run_q_criterion.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda
python scripts/run_q_criterion.py --config configs/3d/C3D4_channel_variational_128.yaml --cuda
python scripts/run_q_criterion.py --config configs/3d/C3D5_channel_swag_128.yaml --cuda
python scripts/run_q_criterion.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --cuda

python scripts/run_multiscale_physics.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda
python scripts/run_multiscale_physics.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda
python scripts/run_multiscale_physics.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda
python scripts/run_multiscale_physics.py --config configs/3d/C3D4_channel_variational_128.yaml --cuda
python scripts/run_multiscale_physics.py --config configs/3d/C3D5_channel_swag_128.yaml --cuda
python scripts/run_multiscale_physics.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --cuda

# PHASE 6: Interpretability Analysis (Requires model + predictions from previous phases)
echo "=== PHASE 6: Interpretability Analysis ==="
python scripts/explain_global.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda
python scripts/explain_global.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda
python scripts/explain_global.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda
python scripts/explain_global.py --config configs/3d/C3D4_channel_variational_128.yaml --cuda
python scripts/explain_global.py --config configs/3d/C3D5_channel_swag_128.yaml --cuda
python scripts/explain_global.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --cuda

python scripts/explain_local.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda
python scripts/explain_local.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda
python scripts/explain_local.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda
python scripts/explain_local.py --config configs/3d/C3D4_channel_variational_128.yaml --cuda
python scripts/explain_local.py --config configs/3d/C3D5_channel_swag_128.yaml --cuda
python scripts/explain_local.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --cuda

python scripts/explain_uncertainty.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda
python scripts/explain_uncertainty.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda
python scripts/explain_uncertainty.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda
python scripts/explain_uncertainty.py --config configs/3d/C3D4_channel_variational_128.yaml --cuda
python scripts/explain_uncertainty.py --config configs/3d/C3D5_channel_swag_128.yaml --cuda
python scripts/explain_uncertainty.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --cuda

python scripts/faithfulness.py --config configs/3d/C3D1_channel_baseline_128.yaml --cuda
python scripts/faithfulness.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml --cuda
python scripts/faithfulness.py --config configs/3d/C3D3_channel_ensemble_128.yaml --cuda
python scripts/faithfulness.py --config configs/3d/C3D4_channel_variational_128.yaml --cuda
python scripts/faithfulness.py --config configs/3d/C3D5_channel_swag_128.yaml --cuda
python scripts/faithfulness.py --config configs/3d/C3D6_channel_physics_informed_128.yaml --cuda

# PHASE 7: Visualization and Reporting (Requires all analysis outputs from Phases 1-6)
echo "=== PHASE 7: Visualization and Reporting ==="
python scripts/make_slice_maps.py --config configs/3d/C3D1_channel_baseline_128.yaml
python scripts/make_slice_maps.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml
python scripts/make_slice_maps.py --config configs/3d/C3D3_channel_ensemble_128.yaml
python scripts/make_slice_maps.py --config configs/3d/C3D4_channel_variational_128.yaml
python scripts/make_slice_maps.py --config configs/3d/C3D5_channel_swag_128.yaml
python scripts/make_slice_maps.py --config configs/3d/C3D6_channel_physics_informed_128.yaml

python scripts/make_figures.py --config configs/3d/C3D1_channel_baseline_128.yaml
python scripts/make_figures.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml
python scripts/make_figures.py --config configs/3d/C3D3_channel_ensemble_128.yaml
python scripts/make_figures.py --config configs/3d/C3D4_channel_variational_128.yaml
python scripts/make_figures.py --config configs/3d/C3D5_channel_swag_128.yaml
python scripts/make_figures.py --config configs/3d/C3D6_channel_physics_informed_128.yaml

python scripts/plot_calibration.py --config configs/3d/C3D1_channel_baseline_128.yaml
python scripts/plot_calibration.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml
python scripts/plot_calibration.py --config configs/3d/C3D3_channel_ensemble_128.yaml
python scripts/plot_calibration.py --config configs/3d/C3D4_channel_variational_128.yaml
python scripts/plot_calibration.py --config configs/3d/C3D5_channel_swag_128.yaml
python scripts/plot_calibration.py --config configs/3d/C3D6_channel_physics_informed_128.yaml

python scripts/plot_sigma_error.py --config configs/3d/C3D1_channel_baseline_128.yaml
python scripts/plot_sigma_error.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml
python scripts/plot_sigma_error.py --config configs/3d/C3D3_channel_ensemble_128.yaml
python scripts/plot_sigma_error.py --config configs/3d/C3D4_channel_variational_128.yaml
python scripts/plot_sigma_error.py --config configs/3d/C3D5_channel_swag_128.yaml
python scripts/plot_sigma_error.py --config configs/3d/C3D6_channel_physics_informed_128.yaml

python scripts/generate_report.py --config configs/3d/C3D1_channel_baseline_128.yaml
python scripts/generate_report.py --config configs/3d/C3D2_channel_mc_dropout_128.yaml
python scripts/generate_report.py --config configs/3d/C3D3_channel_ensemble_128.yaml
python scripts/generate_report.py --config configs/3d/C3D4_channel_variational_128.yaml
python scripts/generate_report.py --config configs/3d/C3D5_channel_swag_128.yaml
python scripts/generate_report.py --config configs/3d/C3D6_channel_physics_informed_128.yaml

# PHASE 6: Global Aggregation Analysis (Requires ALL per-case analyses C3D1-C3D6 to be complete)
echo "=== PHASE 6: Global Aggregation Analysis ==="
python scripts/compare_uq.py --results_dir $ARTIFACTS_ROOT/results
python scripts/step9_aggregate_results.py --results_dir $ARTIFACTS_ROOT/results
python scripts/step10_error_uncertainty_maps.py --artifacts_dir $ARTIFACTS_ROOT
python scripts/step11_quantitative_comparison.py --artifacts_dir $ARTIFACTS_ROOT
python scripts/step12_physics_validation.py --artifacts_dir $ARTIFACTS_ROOT
python scripts/step13_interpretability_analysis.py --artifacts_dir $ARTIFACTS_ROOT

# PHASE 7: Final Reporting and Packaging (Requires Phase 6 completion)
echo "=== PHASE 7: Final Reporting and Packaging ==="
python scripts/step14_summary_report.py --artifacts_dir $ARTIFACTS_ROOT
python scripts/run_complete_analysis.py --artifacts_dir $ARTIFACTS_ROOT
python scripts/report_pack.py --artifacts_dir $ARTIFACTS_ROOT
python scripts/step15_backup_reproducibility.py --artifacts_dir $ARTIFACTS_ROOT
python scripts/step16_dissertation_writeup.py --artifacts_dir $ARTIFACTS_ROOT
python scripts/package_minimal.py --artifacts_dir $ARTIFACTS_ROOT

echo "✅ COMPLETE ANALYSIS PIPELINE FINISHED!"
echo "Generated ALL 210 files across 6 UQ methods"
echo "Results available in: $ARTIFACTS_ROOT/results/"
echo "Final reports in: $ARTIFACTS_ROOT/reports/"
echo "Dissertation materials in: $ARTIFACTS_ROOT/dissertation/"
