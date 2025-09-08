# Repository Structure Reorganization Script
# Creates clean, categorized folder structure for better aesthetics and organization

Write-Host "Reorganizing repository structure for better aesthetics..." -ForegroundColor Green

# Create new organized directory structure
Write-Host "Creating organized folder structure..." -ForegroundColor Yellow

# Job directory reorganization
New-Item -ItemType Directory -Force -Path "job/training" | Out-Null
New-Item -ItemType Directory -Force -Path "job/evaluation" | Out-Null
New-Item -ItemType Directory -Force -Path "job/analysis" | Out-Null
New-Item -ItemType Directory -Force -Path "job/batch_scripts" | Out-Null

# Scripts directory reorganization  
New-Item -ItemType Directory -Force -Path "scripts/training" | Out-Null
New-Item -ItemType Directory -Force -Path "scripts/evaluation" | Out-Null
New-Item -ItemType Directory -Force -Path "scripts/analysis" | Out-Null
New-Item -ItemType Directory -Force -Path "scripts/visualization" | Out-Null
New-Item -ItemType Directory -Force -Path "scripts/data_processing" | Out-Null
New-Item -ItemType Directory -Force -Path "scripts/utilities" | Out-Null

Write-Host "Moving job files to organized folders..." -ForegroundColor Yellow

# Move training job files
Move-Item -Path "job/train_C3D*_primary_final.slurm" -Destination "job/training/" -ErrorAction SilentlyContinue

# Move evaluation job files
Move-Item -Path "job/run_streamlined_stage*.slurm" -Destination "job/evaluation/" -ErrorAction SilentlyContinue
Move-Item -Path "job/run_secondary_evaluation.slurm" -Destination "job/evaluation/" -ErrorAction SilentlyContinue

# Move analysis job files
Move-Item -Path "job/run_calibration_analysis.slurm" -Destination "job/analysis/" -ErrorAction SilentlyContinue
Move-Item -Path "job/run_smoke_test_analysis.slurm" -Destination "job/analysis/" -ErrorAction SilentlyContinue
Move-Item -Path "job/run_stage2_*.slurm" -Destination "job/analysis/" -ErrorAction SilentlyContinue

# Move batch scripts
Move-Item -Path "job/submit_*.sh" -Destination "job/batch_scripts/" -ErrorAction SilentlyContinue

Write-Host "Moving script files to organized folders..." -ForegroundColor Yellow

# Move training scripts
$trainingScripts = @(
    "scripts/run_train.py",
    "scripts/run_train_ens.py", 
    "scripts/run_train_swa.py",
    "scripts/train_multigpu.py"
)
foreach ($script in $trainingScripts) {
    if (Test-Path $script) { Move-Item -Path $script -Destination "scripts/training/" }
}

# Move evaluation scripts
$evaluationScripts = @(
    "scripts/run_eval.py",
    "scripts/run_eval2.py",
    "scripts/run_ensemble_eval.py",
    "scripts/run_streamlined_evaluation.py",
    "scripts/run_secondary_evaluation.py",
    "scripts/predict_*.py",
    "scripts/generate_baseline_predictions.py"
)
foreach ($script in $evaluationScripts) {
    if (Test-Path $script) { Move-Item -Path $script -Destination "scripts/evaluation/" }
}

# Move analysis scripts
$analysisScripts = @(
    "scripts/run_uncertainty_calibration.py",
    "scripts/calibrate_conformal.py",
    "scripts/run_adversarial_robustness.py",
    "scripts/run_cross_validation.py",
    "scripts/run_distribution_shift.py",
    "scripts/run_ensemble_diversity.py",
    "scripts/run_error_analysis.py",
    "scripts/run_gpr.py",
    "scripts/run_sindy.py",
    "scripts/run_temporal_consistency.py",
    "scripts/validate_physics.py",
    "scripts/run_q_criterion.py",
    "scripts/run_multiscale_physics.py",
    "scripts/compare_uq.py",
    "scripts/step*.py"
)
foreach ($script in $analysisScripts) {
    if (Test-Path $script) { Move-Item -Path $script -Destination "scripts/analysis/" }
}

# Move visualization scripts
$visualizationScripts = @(
    "scripts/make_figures.py",
    "scripts/make_slice_maps.py",
    "scripts/plot_calibration.py",
    "scripts/plot_sigma_error.py",
    "scripts/generate_all_calibration_plots.py",
    "scripts/generate_report.py"
)
foreach ($script in $visualizationScripts) {
    if (Test-Path $script) { Move-Item -Path $script -Destination "scripts/visualization/" }
}

# Move data processing scripts
$dataScripts = @(
    "scripts/download_*.py",
    "scripts/create_stratified_splits.py",
    "scripts/make_splits.py",
    "scripts/check_*.py"
)
foreach ($script in $dataScripts) {
    if (Test-Path $script) { Move-Item -Path $script -Destination "scripts/data_processing/" }
}

# Move utility scripts
$utilityScripts = @(
    "scripts/cleanup_repo.*",
    "scripts/quick_compress_results.sh",
    "scripts/download_essential_results.*",
    "scripts/setup_*.py",
    "scripts/package_*.py",
    "scripts/report_pack.py"
)
foreach ($script in $utilityScripts) {
    if (Test-Path $script) { Move-Item -Path $script -Destination "scripts/utilities/" }
}

# Move interpretability scripts
$interpretabilityScripts = @(
    "scripts/explain_*.py",
    "scripts/faithfulness.py"
)
foreach ($script in $interpretabilityScripts) {
    if (Test-Path $script) { Move-Item -Path $script -Destination "scripts/analysis/" }
}

Write-Host "Creating README files for each category..." -ForegroundColor Yellow

# Create README for job/training
$trainingReadme = @"
# Training Jobs

SLURM job files for model training:

- train_C3D1_primary_final.slurm - Baseline model training
- train_C3D2_primary_final.slurm - MC Dropout model training  
- train_C3D3_primary_final.slurm - Ensemble model training
- train_C3D6_primary_final.slurm - Physics-informed model training

Usage: sbatch train_C3D1_primary_final.slurm
"@
$trainingReadme | Out-File -FilePath "job/training/README.md" -Encoding UTF8

# Create README for job/evaluation
@"
# Evaluation Jobs

SLURM job files for model evaluation and prediction generation:

- `run_streamlined_stage*.slurm` - Streamlined evaluation pipeline stages
- `run_secondary_evaluation.slurm` - Domain shift evaluation (Re_τ=5200)

Usage: `sbatch run_streamlined_stage1.slurm`
"@ | Out-File -FilePath "job/evaluation/README.md" -Encoding UTF8

# Create README for job/analysis
@"
# Analysis Jobs

SLURM job files for analysis and calibration:

- `run_calibration_analysis.slurm` - Uncertainty calibration analysis
- `run_smoke_test_analysis.slurm` - Smoke test analysis
- `run_stage2_*.slurm` - Stage 2 analysis jobs

Usage: `sbatch run_calibration_analysis.slurm`
"@ | Out-File -FilePath "job/analysis/README.md" -Encoding UTF8

# Create README for job/batch_scripts
@"
# Batch Submission Scripts

Shell scripts for submitting multiple jobs:

- `submit_all_streamlined_stages.sh` - Submit complete evaluation pipeline
- `submit_all_primary_final_experiments.sh` - Submit all training jobs

Usage: `./submit_all_streamlined_stages.sh`
"@ | Out-File -FilePath "job/batch_scripts/README.md" -Encoding UTF8

# Create README for scripts/training
@"
# Training Scripts

Python scripts for model training:

- `run_train.py` - Main training script
- `run_train_ens.py` - Ensemble training
- `run_train_swa.py` - Stochastic Weight Averaging
- `train_multigpu.py` - Multi-GPU training

Usage: `python run_train.py --config configs/3d_primary_final/C3D1_channel_primary_final_1000.yaml`
"@ | Out-File -FilePath "scripts/training/README.md" -Encoding UTF8

# Create README for scripts/evaluation
@"
# Evaluation Scripts

Python scripts for model evaluation and prediction:

- `run_streamlined_evaluation.py` - Main streamlined evaluation pipeline
- `run_eval.py` - Basic model evaluation
- `run_ensemble_eval.py` - Ensemble evaluation
- `run_secondary_evaluation.py` - Domain shift evaluation
- `predict_*.py` - Prediction generation scripts

Usage: `python run_streamlined_evaluation.py --stage 1`
"@ | Out-File -FilePath "scripts/evaluation/README.md" -Encoding UTF8

# Create README for scripts/analysis
@"
# Analysis Scripts

Python scripts for uncertainty quantification and physics analysis:

- `run_uncertainty_calibration.py` - Uncertainty calibration analysis
- `calibrate_conformal.py` - Conformal prediction calibration
- `validate_physics.py` - Physics validation
- `run_q_criterion.py` - Q-criterion analysis
- `compare_uq.py` - UQ method comparison
- `step*.py` - Analysis pipeline steps

Usage: `python run_uncertainty_calibration.py --config <config_file>`
"@ | Out-File -FilePath "scripts/analysis/README.md" -Encoding UTF8

# Create README for scripts/visualization
@"
# Visualization Scripts

Python scripts for generating plots and reports:

- `make_figures.py` - Generate analysis figures
- `plot_calibration.py` - Calibration plots
- `generate_report.py` - Generate analysis reports
- `make_slice_maps.py` - Generate slice visualizations

Usage: `python make_figures.py --config <config_file>`
"@ | Out-File -FilePath "scripts/visualization/README.md" -Encoding UTF8

# Create README for scripts/data_processing
@"
# Data Processing Scripts

Python scripts for data handling and preprocessing:

- `download_*.py` - Data download scripts
- `create_stratified_splits.py` - Create dataset splits
- `check_*.py` - Data validation scripts

Usage: `python create_stratified_splits.py`
"@ | Out-File -FilePath "scripts/data_processing/README.md" -Encoding UTF8

# Create README for scripts/utilities
@"
# Utility Scripts

Helper scripts for repository management and deployment:

- `cleanup_repo.*` - Repository cleanup scripts
- `download_essential_results.*` - Download key results from CSF3
- `quick_compress_results.sh` - Compress and download results
- `package_*.py` - Packaging utilities

Usage: `./cleanup_repo.ps1` or `./download_essential_results.sh`
"@ | Out-File -FilePath "scripts/utilities/README.md" -Encoding UTF8

Write-Host "Repository reorganization completed!" -ForegroundColor Green
Write-Host ""
Write-Host "NEW STRUCTURE:" -ForegroundColor Cyan
Write-Host "job/" -ForegroundColor White
Write-Host "├── training/          # Model training SLURM jobs" -ForegroundColor Gray
Write-Host "├── evaluation/        # Evaluation and prediction jobs" -ForegroundColor Gray  
Write-Host "├── analysis/          # Analysis and calibration jobs" -ForegroundColor Gray
Write-Host "└── batch_scripts/     # Batch submission scripts" -ForegroundColor Gray
Write-Host ""
Write-Host "scripts/" -ForegroundColor White
Write-Host "├── training/          # Training scripts" -ForegroundColor Gray
Write-Host "├── evaluation/        # Evaluation and prediction scripts" -ForegroundColor Gray
Write-Host "├── analysis/          # UQ and physics analysis scripts" -ForegroundColor Gray
Write-Host "├── visualization/     # Plotting and reporting scripts" -ForegroundColor Gray
Write-Host "├── data_processing/   # Data handling scripts" -ForegroundColor Gray
Write-Host "└── utilities/         # Helper and deployment scripts" -ForegroundColor Gray
Write-Host ""
Write-Host "Each folder now contains a README.md with usage instructions!" -ForegroundColor Green
