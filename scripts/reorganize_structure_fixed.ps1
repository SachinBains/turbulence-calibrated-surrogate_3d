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
Get-ChildItem -Path "job/train_C3D*_primary_final.slurm" -ErrorAction SilentlyContinue | Move-Item -Destination "job/training/"

# Move evaluation job files
Get-ChildItem -Path "job/run_streamlined_stage*.slurm" -ErrorAction SilentlyContinue | Move-Item -Destination "job/evaluation/"
if (Test-Path "job/run_secondary_evaluation.slurm") { Move-Item -Path "job/run_secondary_evaluation.slurm" -Destination "job/evaluation/" }

# Move analysis job files
if (Test-Path "job/run_calibration_analysis.slurm") { Move-Item -Path "job/run_calibration_analysis.slurm" -Destination "job/analysis/" }
if (Test-Path "job/run_smoke_test_analysis.slurm") { Move-Item -Path "job/run_smoke_test_analysis.slurm" -Destination "job/analysis/" }
Get-ChildItem -Path "job/run_stage2_*.slurm" -ErrorAction SilentlyContinue | Move-Item -Destination "job/analysis/"

# Move batch scripts
Get-ChildItem -Path "job/submit_*.sh" -ErrorAction SilentlyContinue | Move-Item -Destination "job/batch_scripts/"

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
    "scripts/run_secondary_evaluation.py"
)
foreach ($script in $evaluationScripts) {
    if (Test-Path $script) { Move-Item -Path $script -Destination "scripts/evaluation/" }
}

# Move prediction scripts
Get-ChildItem -Path "scripts/predict_*.py" -ErrorAction SilentlyContinue | Move-Item -Destination "scripts/evaluation/"
if (Test-Path "scripts/generate_baseline_predictions.py") { Move-Item -Path "scripts/generate_baseline_predictions.py" -Destination "scripts/evaluation/" }

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
    "scripts/compare_uq.py"
)
foreach ($script in $analysisScripts) {
    if (Test-Path $script) { Move-Item -Path $script -Destination "scripts/analysis/" }
}

# Move step analysis scripts
Get-ChildItem -Path "scripts/step*.py" -ErrorAction SilentlyContinue | Move-Item -Destination "scripts/analysis/"

# Move interpretability scripts
$interpretabilityScripts = @(
    "scripts/explain_global.py",
    "scripts/explain_local.py",
    "scripts/explain_uncertainty.py",
    "scripts/faithfulness.py"
)
foreach ($script in $interpretabilityScripts) {
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
Get-ChildItem -Path "scripts/download_*.py" -ErrorAction SilentlyContinue | Move-Item -Destination "scripts/data_processing/"
$dataScripts = @(
    "scripts/create_stratified_splits.py",
    "scripts/make_splits.py",
    "scripts/check_h5.py",
    "scripts/check_loader.py"
)
foreach ($script in $dataScripts) {
    if (Test-Path $script) { Move-Item -Path $script -Destination "scripts/data_processing/" }
}

# Move utility scripts
Get-ChildItem -Path "scripts/cleanup_repo.*" -ErrorAction SilentlyContinue | Move-Item -Destination "scripts/utilities/"
Get-ChildItem -Path "scripts/download_essential_results.*" -ErrorAction SilentlyContinue | Move-Item -Destination "scripts/utilities/"
$utilityScripts = @(
    "scripts/quick_compress_results.sh",
    "scripts/package_minimal.py",
    "scripts/report_pack.py"
)
foreach ($script in $utilityScripts) {
    if (Test-Path $script) { Move-Item -Path $script -Destination "scripts/utilities/" }
}

# Move remaining shell scripts
Get-ChildItem -Path "scripts/*.sh" -ErrorAction SilentlyContinue | Move-Item -Destination "scripts/utilities/"

Write-Host "Creating README files for each category..." -ForegroundColor Yellow

# Create README files
"# Training Jobs`n`nSLURM job files for model training" | Out-File -FilePath "job/training/README.md" -Encoding UTF8
"# Evaluation Jobs`n`nSLURM job files for model evaluation and prediction generation" | Out-File -FilePath "job/evaluation/README.md" -Encoding UTF8
"# Analysis Jobs`n`nSLURM job files for analysis and calibration" | Out-File -FilePath "job/analysis/README.md" -Encoding UTF8
"# Batch Scripts`n`nShell scripts for submitting multiple jobs" | Out-File -FilePath "job/batch_scripts/README.md" -Encoding UTF8

"# Training Scripts`n`nPython scripts for model training" | Out-File -FilePath "scripts/training/README.md" -Encoding UTF8
"# Evaluation Scripts`n`nPython scripts for model evaluation and prediction" | Out-File -FilePath "scripts/evaluation/README.md" -Encoding UTF8
"# Analysis Scripts`n`nPython scripts for uncertainty quantification and physics analysis" | Out-File -FilePath "scripts/analysis/README.md" -Encoding UTF8
"# Visualization Scripts`n`nPython scripts for generating plots and reports" | Out-File -FilePath "scripts/visualization/README.md" -Encoding UTF8
"# Data Processing Scripts`n`nPython scripts for data handling and preprocessing" | Out-File -FilePath "scripts/data_processing/README.md" -Encoding UTF8
"# Utility Scripts`n`nHelper scripts for repository management and deployment" | Out-File -FilePath "scripts/utilities/README.md" -Encoding UTF8

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
