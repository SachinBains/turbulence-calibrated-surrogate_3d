# Essential Results Download Script (PowerShell version)
# Downloads only key files from CSF3 results directories to local machine
# Avoids downloading large prediction arrays and focuses on analysis outputs

# Set paths
$REMOTE_USER = "p78669sb"
$REMOTE_HOST = "csf3.itservices.manchester.ac.uk"
$REMOTE_ARTIFACTS = "/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d"
$LOCAL_RESULTS = "./downloaded_results"

# Create local directory structure
New-Item -ItemType Directory -Force -Path $LOCAL_RESULTS | Out-Null

Write-Host "Downloading essential results from CSF3..." -ForegroundColor Green
Write-Host "Target models: C3D1, C3D2, C3D3, C3D6 (primary_final)" -ForegroundColor Yellow

# Essential files to download from each model
$MODELS = @("C3D1_channel_primary_final_1000", "C3D2_channel_primary_final_1000", "C3D3_channel_primary_final_1000", "C3D6_channel_primary_final_1000")

foreach ($MODEL in $MODELS) {
    Write-Host ""
    Write-Host "=== Downloading essential files for $MODEL ===" -ForegroundColor Cyan
    
    # Create model directory
    New-Item -ItemType Directory -Force -Path "$LOCAL_RESULTS/$MODEL" | Out-Null
    
    # 1. Training logs and metrics
    Write-Host "Downloading training logs..."
    scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ARTIFACTS}/logs/training/${MODEL}*" "$LOCAL_RESULTS/$MODEL/" 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host "No training logs found" -ForegroundColor Yellow }
    
    # 2. Best model checkpoint only
    Write-Host "Downloading best checkpoint..."
    scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ARTIFACTS}/checkpoints/$MODEL/best.pth" "$LOCAL_RESULTS/$MODEL/" 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host "No best checkpoint found" -ForegroundColor Yellow }
    
    # 3. Evaluation results (metrics, not raw predictions)
    Write-Host "Downloading evaluation metrics..."
    scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ARTIFACTS}/results/$MODEL/metrics/" "$LOCAL_RESULTS/$MODEL/" 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host "No metrics found" -ForegroundColor Yellow }
    
    scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ARTIFACTS}/results/$MODEL/reports/" "$LOCAL_RESULTS/$MODEL/" 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host "No reports found" -ForegroundColor Yellow }
    
    scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ARTIFACTS}/results/$MODEL/analysis/" "$LOCAL_RESULTS/$MODEL/" 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host "No analysis found" -ForegroundColor Yellow }
    
    # 4. Calibration results
    Write-Host "Downloading calibration analysis..."
    scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ARTIFACTS}/results/$MODEL/calibration/" "$LOCAL_RESULTS/$MODEL/" 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host "No calibration found" -ForegroundColor Yellow }
    
    # 5. Physics validation results
    Write-Host "Downloading physics validation..."
    scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ARTIFACTS}/results/$MODEL/physics/" "$LOCAL_RESULTS/$MODEL/" 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host "No physics validation found" -ForegroundColor Yellow }
    
    # 6. Figures and plots
    Write-Host "Downloading figures..."
    New-Item -ItemType Directory -Force -Path "$LOCAL_RESULTS/figures" | Out-Null
    scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ARTIFACTS}/figures/$MODEL/" "$LOCAL_RESULTS/figures/" 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host "No figures found" -ForegroundColor Yellow }
    
    # 7. Configuration files
    Write-Host "Downloading config..."
    scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ARTIFACTS}/results/$MODEL/config.yaml" "$LOCAL_RESULTS/$MODEL/" 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host "No config found" -ForegroundColor Yellow }
    
    # 8. Summary statistics (JSON/CSV only)
    Write-Host "Downloading summary statistics..."
    scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ARTIFACTS}/results/$MODEL/*.json" "$LOCAL_RESULTS/$MODEL/" 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host "No JSON summaries found" -ForegroundColor Yellow }
    
    scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ARTIFACTS}/results/$MODEL/*.csv" "$LOCAL_RESULTS/$MODEL/" 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host "No CSV summaries found" -ForegroundColor Yellow }
}

# Download global comparison results
Write-Host ""
Write-Host "=== Downloading global comparison results ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "$LOCAL_RESULTS/comparison" | Out-Null
scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ARTIFACTS}/results/comparison/" "$LOCAL_RESULTS/" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "No comparison results found" -ForegroundColor Yellow }

scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ARTIFACTS}/figures/comparison/" "$LOCAL_RESULTS/figures/" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "No comparison figures found" -ForegroundColor Yellow }

# Download aggregated analysis
Write-Host "Downloading aggregated analysis..."
scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ARTIFACTS}/results/summary/" "$LOCAL_RESULTS/" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "No summary results found" -ForegroundColor Yellow }

# Download calibration analysis for all models
Write-Host "Downloading calibration analysis..."
scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ARTIFACTS}/figures/calibration_primary/" "$LOCAL_RESULTS/figures/" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "No calibration figures found" -ForegroundColor Yellow }

Write-Host ""
Write-Host "=== Download Summary ===" -ForegroundColor Green
Write-Host "Essential results downloaded to: $LOCAL_RESULTS" -ForegroundColor White
Write-Host ""
Write-Host "EXCLUDED (too large):" -ForegroundColor Red
Write-Host "- Raw prediction arrays (predictions/)" -ForegroundColor Gray
Write-Host "- All model checkpoints except best.pth" -ForegroundColor Gray
Write-Host "- Intermediate training checkpoints" -ForegroundColor Gray
Write-Host "- Large HDF5 prediction files" -ForegroundColor Gray
Write-Host ""
Write-Host "INCLUDED (essential for analysis):" -ForegroundColor Green
Write-Host "- Training logs and loss curves" -ForegroundColor Gray
Write-Host "- Best model checkpoints" -ForegroundColor Gray
Write-Host "- Evaluation metrics and reports" -ForegroundColor Gray
Write-Host "- Calibration analysis and plots" -ForegroundColor Gray
Write-Host "- Physics validation results" -ForegroundColor Gray
Write-Host "- Summary statistics (JSON/CSV)" -ForegroundColor Gray
Write-Host "- All figures and visualizations" -ForegroundColor Gray
Write-Host "- Configuration files" -ForegroundColor Gray
Write-Host ""

# Calculate download size
Write-Host "Calculating download size..."
try {
    $size = (Get-ChildItem -Path $LOCAL_RESULTS -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host "Downloaded size: $([math]::Round($size, 2)) MB" -ForegroundColor Green
} catch {
    Write-Host "Size calculation unavailable" -ForegroundColor Yellow
}

Write-Host "Download complete!" -ForegroundColor Green
