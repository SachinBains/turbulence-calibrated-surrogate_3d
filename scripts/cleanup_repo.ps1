# Repository cleanup script (PowerShell version)
# Removes unnecessary files and folders
# Keeps: smoke test configs (3d/), primary_final configs, streamlined pipeline

Write-Host "Starting repository cleanup..." -ForegroundColor Green

# Delete entire folders
Write-Host "Deleting unnecessary folders..." -ForegroundColor Yellow
Remove-Item -Recurse -Force -Path "cranfield" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force -Path "cranfield_analysis" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force -Path "configs/3d_primary" -ErrorAction SilentlyContinue

# Delete root directory files
Write-Host "Deleting root directory clutter..." -ForegroundColor Yellow
Remove-Item -Force -Path "emergency_fix.sh" -ErrorAction SilentlyContinue
Remove-Item -Force -Path "extract_ret1000_structured_96.py" -ErrorAction SilentlyContinue
Remove-Item -Force -Path "run_analysis_fixed.py" -ErrorAction SilentlyContinue
Remove-Item -Force -Path "run_complete_analysis.py" -ErrorAction SilentlyContinue
Remove-Item -Force -Path "runpod_h100_setup.md" -ErrorAction SilentlyContinue
Remove-Item -Force -Path "runpod_setup_guide.md" -ErrorAction SilentlyContinue
Remove-Item -Force -Path "artifcats_dir_tree" -ErrorAction SilentlyContinue

# Delete old job files (keep only primary_final and streamlined)
Write-Host "Cleaning up job directory..." -ForegroundColor Yellow
$oldJobFiles = @(
    "job/train_C3D1.slurm",
    "job/train_C3D2.slurm",
    "job/train_C3D3.slurm",
    "job/train_C3D4.slurm",
    "job/train_C3D5.slurm",
    "job/train_C3D6.slurm",
    "job/train_C3D1_primary.slurm",
    "job/train_C3D2_primary.slurm",
    "job/train_C3D3_primary.slurm",
    "job/train_C3D4_primary.slurm",
    "job/train_C3D5_primary.slurm",
    "job/train_C3D6_primary.slurm",
    "job/submit_all_experiments.sh",
    "job/submit_all_primary_experiments.sh"
)

foreach ($file in $oldJobFiles) {
    Remove-Item -Force -Path $file -ErrorAction SilentlyContinue
}

# Delete temporary/fix scripts
Write-Host "Cleaning up scripts directory..." -ForegroundColor Yellow
$tempScripts = @(
    "scripts/fix_all_issues.py",
    "scripts/fix_all_issues.slurm",
    "scripts/fix_ensemble_structure.py",
    "scripts/fix_smoke_test_splits.py",
    "scripts/fix_smoke_test_splits.slurm",
    "scripts/update_smoke_test_dataset_path.py",
    "scripts/test_ab_split.py",
    "scripts/test_jhtdb_integration.py",
    "scripts/runpod_batch_train.py",
    "scripts/create_secondary_manifest.py",
    "scripts/inspect_jhtdb_structure.py",
    "scripts/validate_3d_setup.py"
)

foreach ($script in $tempScripts) {
    Remove-Item -Force -Path $script -ErrorAction SilentlyContinue
}

Write-Host "Cleanup completed!" -ForegroundColor Green
Write-Host ""
Write-Host "DELETED:" -ForegroundColor Red
Write-Host "- cranfield/ and cranfield_analysis/ folders" -ForegroundColor Gray
Write-Host "- configs/3d_primary/ folder (redundant)" -ForegroundColor Gray
Write-Host "- Root directory clutter files" -ForegroundColor Gray
Write-Host "- Old job files (kept only primary_final and streamlined)" -ForegroundColor Gray
Write-Host "- Temporary fix scripts" -ForegroundColor Gray
Write-Host "- Test scripts" -ForegroundColor Gray
Write-Host "- RunPod specific scripts" -ForegroundColor Gray
Write-Host "- One-time use scripts" -ForegroundColor Gray
Write-Host ""
Write-Host "KEPT:" -ForegroundColor Green
Write-Host "- configs/3d/ (smoke test)" -ForegroundColor Gray
Write-Host "- configs/3d_primary_final/ (main configs)" -ForegroundColor Gray
Write-Host "- configs/3d_secondary/ (domain shift)" -ForegroundColor Gray
Write-Host "- All streamlined pipeline files" -ForegroundColor Gray
Write-Host "- Core analysis and evaluation scripts" -ForegroundColor Gray
Write-Host "- Documentation files" -ForegroundColor Gray
