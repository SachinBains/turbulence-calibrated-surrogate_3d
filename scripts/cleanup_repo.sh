#!/bin/bash

# Repository cleanup script - removes unnecessary files and folders
# Keeps: smoke test configs (3d/), primary_final configs, streamlined pipeline

echo "Starting repository cleanup..."

# Delete entire folders
echo "Deleting unnecessary folders..."
rm -rf cranfield/
rm -rf cranfield_analysis/
rm -rf configs/3d_primary/

# Delete root directory files
echo "Deleting root directory clutter..."
rm -f emergency_fix.sh
rm -f extract_ret1000_structured_96.py
rm -f run_analysis_fixed.py
rm -f run_complete_analysis.py
rm -f runpod_h100_setup.md
rm -f runpod_setup_guide.md
rm -f artifcats_dir_tree

# Delete old job files (keep only primary_final and streamlined)
echo "Cleaning up job directory..."
rm -f job/train_C3D1.slurm
rm -f job/train_C3D2.slurm
rm -f job/train_C3D3.slurm
rm -f job/train_C3D4.slurm
rm -f job/train_C3D5.slurm
rm -f job/train_C3D6.slurm
rm -f job/train_C3D1_primary.slurm
rm -f job/train_C3D2_primary.slurm
rm -f job/train_C3D3_primary.slurm
rm -f job/train_C3D4_primary.slurm
rm -f job/train_C3D5_primary.slurm
rm -f job/train_C3D6_primary.slurm
rm -f job/submit_all_experiments.sh
rm -f job/submit_all_primary_experiments.sh

# Delete temporary/fix scripts
echo "Cleaning up scripts directory..."
rm -f scripts/fix_all_issues.py
rm -f scripts/fix_all_issues.slurm
rm -f scripts/fix_ensemble_structure.py
rm -f scripts/fix_smoke_test_splits.py
rm -f scripts/fix_smoke_test_splits.slurm
rm -f scripts/update_smoke_test_dataset_path.py

# Delete test scripts
rm -f scripts/test_ab_split.py
rm -f scripts/test_jhtdb_integration.py

# Delete RunPod specific scripts
rm -f scripts/runpod_batch_train.py

# Delete duplicate scripts (keep versions in scripts/)
# (run_analysis_fixed.py and run_complete_analysis.py already deleted from root)

# Delete temporary/one-time use scripts
rm -f scripts/create_secondary_manifest.py
rm -f scripts/inspect_jhtdb_structure.py
rm -f scripts/validate_3d_setup.py

echo "Cleanup completed!"
echo ""
echo "DELETED:"
echo "- cranfield/ and cranfield_analysis/ folders"
echo "- configs/3d_primary/ folder (redundant)"
echo "- Root directory clutter files"
echo "- Old job files (kept only primary_final and streamlined)"
echo "- Temporary fix scripts"
echo "- Test scripts"
echo "- RunPod specific scripts"
echo "- One-time use scripts"
echo ""
echo "KEPT:"
echo "- configs/3d/ (smoke test)"
echo "- configs/3d_primary_final/ (main configs)"
echo "- configs/3d_secondary/ (domain shift)"
echo "- All streamlined pipeline files"
echo "- Core analysis and evaluation scripts"
echo "- Documentation files"
