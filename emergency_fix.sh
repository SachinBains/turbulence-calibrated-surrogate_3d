#!/bin/bash
# EMERGENCY FIX - Resolve git conflict and get analysis working immediately

echo "=== EMERGENCY FIX FOR GIT CONFLICT ==="

# Stash any local changes
git stash push -m "emergency_stash_$(date +%s)"

# Force pull the latest changes
git reset --hard HEAD
git pull origin main

# Update paths for friend's environment
python cranfield_analysis/update_paths_for_friend.py

# Generate smoke test splits immediately
export ARTIFACTS_ROOT=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d
python scripts/fix_smoke_test_splits.py --artifacts_root $ARTIFACTS_ROOT

# Fix directory structure
python scripts/fix_ensemble_structure.py --artifacts_dir $ARTIFACTS_ROOT

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "=== EMERGENCY FIX COMPLETE - READY TO RUN ANALYSIS ==="
echo "Now run: ./scripts/run_smoke_test_analysis_fixed.sh"
