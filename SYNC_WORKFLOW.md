# GitHub Sync Workflow for TCS_3D

## Overview
This document outlines the sync workflow between local development, GitHub, and CSF3 execution environments.

## Directory Structure
```
Local: c:\Users\Sachi\OneDrive\Desktop\Dissertation\
├── turbulence-calibrated-surrogate_3d/  # Main repo (synced via GitHub)
└── artifacts_3d/                        # Output artifacts (manual sync)

CSF3: /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/
├── turbulence-calibrated-surrogate_3d/  # Main repo (git pull)
└── artifacts_3d/                        # Output artifacts (rsync/scp)
```

## Sync Workflow

### 1. Local → GitHub → CSF3 (Code Updates)
```bash
# Local (Windows)
cd "c:\Users\Sachi\OneDrive\Desktop\Dissertation\turbulence-calibrated-surrogate_3d"
git add .
git commit -m "Update configs and scripts for 3D experiments"
git push origin main

# CSF3 (Linux)
cd /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_3d
git pull origin main
```

### 2. CSF3 → Local (Results Sync)
```bash
# From local machine, sync artifacts from CSF3
scp -r p78669sb@csf3.itservices.manchester.ac.uk:/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/ "c:\Users\Sachi\OneDrive\Desktop\Dissertation\"
```

### 3. Local → CSF3 (Artifacts Upload)
```bash
# If you need to upload local artifacts to CSF3
scp -r "c:\Users\Sachi\OneDrive\Desktop\Dissertation\artifacts_3d\" p78669sb@csf3.itservices.manchester.ac.uk:/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/
```

## Pre-Deployment Checklist

### Before Pushing to GitHub:
- [ ] All configs use correct CSF3 paths (`/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/`)
- [ ] All scripts use ChannelDataset (not HITDataset)
- [ ] SLURM scripts have correct modules and partition
- [ ] Experiment manifest is up to date

### After Pulling on CSF3:
- [ ] Verify data directory exists: `/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/data_3d/channel_flow_smoke/`
- [ ] Check artifacts directory permissions: `/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/`
- [ ] Activate virtual environment: `source /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/venvs/turbml/bin/activate`
- [ ] Test data loading: `python scripts/check_loader.py --config configs/3d/C3D1_channel_baseline_128.yaml`

## Key Commands

### Git Status Check
```bash
git status
git log --oneline -5
```

### Quick Test
```bash
# Test config loading
python -c "import yaml; print(yaml.safe_load(open('configs/3d/C3D1_channel_baseline_128.yaml'))['dataset']['data_dir'])"

# Test data loading
python scripts/check_loader.py --config configs/3d/C3D1_channel_baseline_128.yaml --split train
```

### Job Submission
```bash
# Submit all experiments
bash job/submit_all_experiments.sh

# Monitor jobs
squeue -u p78669sb
```
