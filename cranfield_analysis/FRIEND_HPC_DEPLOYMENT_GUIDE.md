# Friend's HPC Deployment Guide (n63719vm)

## Quick 15-Minute Setup

### Step 1: SSH to Friend's HPC
```bash
ssh n63719vm@csf3.itservices.manchester.ac.uk
```

### Step 2: Run Setup Script
```bash
# Download and run setup
wget https://raw.githubusercontent.com/SachinBains/turbulence-calibrated-surrogate_3d/main/cranfield_analysis/setup_friend_hpc.sh
chmod +x setup_friend_hpc.sh
./setup_friend_hpc.sh
```

### Step 3: Transfer Your Data & Results
```bash
# From your laptop (after downloading from CSF3)
scp -r artifacts_3d_backup/ n63719vm@csf3.itservices.manchester.ac.uk:/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/artifacts_3d/
scp -r data_3d_backup/ n63719vm@csf3.itservices.manchester.ac.uk:/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/data_3d/
```

### Step 4: Run Complete Analysis
```bash
# On friend's HPC
cd /mnt/iusers01/fse-ugpgt01/mace01/n63719vm/turbulence-calibrated-surrogate_3d
chmod +x cranfield_analysis/run_analysis_friend_hpc.sh
./cranfield_analysis/run_analysis_friend_hpc.sh
```

## What This Does:

1. **Clones your repo** to friend's HPC
2. **Auto-updates ALL config paths** from `p78669sb` → `n63719vm`
3. **Creates directory structure** matching your CSF3 setup
4. **Runs complete analysis pipeline** on all 6 methods (C3D1-C3D6)
5. **Generates justification** for selecting 4 methods for primary dataset

## Key Files Created:

- `update_paths_for_friend.py` - Automatically updates all config paths
- `setup_friend_hpc.sh` - Complete environment setup
- `run_analysis_friend_hpc.sh` - Full analysis pipeline execution

## Expected Output:

- **Complete smoke test analysis** for all 6 methods
- **Quantitative comparison** showing C3D4/C3D5 issues
- **Justification** for focusing on C3D1, C3D2, C3D3, C3D6
- **Results** in `/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/artifacts_3d/results/`

## Timeline:
- Setup: 5 minutes
- Data transfer: 10 minutes  
- Analysis execution: 2-3 hours
- **Total**: Ready to run in 15 minutes, complete in ~3 hours

This gives you the full 6-method analysis to justify your 4-method selection for the primary dataset.
