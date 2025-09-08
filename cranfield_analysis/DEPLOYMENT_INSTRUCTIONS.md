# Friend's HPC Deployment Instructions

## Problem: Git Pull vs Config Paths
The `update_paths_for_friend.py` script modifies config files locally, but `git pull` overwrites these changes with original paths from GitHub.

## Solution Options

### Option 1: Safe Git Pull (Recommended)
Use the provided safe pull script that preserves config changes:

```bash
# Instead of: git pull origin main
# Use this:
chmod +x cranfield_analysis/safe_git_pull.sh
./cranfield_analysis/safe_git_pull.sh
```

This script:
1. Stashes local config changes
2. Pulls from GitHub  
3. Reapplies friend's paths automatically

### Option 2: Manual Process
If you prefer manual control:

```bash
# 1. Stash config changes before pull
git stash push -m "Friend's config paths" configs/

# 2. Pull latest changes
git pull origin main

# 3. Reapply friend's paths
python cranfield_analysis/update_paths_for_friend.py
```

### Option 3: Analysis Script Handles It
The `run_analysis_friend_hpc.sh` script automatically runs `update_paths_for_friend.py` at the start, so even if git pull overwrites configs, they get fixed before analysis begins.

## Recommended Workflow

```bash
# 1. Safe pull (preserves configs)
./cranfield_analysis/safe_git_pull.sh

# 2. Run complete analysis
chmod +x cranfield_analysis/run_analysis_friend_hpc.sh
./cranfield_analysis/run_analysis_friend_hpc.sh
```

## Files That Get Modified
- `configs/3d/*.yaml`
- `configs/3d_primary_final/*.yaml` 
- Any config with paths containing `p78669sb` → `n63719vm`

## Verification
Check that paths were updated correctly:
```bash
grep -r "n63719vm" configs/ | head -5
```

Should show friend's username in all config paths.
