#!/bin/bash
"""
Safe git pull that preserves friend's config paths
Stashes local config changes, pulls updates, then reapplies friend's paths
"""

echo "=== Safe Git Pull for Friend's HPC ==="

# Check if we're on friend's HPC (look for n63719vm in paths)
if [[ ! "$PWD" == *"n63719vm"* ]]; then
    echo "❌ This script should only be run on friend's HPC (n63719vm)"
    exit 1
fi

# Stash any local config changes
echo "1. Stashing local config changes..."
git stash push -m "Friend's HPC config paths" configs/

# Pull latest changes from GitHub
echo "2. Pulling latest changes from GitHub..."
git pull origin main

# Check if pull was successful
if [ $? -ne 0 ]; then
    echo "❌ Git pull failed! Restoring stashed configs..."
    git stash pop
    exit 1
fi

# Reapply friend's paths
echo "3. Reapplying friend's HPC paths..."
python cranfield_analysis/update_paths_for_friend.py

# Verify paths were updated correctly
echo "4. Verifying config paths..."
grep -r "n63719vm" configs/ | head -3

echo "✅ Safe git pull complete!"
echo "Config files updated for friend's HPC (n63719vm)"
