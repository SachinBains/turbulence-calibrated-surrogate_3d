#!/bin/bash
#SBATCH --job-name=fix_smoke_splits
#SBATCH --output=logs/fix_smoke_splits_%j.log
#SBATCH --error=logs/fix_smoke_splits_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G

# Load required modules
module load python/3.8.5

# Set environment variables
export ARTIFACTS_ROOT=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install numpy
else
    source venv/bin/activate
fi

echo "Generating splits for smoke test dataset..."
echo "ARTIFACTS_ROOT: $ARTIFACTS_ROOT"

# Run the smoke test split generation
python scripts/fix_smoke_test_splits.py --artifacts_root $ARTIFACTS_ROOT

echo "Smoke test splits generation complete."
deactivate
