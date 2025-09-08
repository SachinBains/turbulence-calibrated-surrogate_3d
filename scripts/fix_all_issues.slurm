#!/bin/bash
#SBATCH --job-name=fix_all_issues
#SBATCH --output=logs/fix_all_%j.log
#SBATCH --error=logs/fix_all_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G

# Load required modules
module load python/3.8.5

# Set environment variables
export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_3d:$PYTHONPATH
export ARTIFACTS_ROOT=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install numpy h5py scikit-learn
else
    source venv/bin/activate
fi

echo "Running comprehensive fix for all HPC issues..."

# Run the comprehensive fix
python scripts/fix_all_issues.py --artifacts_root $ARTIFACTS_ROOT

echo "Fix complete. Check output above for next steps."
deactivate
