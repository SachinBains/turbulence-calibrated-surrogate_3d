#!/bin/bash
# Setup script for friend's HPC (n63719vm) - Run this after SSH login

echo "Setting up analysis pipeline on friend's HPC (n63719vm)..."

# 1. Clone repository
echo "Cloning repository..."
cd /mnt/iusers01/fse-ugpgt01/mace01/n63719vm/
git clone https://github.com/SachinBains/turbulence-calibrated-surrogate_3d.git
cd turbulence-calibrated-surrogate_3d

# 2. Create directory structure
echo "Creating directory structure..."
mkdir -p /mnt/iusers01/fse-ugpgt01/mace01/n63719vm/artifacts_3d/{results,checkpoints,logs,cache}
mkdir -p /mnt/iusers01/fse-ugpgt01/mace01/n63719vm/data_3d

# 3. Load modules
echo "Loading modules..."
module load python/3.13.1
module load cuda/12.6.2
module load gcc/13.3.0

# 4. Install dependencies
echo "Installing Python packages..."
pip install --user pyyaml numpy torch torchvision h5py matplotlib seaborn pandas tqdm scikit-learn

# 5. Update all config paths for friend's username
echo "Updating config paths for n63719vm..."
python cranfield_analysis/update_paths_for_friend.py

# 6. Set environment variables
export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/turbulence-calibrated-surrogate_3d:$PYTHONPATH
export ARTIFACTS_ROOT=/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/artifacts_3d

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Transfer your results: scp -r artifacts_3d_backup/ n63719vm@csf3:~/artifacts_3d/"
echo "2. Transfer your data: scp -r data_3d_backup/ n63719vm@csf3:~/data_3d/"
echo "3. Run analysis pipeline with updated configs"
echo ""
echo "Environment ready for analysis!"
