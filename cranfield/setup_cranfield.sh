#!/bin/bash
# Cranfield HPC Setup Script for C3D4 Variational Training

echo "Setting up C3D4 Variational Training on Cranfield HPC..."

# 1. Create directory structure
echo "Creating directory structure..."
mkdir -p /home/$USER/data/channel_flow_1000
mkdir -p /home/$USER/artifacts_3d/{results,checkpoints,logs,cache}
mkdir -p /home/$USER/turbulence-calibrated-surrogate_3d/logs

# 2. Extract data (assumes data package already transferred)
echo "Extracting data package..."
if [ -f "/home/$USER/cranfield_c3d4_data.tar.gz" ]; then
    cd /home/$USER
    tar -xzf cranfield_c3d4_data.tar.gz
    mv jhtdb_96cubed_production_structured /home/$USER/data/channel_flow_1000/
    echo "Data extracted successfully"
else
    echo "WARNING: Data package not found at /home/$USER/cranfield_c3d4_data.tar.gz"
    echo "Please transfer the data package first"
fi

# 3. Clone repository (if not already done)
if [ ! -d "/home/$USER/turbulence-calibrated-surrogate_3d" ]; then
    echo "Cloning repository..."
    cd /home/$USER
    git clone https://github.com/SachinBains/turbulence-calibrated-surrogate_3d.git
else
    echo "Repository already exists, pulling latest changes..."
    cd /home/$USER/turbulence-calibrated-surrogate_3d
    git pull origin main
fi

# 4. Create virtual environment
echo "Setting up Python environment..."
python3 -m venv /home/$USER/venv
source /home/$USER/venv/bin/activate

# 5. Install dependencies
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r cranfield/requirements.txt

# 6. Update config with actual username
echo "Updating config with username: $USER"
sed -i "s/USERNAME/$USER/g" cranfield/C3D4_cranfield.yaml
sed -i "s/USERNAME/$USER/g" cranfield/train_C3D4_cranfield.slurm

# 7. Set permissions
chmod +x cranfield/train_C3D4_cranfield.slurm

echo "Setup complete!"
echo ""
echo "To run training:"
echo "1. sbatch cranfield/train_C3D4_cranfield.slurm"
echo "2. Monitor with: squeue -u $USER"
echo "3. Check logs in: logs/C3D4_cranfield_*.out"
echo ""
echo "Data location: /home/$USER/data/channel_flow_1000/jhtdb_96cubed_production_structured/"
echo "Results will be saved to: /home/$USER/artifacts_3d/results/C3D4_channel_primary_final_1000/"
