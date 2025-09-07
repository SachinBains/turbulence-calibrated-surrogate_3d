#!/bin/bash

# Submit all primary final experiments (Y+ range 0-1000)
# 3D Channel Flow - Primary Final Dataset (1200 cubes at 96³ resolution)

echo "Submitting all primary final experiments..."
echo "Dataset: Channel Flow Re_τ=1000, Y+ range [0,1000]"
echo "Resolution: 96³, Total cubes: 1200"
echo ""

# C3D1: Baseline
echo "Submitting C3D1 (Baseline)..."
sbatch job/train_C3D1_primary_final.slurm
sleep 2

# C3D2: MC Dropout
echo "Submitting C3D2 (MC Dropout)..."
sbatch job/train_C3D2_primary_final.slurm
sleep 2

# C3D3: Ensemble
echo "Submitting C3D3 (Ensemble)..."
sbatch job/train_C3D3_primary_final.slurm
sleep 2

# C3D4: Variational
echo "Submitting C3D4 (Variational)..."
sbatch job/train_C3D4_primary_final.slurm
sleep 2

# C3D5: SWAG
echo "Submitting C3D5 (SWAG)..."
sbatch job/train_C3D5_primary_final.slurm
sleep 2

# C3D6: Physics-Informed
echo "Submitting C3D6 (Physics-Informed)..."
sbatch job/train_C3D6_primary_final.slurm

echo ""
echo "All primary final experiments submitted!"
echo "Monitor with: squeue -u p78669sb"
echo "Check logs in: /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/logs/training/"
