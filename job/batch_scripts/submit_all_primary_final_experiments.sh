#!/bin/bash

# Submit primary final experiments optimized for 2-GPU allocation
# Submits 2 jobs at a time to maximize GPU utilization

echo "Submitting experiments optimized for 2-GPU allocation..."
echo "======================================================"

# Batch 1: Submit first 2 jobs (will use both GPUs)
echo "Batch 1: Submitting C3D1 (Baseline) and C3D2 (MC Dropout)..."
JOB1=$(sbatch job/train_C3D1_primary_final.slurm | awk '{print $4}')
JOB2=$(sbatch job/train_C3D2_primary_final.slurm | awk '{print $4}')
echo "  C3D1 Job ID: $JOB1"
echo "  C3D2 Job ID: $JOB2"

# Batch 2: Submit next 2 jobs (dependent on first batch completion)
echo "Batch 2: Submitting C3D3 (Ensemble) and C3D4 (Variational)..."
JOB3=$(sbatch --dependency=afterany:$JOB1 job/train_C3D3_primary_final.slurm | awk '{print $4}')
JOB4=$(sbatch --dependency=afterany:$JOB2 job/train_C3D4_primary_final.slurm | awk '{print $4}')
echo "  C3D3 Job ID: $JOB3 (starts after $JOB1)"
echo "  C3D4 Job ID: $JOB4 (starts after $JOB2)"

# Batch 3: Submit final 2 jobs (dependent on second batch completion)
echo "Batch 3: Submitting C3D5 (SWAG) and C3D6 (Physics-Informed)..."
JOB5=$(sbatch --dependency=afterany:$JOB3 job/train_C3D5_primary_final.slurm | awk '{print $4}')
JOB6=$(sbatch --dependency=afterany:$JOB4 job/train_C3D6_primary_final.slurm | awk '{print $4}')
echo "  C3D5 Job ID: $JOB5 (starts after $JOB3)"
echo "  C3D6 Job ID: $JOB6 (starts after $JOB4)"

echo "======================================================"
echo "All experiments queued with optimal 2-GPU utilization!"
echo "Pipeline: [C3D1,C3D2] → [C3D3,C3D4] → [C3D5,C3D6]"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check logs in: /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/logs/training/"
