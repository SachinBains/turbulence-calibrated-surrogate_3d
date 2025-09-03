#!/bin/bash
# Submit all C3D experiments to SLURM queue
# Usage: bash submit_all_experiments.sh

echo "Submitting all C3D experiments..."

# Submit baseline experiment
echo "Submitting C3D1 (Baseline)..."
C3D1_JOB=$(sbatch job/train_C3D1.slurm | awk '{print $4}')
echo "C3D1 Job ID: $C3D1_JOB"

# Submit MC Dropout experiment
echo "Submitting C3D2 (MC Dropout)..."
C3D2_JOB=$(sbatch job/train_C3D2.slurm | awk '{print $4}')
echo "C3D2 Job ID: $C3D2_JOB"

# Submit Ensemble experiment (depends on C3D1)
echo "Submitting C3D3 (Ensemble)..."
C3D3_JOB=$(sbatch --dependency=afterok:$C3D1_JOB job/train_C3D3.slurm | awk '{print $4}')
echo "C3D3 Job ID: $C3D3_JOB"

# Submit Variational experiment
echo "Submitting C3D4 (Variational)..."
C3D4_JOB=$(sbatch job/train_C3D4.slurm | awk '{print $4}')
echo "C3D4 Job ID: $C3D4_JOB"

# Submit SWAG experiment (depends on C3D1)
echo "Submitting C3D5 (SWAG)..."
C3D5_JOB=$(sbatch --dependency=afterok:$C3D1_JOB job/train_C3D5.slurm | awk '{print $4}')
echo "C3D5 Job ID: $C3D5_JOB"

# Submit Physics-Informed experiment
echo "Submitting C3D6 (Physics-Informed)..."
C3D6_JOB=$(sbatch job/train_C3D6.slurm | awk '{print $4}')
echo "C3D6 Job ID: $C3D6_JOB"

echo ""
echo "All experiments submitted!"
echo "Job IDs:"
echo "  C3D1 (Baseline): $C3D1_JOB"
echo "  C3D2 (MC Dropout): $C3D2_JOB"
echo "  C3D3 (Ensemble): $C3D3_JOB"
echo "  C3D4 (Variational): $C3D4_JOB"
echo "  C3D5 (SWAG): $C3D5_JOB"
echo "  C3D6 (Physics-Informed): $C3D6_JOB"
echo ""
echo "Monitor with: squeue -u p78669sb"
echo "Cancel all with: scancel -u p78669sb"
