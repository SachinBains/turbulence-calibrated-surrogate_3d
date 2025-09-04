#!/bin/bash
# Submit all C3D Primary Dataset experiments to SLURM queue
# Usage: bash submit_all_primary_experiments.sh

echo "Submitting all C3D Primary Dataset experiments (1600 cubes, 96³)..."

# Submit baseline experiment
echo "Submitting C3D1 Primary (Baseline)..."
C3D1_PRIMARY_JOB=$(sbatch job/train_C3D1_primary.slurm | awk '{print $4}')
echo "C3D1 Primary Job ID: $C3D1_PRIMARY_JOB"

# Submit MC Dropout experiment
echo "Submitting C3D2 Primary (MC Dropout)..."
C3D2_PRIMARY_JOB=$(sbatch job/train_C3D2_primary.slurm | awk '{print $4}')
echo "C3D2 Primary Job ID: $C3D2_PRIMARY_JOB"

# Submit Ensemble experiment (depends on C3D1)
echo "Submitting C3D3 Primary (Ensemble)..."
C3D3_PRIMARY_JOB=$(sbatch --dependency=afterok:$C3D1_PRIMARY_JOB job/train_C3D3_primary.slurm | awk '{print $4}')
echo "C3D3 Primary Job ID: $C3D3_PRIMARY_JOB"

# Submit Variational experiment
echo "Submitting C3D4 Primary (Variational)..."
C3D4_PRIMARY_JOB=$(sbatch job/train_C3D4_primary.slurm | awk '{print $4}')
echo "C3D4 Primary Job ID: $C3D4_PRIMARY_JOB"

# Submit SWAG experiment (depends on C3D1)
echo "Submitting C3D5 Primary (SWAG)..."
C3D5_PRIMARY_JOB=$(sbatch --dependency=afterok:$C3D1_PRIMARY_JOB job/train_C3D5_primary.slurm | awk '{print $4}')
echo "C3D5 Primary Job ID: $C3D5_PRIMARY_JOB"

# Submit Physics-Informed experiment
echo "Submitting C3D6 Primary (Physics-Informed)..."
C3D6_PRIMARY_JOB=$(sbatch job/train_C3D6_primary.slurm | awk '{print $4}')
echo "C3D6 Primary Job ID: $C3D6_PRIMARY_JOB"

echo ""
echo "All primary dataset experiments submitted!"
echo "Job IDs:"
echo "  C3D1 Primary (Baseline): $C3D1_PRIMARY_JOB"
echo "  C3D2 Primary (MC Dropout): $C3D2_PRIMARY_JOB"
echo "  C3D3 Primary (Ensemble): $C3D3_PRIMARY_JOB (depends on C3D1)"
echo "  C3D4 Primary (Variational): $C3D4_PRIMARY_JOB"
echo "  C3D5 Primary (SWAG): $C3D5_PRIMARY_JOB (depends on C3D1)"
echo "  C3D6 Primary (Physics-Informed): $C3D6_PRIMARY_JOB"
echo ""
echo "Monitor with: squeue -u $USER"
