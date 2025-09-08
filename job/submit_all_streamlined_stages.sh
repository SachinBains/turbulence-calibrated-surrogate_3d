#!/bin/bash

# Submit all streamlined evaluation stages for C3D1, C3D2, C3D3, C3D6
# Each stage has 20-minute time limit and proper partition configuration

echo "Submitting streamlined evaluation pipeline..."
echo "Models: C3D1, C3D2, C3D3, C3D6"
echo "Time limit: 20 minutes per stage"

# Submit stages sequentially with dependencies
echo "Submitting Stage 1: Core Evaluation"
STAGE1_JOB=$(sbatch --parsable job/run_streamlined_stage1.slurm)
echo "Stage 1 job ID: $STAGE1_JOB"

echo "Submitting Stage 2: Method-specific Predictions (depends on Stage 1)"
STAGE2_JOB=$(sbatch --parsable --dependency=afterok:$STAGE1_JOB job/run_streamlined_stage2.slurm)
echo "Stage 2 job ID: $STAGE2_JOB"

echo "Submitting Stage 3: Conformal Calibration (depends on Stage 2)"
STAGE3_JOB=$(sbatch --parsable --dependency=afterok:$STAGE2_JOB job/run_streamlined_stage3.slurm)
echo "Stage 3 job ID: $STAGE3_JOB"

echo "Submitting Stage 4: Uncertainty Calibration (depends on Stage 3)"
STAGE4_JOB=$(sbatch --parsable --dependency=afterok:$STAGE3_JOB job/run_streamlined_stage4.slurm)
echo "Stage 4 job ID: $STAGE4_JOB"

echo "Submitting Stage 5: Physics Validation (depends on Stage 4)"
STAGE5_JOB=$(sbatch --parsable --dependency=afterok:$STAGE4_JOB job/run_streamlined_stage5.slurm)
echo "Stage 5 job ID: $STAGE5_JOB"

echo "Submitting Stage 6: Visualization (depends on Stage 5)"
STAGE6_JOB=$(sbatch --parsable --dependency=afterok:$STAGE5_JOB job/run_streamlined_stage6.slurm)
echo "Stage 6 job ID: $STAGE6_JOB"

echo "Submitting Stage 7: Global Comparison (depends on Stage 6)"
STAGE7_JOB=$(sbatch --parsable --dependency=afterok:$STAGE6_JOB job/run_streamlined_stage7.slurm)
echo "Stage 7 job ID: $STAGE7_JOB"

echo ""
echo "All stages submitted successfully!"
echo "Job chain: $STAGE1_JOB -> $STAGE2_JOB -> $STAGE3_JOB -> $STAGE4_JOB -> $STAGE5_JOB -> $STAGE6_JOB -> $STAGE7_JOB"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/slurm/streamlined_stage*_\$JOB_ID.out"
