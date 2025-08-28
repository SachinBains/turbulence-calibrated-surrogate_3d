#!/bin/bash
#SBATCH --job-name=uq_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/eval_uq_%j.out
#SBATCH --error=logs/eval_uq_%j.err

# Job configuration
CONFIG_FILE=${1:-"configs/E1_hit_baseline.yaml"}
METHOD=${2:-"mc"}
SPLIT=${3:-"test"}
CUDA_FLAG="--cuda"

echo "=========================================="
echo "UQ Evaluation Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: $CONFIG_FILE"
echo "Method: $METHOD"
echo "Split: $SPLIT"
echo "Time: $(date)"
echo "=========================================="

# Load modules
module purge
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0

# Activate virtual environment
source ~/venvs/turbml/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create logs directory
mkdir -p logs

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Extract experiment ID from config
EXP_ID=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['experiment_id'])")
RESULTS_DIR="results/$EXP_ID"

echo "Experiment ID: $EXP_ID"
echo "Results directory: $RESULTS_DIR"

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory not found: $RESULTS_DIR"
    echo "Make sure training has been completed first."
    exit 1
fi

# Run prediction
echo "Step 1: Running prediction..."
case $METHOD in
    "mc")
        python scripts/predict_mc.py --config $CONFIG_FILE --split $SPLIT $CUDA_FLAG
        ;;
    "ens"|"ensemble")
        python scripts/predict_ens.py --config $CONFIG_FILE --split $SPLIT $CUDA_FLAG
        ;;
    *)
        echo "Error: Unknown method '$METHOD'. Use 'mc' or 'ens'"
        exit 1
        ;;
esac

if [ $? -ne 0 ]; then
    echo "Prediction failed!"
    exit 1
fi

# Run conformal calibration
echo "Step 2: Running conformal calibration..."
python scripts/calibrate_conformal.py --config $CONFIG_FILE --method $METHOD

if [ $? -eq 0 ]; then
    echo "Step 3: Running prediction with conformal intervals..."
    case $METHOD in
        "mc")
            python scripts/predict_mc.py --config $CONFIG_FILE --split $SPLIT --conformal absolute $CUDA_FLAG
            ;;
        "ens"|"ensemble")
            python scripts/predict_ens.py --config $CONFIG_FILE --split $SPLIT --conformal absolute $CUDA_FLAG
            ;;
    esac
fi

# Run calibration analysis
echo "Step 4: Running calibration analysis..."
python scripts/plot_calibration.py --results_dir $RESULTS_DIR --method $METHOD --split $SPLIT

# Run uncertainty analysis
echo "Step 5: Running uncertainty analysis..."
python scripts/explain_uncertainty.py --results_dir $RESULTS_DIR --method $METHOD --split $SPLIT

# Run physics validation
echo "Step 6: Running physics validation..."
python scripts/validate_physics.py --results_dir $RESULTS_DIR --method $METHOD --split $SPLIT

echo "=========================================="
echo "UQ Evaluation Job Finished"
echo "Time: $(date)"
echo "=========================================="
