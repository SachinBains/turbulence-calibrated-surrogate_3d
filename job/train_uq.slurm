#!/bin/bash
#SBATCH --job-name=uq_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_uq_%j.out
#SBATCH --error=logs/train_uq_%j.err

# Job configuration
CONFIG_FILE=${1:-"configs/E1_hit_baseline.yaml"}
METHOD=${2:-"mc"}
CUDA_FLAG=${3:-"--cuda"}

echo "=========================================="
echo "UQ Training Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: $CONFIG_FILE"
echo "Method: $METHOD"
echo "Time: $(date)"
echo "=========================================="

# Load modules (adjust for your cluster)
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

# Print system info
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Run training based on method
case $METHOD in
    "mc")
        echo "Running MC Dropout training..."
        python scripts/run_train.py --config $CONFIG_FILE $CUDA_FLAG
        ;;
    "ensemble")
        echo "Running Ensemble training..."
        python scripts/run_train_ens.py --config $CONFIG_FILE $CUDA_FLAG
        ;;
    "swa")
        if [ -z "$4" ]; then
            echo "Error: SWA training requires pretrained model path as 4th argument"
            exit 1
        fi
        PRETRAINED_PATH=$4
        echo "Running SWA training with pretrained model: $PRETRAINED_PATH"
        python scripts/run_train_swa.py --config $CONFIG_FILE --pretrained $PRETRAINED_PATH $CUDA_FLAG
        ;;
    *)
        echo "Error: Unknown method '$METHOD'. Use 'mc', 'ensemble', or 'swa'"
        exit 1
        ;;
esac

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    
    # Run basic evaluation
    echo "Running evaluation..."
    case $METHOD in
        "mc")
            python scripts/predict_mc.py --config $CONFIG_FILE --split test $CUDA_FLAG
            ;;
        "ensemble")
            python scripts/predict_ens.py --config $CONFIG_FILE --split test $CUDA_FLAG
            ;;
    esac
    
    echo "Job completed at: $(date)"
else
    echo "Training failed with exit code: $?"
    exit 1
fi

echo "=========================================="
echo "UQ Training Job Finished"
echo "=========================================="
