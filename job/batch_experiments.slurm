#!/bin/bash
#SBATCH --job-name=uq_batch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/batch_uq_%j.out
#SBATCH --error=logs/batch_uq_%j.err

# Batch experiment runner for multiple UQ methods and configurations

echo "=========================================="
echo "Batch UQ Experiments Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
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
mkdir -p figures
mkdir -p reproduction_kits

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Define experiment configurations
CONFIGS=(
    "configs/E1_hit_baseline.yaml"
    "configs/E2_hit_bayes.yaml"
    "configs/E5_hit_ens.yaml"
    "configs/E6_hit_ab_ens.yaml"
)

METHODS=("mc" "ensemble")

echo "Running batch experiments for ${#CONFIGS[@]} configs and ${#METHODS[@]} methods"

# Function to run complete experiment pipeline
run_experiment() {
    local config=$1
    local method=$2
    
    echo "----------------------------------------"
    echo "Starting experiment: $config with $method"
    echo "Time: $(date)"
    echo "----------------------------------------"
    
    # Extract experiment ID
    EXP_ID=$(python -c "import yaml; print(yaml.safe_load(open('$config'))['experiment_id'])")
    echo "Experiment ID: $EXP_ID"
    
    # Step 1: Training
    echo "Step 1: Training..."
    case $method in
        "mc")
            python scripts/run_train.py --config $config --cuda
            ;;
        "ensemble")
            python scripts/run_train_ens.py --config $config --cuda
            ;;
    esac
    
    if [ $? -ne 0 ]; then
        echo "Training failed for $config with $method"
        return 1
    fi
    
    # Step 2: Prediction
    echo "Step 2: Prediction..."
    case $method in
        "mc")
            python scripts/predict_mc.py --config $config --split test --cuda
            ;;
        "ensemble")
            python scripts/predict_ens.py --config $config --split test --cuda
            ;;
    esac
    
    if [ $? -ne 0 ]; then
        echo "Prediction failed for $config with $method"
        return 1
    fi
    
    # Step 3: Conformal calibration
    echo "Step 3: Conformal calibration..."
    python scripts/calibrate_conformal.py --config $config --method $method
    
    if [ $? -eq 0 ]; then
        echo "Step 4: Conformal prediction..."
        case $method in
            "mc")
                python scripts/predict_mc.py --config $config --split test --conformal absolute --cuda
                ;;
            "ensemble")
                python scripts/predict_ens.py --config $config --split test --conformal absolute --cuda
                ;;
        esac
    fi
    
    # Step 5: Analysis pipeline
    echo "Step 5: Running analysis pipeline..."
    RESULTS_DIR="results/$EXP_ID"
    
    # Calibration analysis
    python scripts/plot_calibration.py --results_dir $RESULTS_DIR --method $method --split test
    
    # Uncertainty analysis
    python scripts/explain_uncertainty.py --results_dir $RESULTS_DIR --method $method --split test
    
    # Physics validation
    python scripts/validate_physics.py --results_dir $RESULTS_DIR --method $method --split test
    
    echo "Experiment completed: $config with $method"
    return 0
}

# Run all experiments
TOTAL_EXPERIMENTS=$((${#CONFIGS[@]} * ${#METHODS[@]}))
CURRENT_EXP=0
SUCCESSFUL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0

for config in "${CONFIGS[@]}"; do
    for method in "${METHODS[@]}"; do
        CURRENT_EXP=$((CURRENT_EXP + 1))
        echo ""
        echo "=========================================="
        echo "Experiment $CURRENT_EXP of $TOTAL_EXPERIMENTS"
        echo "Config: $config"
        echo "Method: $method"
        echo "=========================================="
        
        if run_experiment $config $method; then
            SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
            echo "✓ SUCCESS: $config with $method"
        else
            FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
            echo "✗ FAILED: $config with $method"
        fi
        
        echo "Progress: $SUCCESSFUL_EXPERIMENTS successful, $FAILED_EXPERIMENTS failed"
    done
done

# Generate comparative analysis
echo ""
echo "=========================================="
echo "Generating Comparative Analysis"
echo "=========================================="

# Find a representative results directory for comparison
COMPARISON_CONFIGS=()
for config in "${CONFIGS[@]}"; do
    EXP_ID=$(python -c "import yaml; print(yaml.safe_load(open('$config'))['experiment_id'])" 2>/dev/null)
    if [ -d "results/$EXP_ID" ]; then
        COMPARISON_CONFIGS+=($config)
    fi
done

if [ ${#COMPARISON_CONFIGS[@]} -gt 0 ]; then
    # Use first successful experiment for comparison base
    FIRST_CONFIG=${COMPARISON_CONFIGS[0]}
    EXP_ID=$(python -c "import yaml; print(yaml.safe_load(open('$FIRST_CONFIG'))['experiment_id'])")
    
    echo "Generating publication figures..."
    python scripts/make_figures.py --results_dir "results/$EXP_ID" --methods "${METHODS[@]}"
    
    echo "Generating HTML report..."
    python scripts/generate_report.py --results_dir "results/$EXP_ID" --methods "${METHODS[@]}"
    
    echo "Creating reproduction kits..."
    for config in "${COMPARISON_CONFIGS[@]}"; do
        EXP_ID=$(python -c "import yaml; print(yaml.safe_load(open('$config'))['experiment_id'])")
        python scripts/package_minimal.py --config $config --results_dir "results/$EXP_ID" --output_dir reproduction_kits
    done
fi

# Final summary
echo ""
echo "=========================================="
echo "Batch Experiments Summary"
echo "=========================================="
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Successful: $SUCCESSFUL_EXPERIMENTS"
echo "Failed: $FAILED_EXPERIMENTS"
echo "Success rate: $(( (SUCCESSFUL_EXPERIMENTS * 100) / TOTAL_EXPERIMENTS ))%"
echo "End time: $(date)"

if [ $FAILED_EXPERIMENTS -eq 0 ]; then
    echo "🎉 All experiments completed successfully!"
    exit 0
else
    echo "⚠️  Some experiments failed. Check individual logs for details."
    exit 1
fi
