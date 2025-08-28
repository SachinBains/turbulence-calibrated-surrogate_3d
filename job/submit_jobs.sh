#!/bin/bash
# Job submission helper script for CSF3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if job is running
check_job_status() {
    local job_id=$1
    squeue -j $job_id &>/dev/null
    return $?
}

# Function to wait for job completion
wait_for_job() {
    local job_id=$1
    local job_name=$2
    
    print_status "Waiting for job $job_id ($job_name) to complete..."
    
    while check_job_status $job_id; do
        sleep 30
    done
    
    # Check if job completed successfully
    sacct -j $job_id --format=State --noheader | grep -q "COMPLETED"
    if [ $? -eq 0 ]; then
        print_success "Job $job_id ($job_name) completed successfully"
        return 0
    else
        print_error "Job $job_id ($job_name) failed or was cancelled"
        return 1
    fi
}

# Function to submit training job
submit_training() {
    local config=$1
    local method=$2
    local pretrained=$3
    
    print_status "Submitting training job for $config with method $method"
    
    if [ "$method" = "swa" ] && [ -z "$pretrained" ]; then
        print_error "SWA training requires pretrained model path"
        return 1
    fi
    
    if [ "$method" = "swa" ]; then
        job_id=$(sbatch --parsable job/train_uq.slurm $config $method --cuda $pretrained)
    else
        job_id=$(sbatch --parsable job/train_uq.slurm $config $method --cuda)
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Training job submitted: $job_id"
        echo $job_id
        return 0
    else
        print_error "Failed to submit training job"
        return 1
    fi
}

# Function to submit evaluation job
submit_evaluation() {
    local config=$1
    local method=$2
    local split=$3
    
    print_status "Submitting evaluation job for $config with method $method"
    
    job_id=$(sbatch --parsable job/eval_uq.slurm $config $method $split)
    
    if [ $? -eq 0 ]; then
        print_success "Evaluation job submitted: $job_id"
        echo $job_id
        return 0
    else
        print_error "Failed to submit evaluation job"
        return 1
    fi
}

# Function to submit batch job
submit_batch() {
    print_status "Submitting batch experiment job"
    
    job_id=$(sbatch --parsable job/batch_experiments.slurm)
    
    if [ $? -eq 0 ]; then
        print_success "Batch job submitted: $job_id"
        echo $job_id
        return 0
    else
        print_error "Failed to submit batch job"
        return 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  train CONFIG METHOD [PRETRAINED]  - Submit training job"
    echo "  eval CONFIG METHOD [SPLIT]        - Submit evaluation job"
    echo "  batch                              - Submit batch experiment job"
    echo "  status [JOB_ID]                    - Check job status"
    echo "  logs JOB_ID                        - Show job logs"
    echo "  cancel JOB_ID                      - Cancel job"
    echo ""
    echo "Methods: mc, ensemble, swa"
    echo "Splits: val, test (default: test)"
    echo ""
    echo "Examples:"
    echo "  $0 train configs/E1_hit_baseline.yaml mc"
    echo "  $0 train configs/E5_hit_ens.yaml ensemble"
    echo "  $0 train configs/E1_hit_baseline.yaml swa results/E1_hit_baseline/best_model.pth"
    echo "  $0 eval configs/E1_hit_baseline.yaml mc test"
    echo "  $0 batch"
    echo "  $0 status 12345"
}

# Main script logic
case "$1" in
    "train")
        if [ $# -lt 3 ]; then
            print_error "Training requires CONFIG and METHOD arguments"
            show_usage
            exit 1
        fi
        
        CONFIG=$2
        METHOD=$3
        PRETRAINED=$4
        
        if [ ! -f "$CONFIG" ]; then
            print_error "Config file not found: $CONFIG"
            exit 1
        fi
        
        submit_training $CONFIG $METHOD $PRETRAINED
        ;;
        
    "eval")
        if [ $# -lt 3 ]; then
            print_error "Evaluation requires CONFIG and METHOD arguments"
            show_usage
            exit 1
        fi
        
        CONFIG=$2
        METHOD=$3
        SPLIT=${4:-"test"}
        
        if [ ! -f "$CONFIG" ]; then
            print_error "Config file not found: $CONFIG"
            exit 1
        fi
        
        submit_evaluation $CONFIG $METHOD $SPLIT
        ;;
        
    "batch")
        submit_batch
        ;;
        
    "status")
        if [ $# -lt 2 ]; then
            print_status "Showing all user jobs:"
            squeue -u $USER
        else
            JOB_ID=$2
            print_status "Status for job $JOB_ID:"
            squeue -j $JOB_ID
            echo ""
            print_status "Detailed job info:"
            sacct -j $JOB_ID --format=JobID,JobName,State,ExitCode,Start,End,Elapsed,MaxRSS
        fi
        ;;
        
    "logs")
        if [ $# -lt 2 ]; then
            print_error "Logs command requires JOB_ID"
            show_usage
            exit 1
        fi
        
        JOB_ID=$2
        
        print_status "Showing logs for job $JOB_ID"
        
        # Find log files
        OUT_LOG=$(find logs -name "*_${JOB_ID}.out" 2>/dev/null | head -1)
        ERR_LOG=$(find logs -name "*_${JOB_ID}.err" 2>/dev/null | head -1)
        
        if [ -f "$OUT_LOG" ]; then
            echo -e "${GREEN}=== STDOUT LOG ===${NC}"
            tail -50 "$OUT_LOG"
        fi
        
        if [ -f "$ERR_LOG" ]; then
            echo -e "${RED}=== STDERR LOG ===${NC}"
            tail -50 "$ERR_LOG"
        fi
        
        if [ ! -f "$OUT_LOG" ] && [ ! -f "$ERR_LOG" ]; then
            print_warning "No log files found for job $JOB_ID"
        fi
        ;;
        
    "cancel")
        if [ $# -lt 2 ]; then
            print_error "Cancel command requires JOB_ID"
            show_usage
            exit 1
        fi
        
        JOB_ID=$2
        print_status "Cancelling job $JOB_ID"
        scancel $JOB_ID
        
        if [ $? -eq 0 ]; then
            print_success "Job $JOB_ID cancelled"
        else
            print_error "Failed to cancel job $JOB_ID"
        fi
        ;;
        
    "pipeline")
        # Run complete pipeline for a config
        if [ $# -lt 3 ]; then
            print_error "Pipeline requires CONFIG and METHOD arguments"
            show_usage
            exit 1
        fi
        
        CONFIG=$2
        METHOD=$3
        
        print_status "Running complete pipeline for $CONFIG with $METHOD"
        
        # Submit training
        train_job=$(submit_training $CONFIG $METHOD)
        if [ $? -ne 0 ]; then
            exit 1
        fi
        
        # Wait for training to complete
        if wait_for_job $train_job "training"; then
            # Submit evaluation
            eval_job=$(submit_evaluation $CONFIG $METHOD "test")
            if [ $? -eq 0 ]; then
                print_success "Pipeline jobs submitted: training=$train_job, evaluation=$eval_job"
            fi
        else
            print_error "Training failed, skipping evaluation"
            exit 1
        fi
        ;;
        
    *)
        print_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac
