#!/bin/bash
#SBATCH -J train_hit
#SBATCH --partition=gpuV
#SBATCH -o job/logs/train_%j.out
#SBATCH -e job/logs/train_%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH --mem=24G
module purge
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0

# Activate virtual environment
source ~/venvs/turbml/bin/activate

# Set Python path and change to project directory
cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

python -u scripts/run_train.py --config ${1} --seed ${2}
