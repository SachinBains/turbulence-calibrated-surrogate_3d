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
python -u scripts/run_train.py --config ${1} --seed ${2}
