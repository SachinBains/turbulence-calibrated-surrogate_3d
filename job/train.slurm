#!/bin/bash
#SBATCH -J train_hit
#SBATCH -o job/logs/train_%j.out
#SBATCH -e job/logs/train_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH --mem=24G
module purge
python -u scripts/run_train.py --config ${1} --seed ${2}
