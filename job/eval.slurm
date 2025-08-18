#!/bin/bash
#SBATCH -J eval_hit
#SBATCH -o job/logs/eval_%j.out
#SBATCH -e job/logs/eval_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH --mem=16G
module purge
python -u scripts/run_eval.py --config ${1} --seed ${2} --mc-samples 32 --temperature-scale --conformal
