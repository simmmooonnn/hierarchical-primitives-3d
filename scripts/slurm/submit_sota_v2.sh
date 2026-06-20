#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:45:00
#SBATCH -J sota_eval_v2
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/sota_eval/v2-slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

# v2: CD, F-Score, fixed Volume IoU (contains()), CLIP-Score, +1-NNA-CD
python -u scripts/eval/eval.py
