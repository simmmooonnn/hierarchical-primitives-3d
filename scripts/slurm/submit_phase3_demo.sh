#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -J phase3_demo
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/phase3_demo/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

python -u scripts/demo/agent_demo_qwen.py
