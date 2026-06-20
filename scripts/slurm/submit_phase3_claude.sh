#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:15:00
#SBATCH -J phase3_claude
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/phase3_claude/slurm-%j.out

# Note: ANTHROPIC_API_KEY must be exported in the submitting shell.
# Slurm propagates env vars by default; the script reads from os.environ.

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

# Install anthropic SDK if missing (idempotent).
pip install --quiet anthropic 2>&1 | tail -2

python -u scripts/demo/agent_demo_claude.py
