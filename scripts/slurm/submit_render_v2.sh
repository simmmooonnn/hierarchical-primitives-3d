#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:15:00
#SBATCH -J render_v2
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/report_figs/v2-slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu
python -u scripts/viz/render_report_visuals.py
