#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:15:00
#SBATCH -J encode_clip
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/clip_caption_embs/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

# CLIP-L text encoder is small (~140M params), should fit comfortably + run < 5 min.
python -u scripts/conditioning/encode_clip.py
