#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -J bundle_embs
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/dinov2_embeddings/bundle-slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

python -u scripts/conditioning/bundle_embeddings.py \
    --in_dir  outputs/dinov2_embeddings \
    --out_file outputs/dinov2_embeddings/bundle.pt
