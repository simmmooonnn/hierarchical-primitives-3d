#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J cache_dinov2
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/dinov2_embeddings/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu
mkdir -p outputs/dinov2_embeddings

# Pre-cache DINOv2-L embeddings on 10k mesh × 8 view normal maps
# Output: outputs/dinov2_embeddings/<mesh_id>.pt  shape (8, 257, 1024) fp16
# Estimated: ~30 min on H100 (model ~300M params, 80k images total)
python -u scripts/conditioning/cache_dinov2.py \
    --render_root outputs/renders \
    --out_root outputs/dinov2_embeddings \
    --model facebook/dinov2-large \
    --batch_meshes 8 \
    --n_views 8 \
    --image_size 224 \
    --log_every 100
