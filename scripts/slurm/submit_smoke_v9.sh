#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 00:15:00
#SBATCH -J smoke_v9
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/smoke_v9/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu
mkdir -p outputs/smoke_v9

# v9 smoke: conditional DiT training, 50 steps from-scratch, no warm-start
# Verify: embeddings load OK, cond passes through, CFG dropout works, gradients finite
python -u scripts/train.py \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --cond_dir  outputs/dinov2_embeddings \
    --cond_dim 1024 --cond_n_views 8 --cond_n_tokens 257 \
    --cond_dropout 0.1 \
    --out_dir outputs/smoke_v9 \
    --depth 12 --hidden_size 512 --num_heads 8 \
    --batch_size 4 \
    --steps 50 \
    --target eps \
    --si trig \
    --t_sampling uniform \
    --t_clamp 0.01 \
    --qk_norm \
    --optimizer adamw \
    --lr 1e-4 --betas 0.9,0.999 --adam_eps 1e-15 --weight_decay 0.0 \
    --ema_decay 0.9999 \
    --grad_clip 1.0 \
    --log_every 10 --ckpt_every 50 \
    --bf16 \
    --resume none
