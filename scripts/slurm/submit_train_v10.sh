#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 10:00:00
#SBATCH -J diffats_v10
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/train_v10/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

# v10 = v7.4b warm-start + CLIP-L TEXT cross-attention
#   cond_dim 768 (CLIP-L), cond_n_views=1 (no view), cond_n_tokens=77 (CLIP max)
#   Note: NOT warm-start from v9 because cond_proj has different input dim (1024 vs 768)
#   v7.4b is clean SI backbone, cross-attn + cond_proj will init zero-init.
mkdir -p outputs/train_v10

python -u scripts/train.py \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --cond_dir  outputs/clip_caption_embs \
    --cond_dim 768 --cond_n_views 1 --cond_n_tokens 77 \
    --cond_dropout 0.1 \
    --init_from outputs/train_v7_4b/ckpt_step625000.pt \
    --out_dir outputs/train_v10 \
    --depth 12 --hidden_size 512 --num_heads 8 \
    --batch_size 32 \
    --steps 200000 \
    --target eps \
    --si trig \
    --t_sampling uniform \
    --t_clamp 0.01 \
    --qk_norm \
    --optimizer lion \
    --lr 3e-5 --betas 0.95,0.98 --weight_decay 0.0 \
    --ema_decay 0.9999 \
    --grad_clip 1.0 \
    --log_every 500 --ckpt_every 5000 \
    --bf16 \
    --resume auto
