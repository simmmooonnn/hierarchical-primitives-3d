#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH -t 10:00:00
#SBATCH -J diffats_v9
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/train_v9/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

# v9 = v7.4b(SI/Lion/QK-norm) warm-start + 8-view DINOv2-L cross-attention conditioning
#   Backbone weights init from v7.4b ckpt_step625000 (best uncond SI model so far)
#   Cross-attn + cond_proj layers init from random (zero-init proj → identity at start)
#   CFG dropout 0.1 (Esser+ SDXL standard)
#   Bundle.pt provides ~30s embedding load vs ~10min per-file
mkdir -p outputs/train_v9

python -u scripts/train.py \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --cond_dir  outputs/dinov2_embeddings \
    --cond_dim 1024 --cond_n_views 8 --cond_n_tokens 257 \
    --cond_dropout 0.1 \
    --init_from outputs/train_v7_4b/ckpt_step625000.pt \
    --out_dir outputs/train_v9 \
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
