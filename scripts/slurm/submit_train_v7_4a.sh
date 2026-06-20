#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 10:00:00
#SBATCH -J diffats_v7_4a
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/train_v7_4a/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

# v7.4a = SI(trig) + MODERN AdamW recipe + QK-RMSNorm (SD3/TRELLIS standard)
#   AdamW β=(0.9, 0.999), ε=1e-15 (FM standard)
#   QK-RMSNorm in DiT attention (prevents exploding norms)
#   ckpt every 5k step for fast crash recovery
#   target_rms / pred_rms / grad_norm logged each step
mkdir -p outputs/train_v7_4a

python -u scripts/train.py \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --out_dir outputs/train_v7_4a \
    --depth 12 --hidden_size 512 --num_heads 8 \
    --batch_size 32 \
    --steps 625000 \
    --target eps \
    --si trig \
    --t_sampling uniform \
    --t_clamp 0.01 \
    --qk_norm \
    --optimizer adamw \
    --lr 1e-4 --betas 0.9,0.999 --adam_eps 1e-15 --weight_decay 0.0 \
    --ema_decay 0.9999 \
    --grad_clip 1.0 \
    --log_every 500 --ckpt_every 5000 \
    --bf16 \
    --resume auto
