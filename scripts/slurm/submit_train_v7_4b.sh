#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 10:00:00
#SBATCH -J diffats_v7_4b
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/train_v7_4b/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

# v7.4b = SI(trig) + LION optimizer + QK-RMSNorm
#   Lion lr=3e-5 (Lion needs ~3-5× smaller LR than AdamW)
#   Lion betas (0.95, 0.98) — original Chen+ recommendation
#   No β2 second-moment tracking → immune to AdamW state pollution
#   QK-RMSNorm (same as v7.4a)
#   ckpt every 5k step + diagnostic logging (same)
mkdir -p outputs/train_v7_4b

python -u scripts/train.py \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --out_dir outputs/train_v7_4b \
    --depth 12 --hidden_size 512 --num_heads 8 \
    --batch_size 32 \
    --steps 625000 \
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
