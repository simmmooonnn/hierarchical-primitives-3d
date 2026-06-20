#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH -t 09:30:00
#SBATCH -J diffats_v9_cont
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/train_v9/slurm-cont-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

# v9 continuation: 200k → 400k. Resume picks up ckpt_step200000.pt automatically.
python -u scripts/train.py \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --cond_dir  outputs/dinov2_embeddings \
    --cond_dim 1024 --cond_n_views 8 --cond_n_tokens 257 \
    --cond_dropout 0.1 \
    --out_dir outputs/train_v9 \
    --depth 12 --hidden_size 512 --num_heads 8 \
    --batch_size 32 \
    --steps 400000 \
    --target eps \
    --si trig \
    --t_sampling uniform \
    --t_clamp 0.01 \
    --qk_norm \
    --optimizer lion \
    --lr 3e-5 --betas 0.95,0.98 --weight_decay 0.0 \
    --ema_decay 0.9999 \
    --grad_clip 1.0 \
    --log_every 500 --ckpt_every 10000 \
    --bf16 \
    --resume auto
