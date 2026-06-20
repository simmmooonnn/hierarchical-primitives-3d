#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 10:00:00
#SBATCH -J diffats_v7_3
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/train_v7_3/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

# v7.3 = SI (trig) MINIMAL-CHANGE from v6 — only switch eps→velocity, everything else SAME as v6
#   - SI trig schedule (the only real architectural change)
#   - UNIFORM t sampling  (NOT logit-normal — we don't want to confound the test)
#   - CONST LR 1e-4       (NOT cosine — v7.2 was starved by cosine→0)
#   - WD = 0              (match v6 — v7.2's WD=0.01 over-regularized)
#   - EMA β=0.9999        (only affects sampling, doesn't hurt training)
#   - grad_clip = 1.0     (safety against spike, v7 truly crashed before)
#   - t clamp [0.01, 0.99] (safety margin from singular endpoints)
#   - 625k step from scratch (same length as v6 for clean apples-to-apples)
mkdir -p outputs/train_v7_3

python -u scripts/train.py \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --out_dir outputs/train_v7_3 \
    --depth 12 --hidden_size 512 --num_heads 8 \
    --batch_size 32 \
    --steps 625000 \
    --target eps \
    --si trig \
    --t_sampling uniform \
    --t_clamp 0.01 \
    --optimizer adamw \
    --lr 1e-4 --betas 0.9,0.99 --weight_decay 0.0 \
    --ema_decay 0.9999 \
    --grad_clip 1.0 \
    --log_every 500 --ckpt_every 25000 \
    --bf16 \
    --resume auto
