#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 10:00:00
#SBATCH -J diffats_v7_2
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/train_v7_2/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

# v7.2 = SI(trig) + 5 stability tricks (from scratch):
#   - logit-normal t sampling (SD3 trick)
#   - EMA beta=0.9999
#   - cosine LR decay with 2k step warmup
#   - t clamp [0.01, 0.99]   (wider margin from endpoints)
#   - weight_decay=0.01
#   + grad_clip=1.0 + non-finite guard
mkdir -p outputs/train_v7_2

python -u scripts/train.py \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --out_dir outputs/train_v7_2 \
    --depth 12 --hidden_size 512 --num_heads 8 \
    --batch_size 32 \
    --steps 625000 \
    --target eps \
    --si trig \
    --t_sampling logit_normal \
    --t_clamp 0.01 \
    --optimizer adamw \
    --lr 1e-4 --betas 0.9,0.99 --weight_decay 0.01 \
    --lr_schedule cosine \
    --warmup_steps 2000 \
    --ema_decay 0.9999 \
    --grad_clip 1.0 \
    --log_every 500 --ckpt_every 25000 \
    --bf16 \
    --resume auto
