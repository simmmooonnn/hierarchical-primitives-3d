#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH -J diffats_v7_1
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/train_v7_1/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

# v7.1 = v7 + grad clipping (1.0) + nan/inf guard
#        从崩溃前的 ckpt_step175000 续训到 625000
#        (出现 ckpt_step175000.pt 拷贝在 train_v7_1/ 内 → --resume auto 自动识别)
python -u scripts/train.py \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --out_dir outputs/train_v7_1 \
    --depth 12 --hidden_size 512 --num_heads 8 \
    --batch_size 32 \
    --steps 625000 \
    --target eps \
    --si trig \
    --optimizer adamw \
    --lr 1e-4 --betas 0.9,0.99 --weight_decay 0.0 \
    --grad_clip 1.0 \
    --log_every 500 --ckpt_every 25000 \
    --bf16 \
    --resume auto
