#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH -J diffats_v6
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/train_v6/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

# v6 = 数据扩充(10k mesh,inside_frac ∈ [0.02, 0.25] 过滤极端)
#      + 625k step = 2000 epoch(对齐论文 625k 总 step)
#      + Min-SNR-γ=5 加权(Hang et al. 2023,平衡 per-t 损失)
python -u scripts/train.py \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --out_dir outputs/train_v6 \
    --depth 12 --hidden_size 512 --num_heads 8 \
    --batch_size 32 \
    --steps 625000 \
    --target eps --optimizer adamw \
    --lr 1e-4 --betas 0.9,0.99 --weight_decay 0.0 \
    --min_snr_gamma 5.0 \
    --log_every 500 --ckpt_every 25000 \
    --bf16 \
    --resume auto
