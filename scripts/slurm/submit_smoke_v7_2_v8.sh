#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -J smoke_v7_2_v8
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/smoke_v7_2_v8/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu
mkdir -p outputs/smoke_v7_2_v8/v7_2_path outputs/smoke_v7_2_v8/v8_path

echo "=============================="
echo "Smoke #1: v7.2 path (SI + 5 tricks), 100 steps from scratch"
echo "=============================="
python -u scripts/train.py \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --out_dir outputs/smoke_v7_2_v8/v7_2_path \
    --depth 12 --hidden_size 512 --num_heads 8 \
    --batch_size 8 \
    --steps 100 \
    --target eps \
    --si trig \
    --t_sampling logit_normal \
    --t_clamp 0.01 \
    --optimizer adamw \
    --lr 1e-4 --betas 0.9,0.99 --weight_decay 0.01 \
    --lr_schedule cosine \
    --warmup_steps 20 \
    --ema_decay 0.9999 \
    --grad_clip 1.0 \
    --log_every 20 --ckpt_every 100 \
    --bf16 \
    --resume none

echo ""
echo "=============================="
echo "Smoke #2: v8 path (DDPM+MinSNR + Eikonal), 100 steps, resume from v6"
echo "=============================="
# Copy v6 ckpt as start
if [ ! -f outputs/smoke_v7_2_v8/v8_path/ckpt_step625000.pt ]; then
    cp outputs/train_v6/ckpt_step625000.pt outputs/smoke_v7_2_v8/v8_path/ckpt_step625000.pt
fi
python -u scripts/train.py \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --out_dir outputs/smoke_v7_2_v8/v8_path \
    --depth 12 --hidden_size 512 --num_heads 8 \
    --batch_size 8 \
    --steps 625100 \
    --target eps \
    --min_snr_gamma 5 \
    --optimizer adamw \
    --lr 5e-5 --betas 0.9,0.99 --weight_decay 0.0 \
    --eikonal_weight 0.1 \
    --eikonal_k_points 2048 \
    --grad_clip 1.0 \
    --log_every 20 --ckpt_every 100 \
    --bf16 \
    --resume auto

echo "SMOKE DONE"
