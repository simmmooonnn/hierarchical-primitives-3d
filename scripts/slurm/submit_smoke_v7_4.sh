#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -J smoke_v7_4
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/smoke_v7_4/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
# Install Lion if not present
pip install lion-pytorch 2>&1 | tail -2 || echo "lion-pytorch already installed"
cd $SCRATCH/diffats_gpu
mkdir -p outputs/smoke_v7_4/a outputs/smoke_v7_4/b

echo "============== Smoke v7.4a (AdamW β2=0.999 ε=1e-15 + QK-norm) =============="
python -u scripts/train.py \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --out_dir outputs/smoke_v7_4/a \
    --depth 12 --hidden_size 512 --num_heads 8 \
    --batch_size 8 \
    --steps 100 \
    --target eps \
    --si trig \
    --t_sampling uniform \
    --t_clamp 0.01 \
    --qk_norm \
    --optimizer adamw \
    --lr 1e-4 --betas 0.9,0.999 --adam_eps 1e-15 --weight_decay 0.0 \
    --ema_decay 0.9999 \
    --grad_clip 1.0 \
    --log_every 20 --ckpt_every 100 \
    --bf16 \
    --resume none

echo ""
echo "============== Smoke v7.4b (Lion + QK-norm) =============="
python -u scripts/train.py \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --out_dir outputs/smoke_v7_4/b \
    --depth 12 --hidden_size 512 --num_heads 8 \
    --batch_size 8 \
    --steps 100 \
    --target eps \
    --si trig \
    --t_sampling uniform \
    --t_clamp 0.01 \
    --qk_norm \
    --optimizer lion \
    --lr 3e-5 --betas 0.95,0.98 --weight_decay 0.0 \
    --ema_decay 0.9999 \
    --grad_clip 1.0 \
    --log_every 20 --ckpt_every 100 \
    --bf16 \
    --resume none

echo "SMOKE DONE"
