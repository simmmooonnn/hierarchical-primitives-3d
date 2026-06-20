#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -J diffats_v8_2
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/train_v8_2/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

# v8.2 = v6 (DDPM eps + Min-SNR-γ=5) + WEAK sign loss only (no Eikonal)
#   Hot-start from v6 ckpt_step625000 → fine-tune to step 650000 (+25k step)
#   Sign weight = 0.05 (vs failed v8' 0.2 — much gentler)
#   LR = 2.5e-5 (vs v6 1e-4 — quarter for fine-tune)
#   Tests: does weak sign loss preserve big_comp gain WITHOUT damaging eps fidelity?
mkdir -p outputs/train_v8_2

if [ ! -f outputs/train_v8_2/ckpt_step625000.pt ]; then
    cp outputs/train_v6/ckpt_step625000.pt outputs/train_v8_2/ckpt_step625000.pt
    echo "Copied v6 ckpt_step625000.pt → outputs/train_v8_2/"
fi

python -u scripts/train.py \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --out_dir outputs/train_v8_2 \
    --depth 12 --hidden_size 512 --num_heads 8 \
    --batch_size 32 \
    --steps 650000 \
    --target eps \
    --min_snr_gamma 5 \
    --optimizer adamw \
    --lr 2.5e-5 --betas 0.9,0.99 --weight_decay 0.0 \
    --sign_weight 0.05 \
    --tau_sign 0.005 \
    --eikonal_weight 0.0 \
    --ema_decay 0.9999 \
    --grad_clip 1.0 \
    --log_every 200 --ckpt_every 5000 \
    --bf16 \
    --resume auto
