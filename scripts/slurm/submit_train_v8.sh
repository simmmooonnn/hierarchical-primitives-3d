#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH -J diffats_v8p
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/train_v8p/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

# v8' = v6 (DDPM eps + Min-SNR-γ=5) + Eikonal + Sign-consistency
#   Hot-start from v6 ckpt_step625000 → finetune to step 725000 (+100k step)
#   Eikonal: weight=0.1, K=8192 random + near-surface mask, √ᾱ_t reweight
#   Sign:    weight=0.2, BCE on -pred_sdf/0.005, attacks phantom zero-crossings
mkdir -p outputs/train_v8p

# Copy v6 final ckpt as starting point (only once)
if [ ! -f outputs/train_v8p/ckpt_step625000.pt ]; then
    cp outputs/train_v6/ckpt_step625000.pt outputs/train_v8p/ckpt_step625000.pt
    echo "Copied v6 ckpt_step625000.pt → outputs/train_v8p/"
fi

python -u scripts/train.py \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --out_dir outputs/train_v8p \
    --depth 12 --hidden_size 512 --num_heads 8 \
    --batch_size 32 \
    --steps 725000 \
    --target eps \
    --min_snr_gamma 5 \
    --optimizer adamw \
    --lr 5e-5 --betas 0.9,0.99 --weight_decay 0.0 \
    --eikonal_weight 0.1 \
    --eikonal_k_points 8192 \
    --eikonal_tau_factor 0.8 \
    --eikonal_mu 2.0 \
    --sign_weight 0.2 \
    --tau_sign 0.005 \
    --grad_clip 1.0 \
    --log_every 500 --ckpt_every 10000 \
    --bf16 \
    --resume auto
