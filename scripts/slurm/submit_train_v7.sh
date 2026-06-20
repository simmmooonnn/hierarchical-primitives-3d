#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH -J diffats_v7
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/train_v7/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

# v7 = Stochastic Interpolants (Albergo & Vanden-Eijnden 2023)
#      schedule = trig: α=cos(πt/2), β=sin(πt/2) variance-preserving
#      target  = velocity field b = α̇ x_0 + β̇ x_1
#      sampling = ODE (separate eval script,默认 Heun 50 step)
#      same DiT model + 10k filtered ABC + 625k step + AdamW + Min-SNR off
python -u scripts/train.py \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --out_dir outputs/train_v7 \
    --depth 12 --hidden_size 512 --num_heads 8 \
    --batch_size 32 \
    --steps 625000 \
    --target eps \
    --si trig \
    --optimizer adamw \
    --lr 1e-4 --betas 0.9,0.99 --weight_decay 0.0 \
    --log_every 500 --ckpt_every 25000 \
    --bf16 \
    --resume auto
