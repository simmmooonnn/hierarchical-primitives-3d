#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH -J smoke_v8_v2
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/smoke_v8_v2/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu
mkdir -p outputs/smoke_v8_v2

# Copy v6 ckpt as start (if not already)
if [ ! -f outputs/smoke_v8_v2/ckpt_step625000.pt ]; then
    cp outputs/train_v6/ckpt_step625000.pt outputs/smoke_v8_v2/ckpt_step625000.pt
fi

echo "=============================="
echo "Smoke: v8 path (DDPM+MinSNR + TSDF-aware Eikonal v2)"
echo "  K=8192, tau_factor=0.8 (=> near-surface band |sdf|<0.0126)"
echo "  weight=0.1, timestep weighting=sqrt(α_bar)"
echo "=============================="
python -u scripts/train.py \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --out_dir outputs/smoke_v8_v2 \
    --depth 12 --hidden_size 512 --num_heads 8 \
    --batch_size 8 \
    --steps 625100 \
    --target eps \
    --min_snr_gamma 5 \
    --optimizer adamw \
    --lr 5e-5 --betas 0.9,0.99 --weight_decay 0.0 \
    --eikonal_weight 0.1 \
    --eikonal_k_points 8192 \
    --eikonal_tau_factor 0.8 \
    --eikonal_mu 2.0 \
    --grad_clip 1.0 \
    --log_every 10 --ckpt_every 100 \
    --bf16 \
    --resume auto
