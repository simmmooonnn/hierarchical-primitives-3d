#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -J v6_clip
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/samples_v6_step625k_clipped/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu
mkdir -p outputs/samples_v6_step625k_clipped

# Re-sample v6 with TSDF clipping to ±μh,同 seed=42 跟原版 16 sample 一一对照
python -u -m diffats.sampling \
    --ckpt outputs/train_v6/ckpt_step625000.pt \
    --out_dir outputs/samples_v6_step625k_clipped \
    --n 16 \
    --n_steps 250 \
    --seed 42 \
    --clip_tsdf \
    --mu_factor 2.0

# Decimate 同样 4 个样本(0/4/7/13)
python -u scripts/viz/decimate_samples.py \
    --in_dir  outputs/samples_v6_step625k_clipped \
    --out_dir outputs/samples_v6_step625k_clipped/decimated \
    --samples 0 4 7 13 \
    --target_faces 100000
