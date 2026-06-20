#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -J eval_v7_175k
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/eval_v7_175k/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu
mkdir -p outputs/eval_v7_175k outputs/samples_v7_175k

# Eval v7 (SI) at ckpt_step175000 — 崩溃前最后健康 ckpt
python -u src/diag_v4.py \
    --ckpt outputs/train_v7/ckpt_step175000.pt \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --n 64 \
    --out_path outputs/eval_v7_175k/diag_step175k.pt

python -u -m diffats.sampling \
    --ckpt outputs/train_v7/ckpt_step175000.pt \
    --out_dir outputs/samples_v7_175k \
    --n 16 \
    --n_steps 50 \
    --solver heun \
    --seed 42

python -u scripts/viz/decimate_samples.py \
    --in_dir  outputs/samples_v7_175k \
    --out_dir outputs/samples_v7_175k/decimated \
    --samples 0 4 7 13 \
    --target_faces 100000
