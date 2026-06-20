#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -J qr_test
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/v6_qr_test/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu
mkdir -p outputs/v6_qr_test

# Sample v6 ckpt with QR-correction enabled, same seed as original
python -u -m diffats.sampling \
    --ckpt outputs/train_v6/ckpt_step625000.pt \
    --out_dir outputs/v6_qr_test \
    --n 16 \
    --n_steps 250 \
    --seed 42 \
    --qr_correct
