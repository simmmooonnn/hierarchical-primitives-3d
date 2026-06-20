#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -J eval_v8_2
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/samples_v8_2/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu
mkdir -p outputs/eval_v8_2 outputs/samples_v8_2 outputs/samples_v8_2_ema

# v8.2 diag (DDPM eps per-t loss)
python -u src/diag_v4.py \
    --ckpt outputs/train_v8_2/ckpt_step650000.pt \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --n 64 \
    --out_path outputs/eval_v8_2/diag.pt

# Sample with raw weights
python -u -m diffats.sampling \
    --ckpt outputs/train_v8_2/ckpt_step650000.pt \
    --out_dir outputs/samples_v8_2 \
    --n 16 \
    --n_steps 250 \
    --seed 42

# Sample with EMA weights
python -u -m diffats.sampling \
    --ckpt outputs/train_v8_2/ckpt_step650000.pt \
    --out_dir outputs/samples_v8_2_ema \
    --n 16 \
    --n_steps 250 \
    --seed 42 \
    --use_ema
