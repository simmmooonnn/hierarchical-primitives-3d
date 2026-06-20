#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:45:00
#SBATCH -J eval_v7_4ab
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/eval_v7_4ab/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu
mkdir -p outputs/eval_v7_4ab outputs/samples_v7_4a outputs/samples_v7_4b

echo "=========================================="
echo " v7.4a: sample 16 with EMA weights (SI Heun 50-step)"
echo "=========================================="
python -u -m diffats.sampling \
    --ckpt outputs/train_v7_4a/ckpt_step625000.pt \
    --out_dir outputs/samples_v7_4a \
    --n 16 \
    --n_steps 50 \
    --solver heun \
    --seed 42 \
    --use_ema

echo ""
echo "=========================================="
echo " v7.4b: sample 16 with EMA weights (SI Heun 50-step)"
echo "=========================================="
python -u -m diffats.sampling \
    --ckpt outputs/train_v7_4b/ckpt_step625000.pt \
    --out_dir outputs/samples_v7_4b \
    --n 16 \
    --n_steps 50 \
    --solver heun \
    --seed 42 \
    --use_ema

echo "ALL DONE"
