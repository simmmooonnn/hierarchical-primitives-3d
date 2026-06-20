#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:45:00
#SBATCH -J eval_v7_2_v8p
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/eval_v7_2_v8p/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu
mkdir -p outputs/eval_v7_2_v8p outputs/samples_v7_2 outputs/samples_v8p

echo "=============================="
echo " v7.2 diag (SI per-t loss)"
echo "=============================="
python -u src/diag_v4.py \
    --ckpt outputs/train_v7_2/ckpt_step625000.pt \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --n 64 \
    --out_path outputs/eval_v7_2_v8p/diag_v7_2.pt

echo ""
echo "=============================="
echo " v7.2 sample (EMA weights, SI Heun ODE 50-step)"
echo "=============================="
python -u -m diffats.sampling \
    --ckpt outputs/train_v7_2/ckpt_step625000.pt \
    --out_dir outputs/samples_v7_2 \
    --n 16 \
    --n_steps 50 \
    --solver heun \
    --seed 42 \
    --use_ema

echo ""
echo "=============================="
echo " v8' diag (DDPM eps per-t loss)"
echo "=============================="
python -u src/diag_v4.py \
    --ckpt outputs/train_v8p/ckpt_step725000.pt \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --n 64 \
    --out_path outputs/eval_v7_2_v8p/diag_v8p.pt

echo ""
echo "=============================="
echo " v8' sample (raw weights, DDIM 250-step)"
echo "=============================="
python -u -m diffats.sampling \
    --ckpt outputs/train_v8p/ckpt_step725000.pt \
    --out_dir outputs/samples_v8p \
    --n 16 \
    --n_steps 250 \
    --seed 42

echo "ALL DONE"
