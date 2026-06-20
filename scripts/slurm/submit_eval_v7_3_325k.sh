#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -J eval_v7_3_325k
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/samples_v7_3_325k/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu
mkdir -p outputs/samples_v7_3_325k

# Eval v7.3 step 325k (the LAST HEALTHY ckpt before crash at step 331228)
# Crash analysis: model was at MA loss 0.115 for 80k step then died in 500 step
# Use EMA weights — EMA tracks ~10k step horizon, completely pre-crash
python -u -m diffats.sampling \
    --ckpt outputs/train_v7_3/ckpt_step325000.pt \
    --out_dir outputs/samples_v7_3_325k \
    --n 16 \
    --n_steps 50 \
    --solver heun \
    --seed 42 \
    --use_ema
