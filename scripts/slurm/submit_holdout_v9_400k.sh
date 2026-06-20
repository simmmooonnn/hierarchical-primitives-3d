#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -J v9_400k_eval
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/holdout/slurm-400k-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

# Reuse same holdout bundle + mesh_ids — compare v9 step 200k vs 400k at CFG=3.0
HOLDOUT_IDS=$(python -c "
import torch
b = torch.load('outputs/holdout/bundle.pt', weights_only=False)
print(','.join(b['mesh_ids']))
")
mkdir -p outputs/holdout/samples_400k_cfg3.0

python -u -m diffats.sampling \
    --ckpt outputs/train_v9/ckpt_step400000.pt \
    --out_dir outputs/holdout/samples_400k_cfg3.0 \
    --cond_bundle outputs/holdout/bundle.pt \
    --cond_mesh_ids "$HOLDOUT_IDS" \
    --cfg_scale 3.0 \
    --n_steps 50 --solver heun --seed 42 \
    --use_ema
