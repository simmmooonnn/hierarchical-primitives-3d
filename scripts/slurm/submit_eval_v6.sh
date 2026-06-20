#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -J eval_v6
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/eval_v6/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu
mkdir -p outputs/eval_v6 outputs/samples_v6_step625k

# (1) per-t loss diagnostic at step 625k
python -u src/diag_v4.py \
    --ckpt outputs/train_v6/ckpt_step625000.pt \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --n 64 \
    --out_path outputs/eval_v6/diag_step625k.pt

# (2) DDIM 250-step sample, 16 shapes (same seed as v5c for fair comparison)
python -u -m diffats.sampling \
    --ckpt outputs/train_v6/ckpt_step625000.pt \
    --out_dir outputs/samples_v6_step625k \
    --n 16 \
    --n_steps 250 \
    --seed 42

# (3) Decimate a few samples for visual inspection
python -u scripts/viz/decimate_samples.py \
    --in_dir  outputs/samples_v6_step625k \
    --out_dir outputs/samples_v6_step625k/decimated \
    --samples 0 4 7 13 \
    --target_faces 100000
