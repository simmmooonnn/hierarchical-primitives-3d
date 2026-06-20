#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -J eval_v5c
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/eval_v5c_step200k/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu
mkdir -p outputs/eval_v5c_step200k

# (1) Final per-t loss diagnostic at step 200k
python -u src/diag_v4.py \
    --ckpt outputs/train_v5c_medoid/ckpt_step200000.pt \
    --shard_dir data/tucker_N256_R24_relaxed_medoid \
    --n 64 \
    --out_path outputs/eval_v5c_step200k/diag_step200k.pt

# (2) DDIM 250-step sample, 16 shapes
python -u -m diffats.sampling \
    --ckpt outputs/train_v5c_medoid/ckpt_step200000.pt \
    --out_dir outputs/samples_v5c_step200k \
    --n 16 \
    --n_steps 250 \
    --seed 42

# (3) Decimate a few samples for visual inspection
python -u scripts/viz/decimate_samples.py \
    --in_dir  outputs/samples_v5c_step200k \
    --out_dir outputs/samples_v5c_step200k/decimated \
    --samples 0 4 7 13 \
    --target_faces 100000
