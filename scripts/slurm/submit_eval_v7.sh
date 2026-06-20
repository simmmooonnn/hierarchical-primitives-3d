#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -J eval_v7
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/eval_v7/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu
mkdir -p outputs/eval_v7 outputs/samples_v7_step625k

# (1) per-t loss diagnostic at step 625k (uses DDPM-style t bins,
#      仍然 informative because SI's t ∈ [0,1] maps to T*t for embedder)
python -u src/diag_v4.py \
    --ckpt outputs/train_v7/ckpt_step625000.pt \
    --shard_dir data/tucker_N256_R24_v2_medoid \
    --n 64 \
    --out_path outputs/eval_v7/diag_step625k.pt

# (2) ODE sample 16 shapes (Heun 50 step, seed=42 matching v6 for fair compare)
python -u -m diffats.sampling \
    --ckpt outputs/train_v7/ckpt_step625000.pt \
    --out_dir outputs/samples_v7_step625k \
    --n 16 \
    --n_steps 50 \
    --solver heun \
    --seed 42

# (3) Decimate a few for visual inspection
python -u scripts/viz/decimate_samples.py \
    --in_dir  outputs/samples_v7_step625k \
    --out_dir outputs/samples_v7_step625k/decimated \
    --samples 0 4 7 13 \
    --target_faces 100000
