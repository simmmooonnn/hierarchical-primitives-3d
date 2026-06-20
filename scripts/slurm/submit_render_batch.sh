#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J render_10k
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/renders/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu
mkdir -p outputs/renders

# Render all 10k ABC meshes from Tucker training set
# 8 views @ 224×224, RGB + normal + depth = 24 PNGs per mesh
# Total expected: 240k PNGs, ~53 min based on smoke test (0.32s/mesh)
python -u scripts/conditioning/render_batch.py \
    --mesh_id_map data/mesh_id_to_path.json \
    --out_root outputs/renders \
    --size 224 \
    --n_views 8 \
    --distance 1.6 \
    --elevation 30 \
    --log_every 200
