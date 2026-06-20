#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:15:00
#SBATCH -J render_smoke
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/render_smoke/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu
mkdir -p outputs/render_smoke

# Smoke test: render 4 ABC samples × 8 views (24 PNGs each), get per-mesh time
# Picks first 4 .obj files we can find
MESHES=$(find abc_obj/obj -name '*.obj' | head -4)
echo "Test meshes:"
echo "$MESHES"
echo "---"

python -u -m diffats.render \
    --mesh $MESHES \
    --out_root outputs/render_smoke \
    --size 224 \
    --n_views 8 \
    --distance 1.6 \
    --elevation 30
