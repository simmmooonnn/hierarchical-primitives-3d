#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -t 00:45:00
#SBATCH -J build_v2
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/build_v2/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

mkdir -p outputs/build_v2

python -u scripts/data/build_dataset.py \
    --obj_dir abc_obj/obj \
    --out_json meshes_v2_filtered.json \
    --inside_frac_min 0.02 \
    --inside_frac_max 0.25 \
    --grid_N 32 \
    --n_workers 60
