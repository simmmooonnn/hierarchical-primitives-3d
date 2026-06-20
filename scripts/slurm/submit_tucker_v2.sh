#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -J tucker_v2
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/data/tucker_N256_R24_v2/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

mkdir -p data/tucker_N256_R24_v2

# 用 chapter 0+1+2 过滤后的 ~10k mesh,跟 v5c 同 Tucker 配置(N=256, R=24, mu=2.0)
# 复用现有 anchor(ch0 样本 42 作为 reference)
python -u scripts/data/preprocess_tucker.py \
    --mesh_list $SCRATCH/diffats_gpu/meshes_v2_filtered_10k.json \
    --out_dir   $SCRATCH/diffats_gpu/data/tucker_N256_R24_v2 \
    --N 256 --rank 24 --mu_factor 2.0 \
    --shard_size 100 \
    --n_workers 60 \
    --anchor_path $SCRATCH/diffats_gpu/data/tucker_N256_R24/ref_anchor.pt
