#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -J realign_v2
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/data/tucker_N256_R24_v2_medoid/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

mkdir -p data/tucker_N256_R24_v2_medoid

python -u scripts/data/realign.py \
    --in_dir  data/tucker_N256_R24_v2 \
    --out_dir data/tucker_N256_R24_v2_medoid
