#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 00:45:00
#SBATCH -J eval_v9
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/samples_v9/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu
mkdir -p outputs/samples_v9 outputs/samples_v9_cfg2 outputs/samples_v9_uncond

# Pick 16 mesh_ids spread across the dataset: 00000-09999 → step 625
MESH_IDS="00000,00625,01250,01875,02500,03125,03750,04375,05000,05625,06250,06875,07500,08125,08750,09375"

echo "===== v9 with cfg_scale=1.0 (raw cond, EMA) ====="
python -u -m diffats.sampling \
    --ckpt outputs/train_v9/ckpt_step200000.pt \
    --out_dir outputs/samples_v9 \
    --cond_bundle outputs/dinov2_embeddings/bundle.pt \
    --cond_mesh_ids $MESH_IDS \
    --cfg_scale 1.0 \
    --n_steps 50 \
    --solver heun \
    --seed 42 \
    --use_ema

echo ""
echo "===== v9 with cfg_scale=2.0 (CFG amplified, EMA) ====="
python -u -m diffats.sampling \
    --ckpt outputs/train_v9/ckpt_step200000.pt \
    --out_dir outputs/samples_v9_cfg2 \
    --cond_bundle outputs/dinov2_embeddings/bundle.pt \
    --cond_mesh_ids $MESH_IDS \
    --cfg_scale 2.0 \
    --n_steps 50 \
    --solver heun \
    --seed 42 \
    --use_ema

echo ""
echo "===== v9 UNCONDITIONAL (zero cond) — sanity baseline ====="
python -u -m diffats.sampling \
    --ckpt outputs/train_v9/ckpt_step200000.pt \
    --out_dir outputs/samples_v9_uncond \
    --n 16 \
    --n_steps 50 \
    --solver heun \
    --seed 42 \
    --use_ema

echo "ALL DONE"
