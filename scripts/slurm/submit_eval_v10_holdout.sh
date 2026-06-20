#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -J eval_v10
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/holdout/v10-slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu

# Use SAME 16 hold-out mesh_ids that we used for v9 image cond eval.
# These IDs are 8-digit folder + hash strings; same captions were generated in caption_batch.
HOLDOUT_IDS=$(awk '{print $1}' outputs/holdout/mesh_list.txt | xargs -I{} basename {} .obj | tr '\n' ',' | sed 's/,$//')
echo "Hold-out IDs: $HOLDOUT_IDS"

# Three CFG values + one uncond (cond=0 forced via zero embeddings)
for CFG in 1.0 3.0 7.5; do
    OUT="outputs/holdout/v10_samples_cfg${CFG}"
    mkdir -p "$OUT"
    echo ""
    echo "===== v10 with cfg_scale=$CFG (text cond, EMA) ====="
    python -u -m diffats.sampling \
        --ckpt outputs/train_v10/ckpt_step200000.pt \
        --out_dir "$OUT" \
        --cond_bundle outputs/clip_caption_embs/bundle.pt \
        --cond_mesh_ids "$HOLDOUT_IDS" \
        --cfg_scale "$CFG" \
        --n_steps 50 --solver heun --seed 42 \
        --use_ema
done

# Uncond baseline: same seed, no cond at all
mkdir -p outputs/holdout/v10_samples_uncond
echo ""
echo "===== v10 UNCONDITIONAL (zero cond) ====="
python -u -m diffats.sampling \
    --ckpt outputs/train_v10/ckpt_step200000.pt \
    --out_dir outputs/holdout/v10_samples_uncond \
    --n 16 \
    --n_steps 50 --solver heun --seed 42 \
    --use_ema

echo "ALL DONE"
