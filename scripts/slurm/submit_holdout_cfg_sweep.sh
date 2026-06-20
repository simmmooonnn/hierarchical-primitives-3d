#!/bin/bash
#SBATCH -A eng260004-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -J v9_holdout_cfg
#SBATCH -o /anvil/scratch/x-zsu7/diffats_gpu/outputs/holdout/slurm-%j.out

set -e
module load conda/2026.03
source activate $SCRATCH/envs/diffats
cd $SCRATCH/diffats_gpu
mkdir -p outputs/holdout outputs/holdout/renders outputs/holdout/embs

# === 1. Pick 16 hold-out meshes (unseen during training) ===
echo "=== Step 1: pick hold-out meshes ==="
python -u scripts/eval/pick_holdout.py

# === 2. Render 16 × 8 view normal maps ===
echo ""
echo "=== Step 2: render hold-out views ==="
MESHES=$(awk '{print $1}' outputs/holdout/mesh_list.txt | tr '\n' ' ')
python -u -m diffats.render \
    --mesh $MESHES \
    --out_root outputs/holdout/renders \
    --size 224 --n_views 8 --distance 1.6 --elevation 30

# === 3. Cache DINOv2-L embeddings ===
echo ""
echo "=== Step 3: cache DINOv2 embeddings ==="
python -u scripts/conditioning/cache_dinov2.py \
    --render_root outputs/holdout/renders \
    --out_root outputs/holdout/embs \
    --model facebook/dinov2-large \
    --batch_meshes 4 \
    --n_views 8 --image_size 224 \
    --log_every 8

# === 4. Build hold-out bundle for sampler ===
echo ""
echo "=== Step 4: bundle hold-out embeddings ==="
python -u scripts/conditioning/bundle_embeddings.py \
    --in_dir  outputs/holdout/embs \
    --out_file outputs/holdout/bundle.pt

# === 5. List the bundled mesh_ids ===
HOLDOUT_IDS=$(python -c "
import torch
b = torch.load('outputs/holdout/bundle.pt', weights_only=False)
print(','.join(b['mesh_ids']))
")
echo "Hold-out mesh_ids: $HOLDOUT_IDS"

# === 6. CFG sweep: 6 values × 16 samples ===
for CFG in 1.0 1.5 2.0 3.0 5.0 7.0; do
    OUT="outputs/holdout/samples_cfg${CFG}"
    mkdir -p "$OUT"
    echo ""
    echo "=== Step 6: sample with CFG=$CFG ==="
    python -u -m diffats.sampling \
        --ckpt outputs/train_v9/ckpt_step200000.pt \
        --out_dir "$OUT" \
        --cond_bundle outputs/holdout/bundle.pt \
        --cond_mesh_ids "$HOLDOUT_IDS" \
        --cfg_scale "$CFG" \
        --n_steps 50 --solver heun --seed 42 \
        --use_ema
done

echo ""
echo "ALL DONE"
