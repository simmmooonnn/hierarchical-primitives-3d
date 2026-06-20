<div align="center">

# Hierarchical Primitives for 3D Shape Generation

**A Tucker-factor latent diffusion transformer (codename: DiffATs)**
**— unconditional · image-conditional · text-conditional 3D CAD generation**

[Paper (coming soon)] · [Project Page (TODO)] · [Pretrained Weights (TODO)]

</div>

---

## Overview

DiffATs generates 3D CAD shapes by running a **diffusion transformer (DiT)** entirely inside a
**Tucker-decomposed latent space** of a Truncated Signed Distance Field (TSDF). Instead of denoising a
256³ voxel grid (16.7 M values), the model denoises a ~160 K-element factor representation — roughly a
**100× smaller** latent — which makes single-GPU training tractable while still reconstructing detailed
geometry via marching cubes.

**Pipeline:** `ABC mesh → TSDF (N=256) → Tucker (R=24) → DiT (Stochastic Interpolants) → ODE sample → marching cubes`

Key features:
- **Tucker-factor latent.** Diffuse on `(G, U₂, U₃)` with `G = 𝒞 ×₁ U₁` (paper-faithful *l=2* tensor-group
  parameterization; mode-1 absorbed into the core, see [model docstring](diffats/models.py)).
- **Stochastic Interpolants / Flow Matching** (Albergo & Vanden-Eijnden, 2023) with a variance-preserving
  trig schedule `α=cos(πt/2), β=sin(πt/2)`.
- **AdaLN-Zero DiT** with optional **QK-RMSNorm** (SD3/TRELLIS) to fix the flow-matching low-noise
  pathology ([arXiv:2509.20952](https://arxiv.org/abs/2509.20952)).
- **Conditional generation** via zero-init cross-attention: **DINOv2-L** multi-view images (8 views) or
  **CLIP-L** text, with classifier-free guidance.
- **Fast sampling**: 50-step Heun ODE solver, ~1.5 s/shape.

| Model | Params | Condition | Encoder |
|------|-------:|-----------|---------|
| Unconditional | 59.5 M | — | — |
| Image-conditional | ~72 M | 8 views | DINOv2-L/14 (8×257×1024) |
| Text-conditional | ~72 M | caption | CLIP-L/14 (77×768) |

---

## Installation

```bash
git clone https://github.com/[YOUR_USERNAME]/hierarchical-primitives-3d.git
cd hierarchical-primitives-3d

# create an environment (conda or venv), then install the CUDA torch build for your driver:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# install DiffATs (editable) — puts the `diffats` package on your path so scripts can import it
pip install -e .
# optional: agent-demo dependency (Phase 3)
pip install -e ".[agent]"
```

`pip install -e .` is **required** so `scripts/*.py` can `import diffats`. All dependencies are declared
in `pyproject.toml`; `requirements.txt` / `environment.yml` are kept for non-editable setups.

Headless rendering (for the conditioning pipeline) needs EGL/OSMesa available to `pyrender`.
Tested on Python 3.10/3.11, CUDA 12.x, A100/H100.

Data and outputs are resolved relative to the current directory (run scripts from the repo root), or set
`export DIFFATS_ROOT=/path/to/data` to point elsewhere.

---

## Data Preparation

DiffATs trains on the [ABC dataset](https://deep-geometry.github.io/abc-dataset/). Raw meshes and
preprocessed latents are **not** included in this repo (see [Weights & Data](#weights--data)).

```bash
# 1) Filter ABC meshes by inside-fraction → ~10k usable parts
python scripts/data/build_dataset.py --obj_dir abc_obj/obj \
    --out_json meshes_v2_filtered.json \
    --inside_frac_min 0.02 --inside_frac_max 0.25 --grid_N 32 --n_workers 60

# 2) TSDF + Tucker decomposition (N=256, R=24, μ=2·voxel), OP-aligned to an anchor
python scripts/data/preprocess_tucker.py --mesh_list meshes_v2_filtered_10k.json \
    --out_dir data/tucker_N256_R24_v2 --N 256 --rank 24 --mu_factor 2.0 \
    --shard_size 100 --n_workers 60 --anchor_path data/tucker_N256_R24/ref_anchor.pt

# 3) Medoid orthogonal-Procrustes realignment (continuity in latent space)
python scripts/data/realign.py --in_dir data/tucker_N256_R24_v2 \
    --out_dir data/tucker_N256_R24_v2_medoid
```

See `configs/data_tucker.yaml` for all parameters, and `docs/Tucker_Reconstruction_Report.docx` for a
quantitative analysis of Tucker reconstruction fidelity (mean Chamfer ≈ 3.9 voxels).

### Conditioning features (optional)

```bash
# Image condition — render 8 views, encode with DINOv2-L, bundle
python -m diffats.render            # → outputs/renders
python scripts/conditioning/cache_dinov2.py --render_root outputs/renders \
    --out_root outputs/dinov2_embeddings --model facebook/dinov2-large \
    --n_views 8 --image_size 224
python scripts/conditioning/bundle_embeddings.py

# Text condition — caption with a VLM, encode with CLIP-L
python scripts/conditioning/caption.py
python scripts/conditioning/encode_clip.py
```

---

## Training

```bash
# Unconditional Stochastic-Interpolants base model (≈625k steps)
python scripts/train.py --shard_dir data/tucker_N256_R24_v2_medoid \
    --out_dir outputs/train_uncond_si \
    --depth 12 --hidden_size 512 --num_heads 8 --batch_size 32 \
    --steps 625000 --target eps --si trig \
    --optimizer adamw --lr 1e-4 --betas 0.9,0.99

# Image-conditional (warm-start from the unconditional checkpoint)
python scripts/train.py --shard_dir data/tucker_N256_R24_v2_medoid \
    --cond_dir outputs/dinov2_embeddings --cond_dim 1024 \
    --cond_n_views 8 --cond_n_tokens 257 --cond_dropout 0.1 \
    --init_from outputs/train_uncond_si/ckpt_step625000.pt \
    --out_dir outputs/train_image_cond \
    --depth 12 --hidden_size 512 --num_heads 8 --batch_size 32 --steps 200000 \
    --target eps --si trig --t_clamp 0.01 --qk_norm \
    --optimizer lion --lr 3e-5 --betas 0.95,0.98 --ema_decay 0.9999 --grad_clip 1.0

# Text-conditional — same, with CLIP features (cond_dim 768, 77 tokens)
```

Full hyperparameters: `configs/train_{uncond_si,image_cond,text_cond}.yaml`.
The exact cluster (SLURM) invocations used in the paper are in `scripts/slurm/`.

---

## Sampling

```bash
python -m diffats.sampling \
    --ckpt outputs/train_image_cond/ckpt_step200000.pt \
    --out_dir outputs/samples \
    --cond_bundle outputs/dinov2_embeddings/bundle.pt \
    --cond_mesh_ids 00000,00625,01250 \
    --cfg_scale 2.0 --n_steps 50 --solver heun --seed 42 --use_ema
```

The sampler auto-detects DDPM vs. Stochastic-Interpolant checkpoints (`ckpt['si']`) and reconstructs the
mesh via `X = G ×₂ U₂ ×₃ U₃` followed by marching cubes.

---

## Evaluation

```bash
python scripts/eval/eval.py     # Chamfer Distance, F-Score@τ, CLIP-Score
```

---

## Agent Demo (Phase 3)

End-to-end **text prompt → LLM refinement → 3D shape**. Requires `ANTHROPIC_API_KEY` in the environment
(read from env only — never hardcoded; see [security note](REPO_NOTES.md#security)).

```bash
export ANTHROPIC_API_KEY=sk-...
python scripts/demo/agent_demo_claude.py
```

---

## Repository Layout

```
hierarchical-primitives-3d/
├── pyproject.toml          # installable package metadata + dependencies
├── diffats/                # ── the importable library ──
│   ├── __init__.py
│   ├── models.py           #   the DiT (l=2 TGP, AdaLN-Zero, cross-attention)
│   ├── sampling.py         #   ODE / DDIM samplers + SDF reconstruction
│   ├── render.py           #   multi-view headless rendering (shared)
│   └── utils.py            #   TSDF, Tucker, marching cubes, metrics
├── scripts/                # ── runnable entry points (import diffats) ──
│   ├── train.py               # training (uncond + conditional) — main entry
│   ├── data/                  # build_dataset · preprocess_tucker · realign · recon_vs_original
│   ├── conditioning/          # cache_dinov2 · bundle_embeddings · caption · encode_clip · render_batch
│   ├── eval/                  # eval (CD/F-Score/CLIP) · analyze_holdout · pick_holdout
│   ├── demo/                  # agent_demo_claude · agent_demo_qwen (Phase 3)
│   ├── viz/                   # render_report_visuals · decimate_samples
│   └── slurm/                 # SLURM submit scripts (cluster reference)
├── configs/                # YAML configs mirroring the paper runs
├── assets/                 # README figures
├── docs/                   # written technical reports (.docx)
└── REPO_NOTES.md           # provenance, TODOs, security
```

Run the library directly with `python -m diffats.sampling …`, or the entry points with
`python scripts/train.py …` (after `pip install -e .`).

---

## Weights & Data

Pretrained checkpoints and preprocessed Tucker shards are hosted externally (too large for git):

- **Pretrained weights:** _TODO — upload to Hugging Face / GitHub Releases and link here._
- **ABC dataset:** download from the [official source](https://deep-geometry.github.io/abc-dataset/),
  then run [Data Preparation](#data-preparation).

---

## Citation

```bibtex
@misc{diffats2026,
  title  = {Hierarchical Primitives for 3D Shape Generation},
  author = {[Your Name]},
  year   = {2026},
  note   = {University of Michigan}
}
```

## Acknowledgements

Built on Stochastic Interpolants (Albergo & Vanden-Eijnden, 2023), DiT (Peebles & Xie, 2023),
DINOv2, CLIP, and the ABC dataset. See `docs/` for detailed experiment reports.
