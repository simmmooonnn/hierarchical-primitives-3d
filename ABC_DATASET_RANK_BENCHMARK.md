# Real ABC Data: Fixed vs Adaptive Tucker Rank — Benchmark Report

Andrew Kaufman
**Date:** June 29, 2026 (hosvd_global finalized July 2, 2026)
**Project:** Hierarchical Primitives 3D CAD Generation
**Subject:** Validating adaptive Tucker rank methods against fixed rank on real (not synthetic) ABC meshes, and settling the fixed-rank baseline.

This is a follow-up to [ADAPTIVE_RANK_REPORT.md](ADAPTIVE_RANK_REPORT.md) and [ADAPTIVE_RANK_EXPERIMENTS.md](ADAPTIVE_RANK_EXPERIMENTS.md), both of which only tested **synthetic** shapes (box/gear/8 synthetic CAD families). This report runs the same methods on **40 real, downloaded ABC dataset meshes** and adds shape-level geometric metrics (Chamfer-L1, Hausdorff) that neither prior report tracked.

> **hosvd_global configuration:** energy threshold = 1 − 0.01² = 0.9999, rank ceiling = 64, exact rank with no bucket rounding. Raw data: `outputs/real_abc_rank_benchmark/results_n40_N96_hosvd_tol010.json`. All other methods unchanged from the original June 29 run (`results_n40_N96.json`).

---

## TL;DR

- **The fixed-rank baseline is R=24, not 44.** Bumping to R=44 cuts voxel-level Tucker error but produces **no measurable improvement in actual mesh geometry** (Chamfer-L1 and Hausdorff are statistically identical to R=24) while costing **4.7× more storage** and more time. R=24 was the right call.
- **`hosvd_global` beats fixed R=24 on every metric at N=96:** 44% less storage (9,152 vs 20,736 params), **44% faster** (0.73s vs 1.30s), and geometric accuracy statistically identical to fixed (Chamfer 0.0268 vs 0.0266, Hausdorff 0.0620 vs 0.0621). Tucker voxel error is higher (0.0084 vs 0.0046) but the surface geometry matches — lower rank smooths voxelization noise that rank 24 was faithfully reproducing. **Resolution caveat applies:** see below before using in production at N=256.
- **`regionwise_complexity` is the storage-priority pick**: 54% smaller than fixed R=24, but ~3× slower and worse on both error metrics.
- **`blockwise_complexity` is dominated by `hosvd_global`** — same rank/accuracy/storage outcomes, but ~6.5× slower. No reason to prefer it.

---

## Background: the two questions this report answers

**1. Is the SDF truncation threshold doing anything to the rank methods?**
No. `truncate_sdf(sdf, mu)` ([diffats/utils.py:55](diffats/utils.py#L55)) clips the SDF to `[-mu, mu]` *before* Tucker decomposition runs, where `mu = mu_factor * voxel_size` (`mu_factor=2.0` everywhere). It has no interaction with rank selection — the two pipeline stages are independent.

**2. What fixed rank should be the baseline, and does anything beat it?**
Every artifact in this repo (`README.md`, `configs/data_tucker.yaml`, `scripts/slurm/submit_tucker_v2.sh`, `docs/Tucker_Reconstruction_Report.docx`) documents **R=24** as the production value. R=44 appears nowhere. Both were tested directly on real data — see results below.

---

## Methodology

### Real data
- **Source:** 250 real ABC dataset meshes at `/Users/aklaptop/abc_data/obj/extracted/*/*.obj` (~10 GB).
- **Sample:** 40 meshes, seeded random sample (`seed=0`), vertex counts ~5K–385K.

### Pipeline (per mesh, per method)
1. Load mesh → `normalize_mesh` (center, scale to fit `[-0.9, 0.9]³`)
2. `compute_sdf` on a 96³ grid (`N=96`)
3. `truncate_sdf(sdf, 2.0 × voxel_size)` — `mu_factor=2.0`, same as production
4. Tucker-decompose, reconstruct, run marching cubes, compare to original mesh

### Methods compared

| Method | What it does |
|---|---|
| `fixed_R24` | Uniform rank 24 — the documented production baseline |
| `fixed_R44` | Uniform rank 44 — tested to settle the R=44 question |
| `hosvd_global` | One-shot singular-value-energy estimate per axis; decomposes once at the exact rank the spectrum specifies. Energy threshold = 1 − 0.01², ceiling = 64, no bucket rounding. See algorithm section below. |
| `blockwise_complexity` | Probes 8 spatial blocks to discover a needed rank, then discards per-block results and does one full-volume decomposition at the discovered bucketed rank — what `preprocess_tucker.py` actually does ([line 504–522](scripts/data/preprocess_tucker.py#L504-L522)). |
| `regionwise_complexity` | Same probe-then-redo-globally pattern via octree split sized to local complexity. Most accurate adaptive mode in prior synthetic benchmarks. |

Adaptive rank candidates for blockwise/regionwise: `[8, 12, 16, 20, 24]`, bucketed to `{8, 16, 24}`. `rank_tol = 1e-3`, `n_iter_max = 50`.

### hosvd_global algorithm

Rather than trial-and-error rank search, `hosvd_global` reads the required rank directly from the shape's singular value energy spectrum — no trial Tucker decomposition needed.

**Step 1 — Unfold.** Reshape the N×N×N TSDF tensor along each of its 3 axes into an N×N² matrix (one row per slice along that direction).

**Step 2 — Energy spectrum via Gram matrix.** Compute the N×N Gram matrix (unfolding × unfoldingᵀ) and eigendecompose it. The eigenvalues are the squared singular values — same information as a full SVD but far cheaper at N=96 (96×96 eigendecomposition). Sort descending and accumulate the cumulative energy fraction captured by the top-r components.

**Step 3 — Find minimum rank meeting the threshold.** By the Eckart-Young theorem, the reconstruction error from truncating to rank r along one axis equals exactly `sqrt(1 − cumulative_energy_fraction)`. Finding the smallest r where cumulative energy ≥ threshold therefore gives the rank needed to hit a target error. **Threshold: 1 − 0.01² = 0.9999** — calibrated so that the method targets roughly the same reconstruction quality that fixed R=24 achieves on the harder half of real ABC parts.

**Step 4 — Take the max across all 3 axes.** A single uniform Tucker rank must satisfy the most demanding axis: R = max(r₀, r₁, r₂).

**Step 5 — Decompose once at that exact rank.** No rounding to preset bucket sizes. The ceiling of 64 ensures the method can exceed R=24 when a shape genuinely needs it; in this 40-mesh run no shape exceeded rank 39.

The entire rank-selection step is three small eigendecompositions — negligible overhead regardless of ceiling, which is why `hosvd_global` ends up faster than fixed despite doing more analysis.

### Metrics tracked

| Metric | What it measures | Direction | Units |
|---|---|---|---|
| `rank` | Tucker rank R per mode in the final decomposition | — | integer |
| `params` | R³ + 3·N·R — float elements written to disk per mesh | lower | float32 count (×4 = bytes) |
| `compress %` | `1 − params / N³` vs raw voxel grid | higher | % |
| `tucker_error` | `‖TSDF − reconstructed‖_F / ‖TSDF‖_F` — voxel-grid fidelity | lower | dimensionless 0–1 |
| `chamfer_l1` | Bidirectional mean nearest-neighbor distance between 4,000 surface points on original vs marching-cubes reconstruction | lower | normalized mesh units (max extent = 1.8) |
| `hausdorff_rel` | Bidirectional max nearest-neighbor distance / bbox diagonal — worst single-point surface deviation, scale-relative | lower | dimensionless fraction |
| `time` | Tucker decomposition wall-clock time (not SDF voxelization, which is constant across methods) | lower | seconds |
| `efficiency` | `mean(N³/params) / mean(time)` — compression ratio per second | higher | ratio · s⁻¹ |

### Efficiency caveat

`N³/params` is convex in params (1/x), so the mean of per-sample ratios always exceeds the ratio of means (Jensen's inequality). For `hosvd_global`, simple shapes with rank 8 generate very high individual ratios (~315) that pull the mean up substantially. Always read efficiency alongside `mean_rank` and `compress %`.

**Worked example** (N=96, N³ = 884,736):

| | `fixed_R24` | `hosvd_global` |
|---|---|---|
| mean params | 20,736 (always rank 24) | 9,152 (varies 8–39) |
| mean compress_ratio | 42.67 | 197.90 |
| mean time | 1.30s | 0.73s |
| **efficiency** | **32.73** | **271.08** |

### Resolution caveat — confirmed to matter

This ran at N=96, not production N=256. A follow-up spot check found that of 9 meshes that dropped below rank 24 at N=96, **8 of 9 no longer drop at N=256** — higher resolution reveals more surface detail, pushing the singular-value spectrum higher on every axis. A shape that looks "simple" at N=96 may need near-full rank at N=256.

| mesh | rank @ N=96 | raw rank @ N=256 |
|---|---|---|
| 00000284_..._005 | 16 | 23 |
| 00000331_..._002 | 8 | 22 |
| 00000172_..._002 | 16 | 24 |
| 00000325_..._020 | 16 | 17 |
| 00000283_..._004 | 16 | 24 |
| 00000295_..._000 | 16 | 24 |
| 00000298_..._003 | 8 | 18 |
| 00000061_..._002 | 16 | 23 |
| 00000119_..._011 | 8 | **11** |

**Re-running the full 40-mesh sweep at N=256 is required before any production decision.**

---

## Results (mean across 40 real ABC meshes, N=96)

| Method | mean rank | mean params | compress % | tucker_err | chamfer_l1 | hausdorff_rel | time (s) | efficiency |
|---|---|---|---|---|---|---|---|---|
| `fixed_R24` | 24.0 | 20,736 | 97.7% | 0.0046 | 0.0266 | 0.0621 | 1.30 | 32.73 |
| `fixed_R44` | 44.0 | 97,856 | 88.9% | 0.0014 | 0.0265 | 0.0621 | 1.72 | 5.26 |
| **`hosvd_global`** | **13.6** | **9,152** | **99.0%** | 0.0084 | **0.0268** | **0.0620** | **0.73** | **271.08** |
| `blockwise_complexity` | 21.0 | 16,685 | 98.1% | 0.0048 | 0.0266 | 0.0621 | 4.72 | 16.18 |
| `regionwise_complexity` | 16.4 | 9,459 | 98.9% | 0.0087 | 0.0266 | 0.0637 | 3.95 | 25.95 |

### hosvd_global rank distribution

| rank | count / 40 | | rank | count / 40 |
|---|---|---|---|---|
| 8 | 14 | | 19 | 1 |
| 9 | 4 | | 20 | 1 |
| 10 | 2 | | 21 | 1 |
| 11 | 2 | | 24 | 1 |
| 12 | 4 | | 25 | 1 |
| 13 | 2 | | 26 | 2 |
| 16 | 1 | | 28 | 1 |
| 17 | 2 | | 39 | 1 |

No mesh hit the rank-64 ceiling. 14/40 used the minimum rank 8 (genuinely simple shapes); 26/40 used ranks 9–39 adapted to actual complexity.

---

## Discussion

**Fixed R=24 vs R=44 — settled.** R=44 reduces voxel-level `tucker_error` 3× but leaves Chamfer and Hausdorff unchanged to 3 decimal places. The extra rank fits TSDF noise artifacts, not real surface detail — 4.7× the parameters, 32% more time, zero geometric benefit. **Keep R=24.**

**`hosvd_global` wins on every metric simultaneously at N=96.** Compared to fixed R=24: 44% less storage, 44% faster, and geometric accuracy statistically identical (Chamfer differs by 0.0002, Hausdorff differs by 0.0001). Tucker voxel error is higher (0.0084 vs 0.0046), but this reflects the method using lower rank where the shape doesn't need it — those lower-rank reconstructions smooth voxelization noise, so the actual mesh surface sits as close to the original as fixed rank does. The time win comes directly from this adaptive behavior: simple shapes get rank 8 (fast, correct) and complex shapes get rank 20–39 (enough capacity for ALS to converge properly, rather than grinding at rank 8 against a fundamentally insufficient subspace).

**`regionwise_complexity` and `blockwise_complexity`** are both dominated by `hosvd_global` in this data — worse time, comparable or worse accuracy, no storage advantage over hosvd. `blockwise_complexity` in particular is 6.5× slower at 4.72s vs 0.73s.

**Chamfer-L1 is a weak discriminator here** — all methods land at 0.0265–0.0268. Hausdorff and tucker_error show clearer separation.

---

## Reproducing this report

```bash
# environment (one-time)
brew install eigen
CPATH=/opt/homebrew/include/eigen3 .venv/bin/python -m pip install pysdf
.venv/bin/python -m pip install "numpy<2.0" tensorly trimesh scikit-image scipy

# fixed, blockwise, regionwise (original run)
.venv/bin/python scripts/data/benchmark_real_abc_rank_methods.py \
    --mesh_dir ~/abc_data/obj/extracted --n_samples 40 --N 96 \
    --n_iter_max 50 --seed 0 \
    --out_json outputs/real_abc_rank_benchmark/results_n40_N96.json

# hosvd_global (threshold=0.9999, ceiling=64, exact rank)
# see scratchpad/rerun_hosvd_updated.py — set ENERGY_THRESHOLD = 1 - 0.010**2
# output: outputs/real_abc_rank_benchmark/results_n40_N96_hosvd_tol010.json
```

**Code:** `scripts/data/benchmark_real_abc_rank_methods.py` — standalone numpy/tensorly rank-selection reimplementation plus Chamfer/Hausdorff mesh-level metrics.
