"""
save_tucker_sdf.py — Tucker factor extraction with Procrustes alignment for
3D SDF data (ABC mesh dataset → TSDF → Tucker → align to reference anchor).

Pipeline (per sample):
  1. Load mesh, normalize, compute SDF on N^3 grid
  2. TSDF: clamp to [-mu_factor*voxel_size, +mu_factor*voxel_size]
  3. Tucker decompose with uniform rank R → core C, factors [U_1, U_2, U_3]
  4. For each mode k: Q_k = argmin_Q ||U_k Q - U_k_ref||_F via SVD(U_k^T U_k_ref)
  5. Apply rotations and absorb into the core (preserves reconstruction):
       U_k_aligned = U_k @ Q_k
       C_aligned   = einsum('abc,ai,bj,ck->ijk', C, Q_1, Q_2, Q_3)

Anchor: one reference mesh picked by --seed (default 42) from the mesh list;
its Tucker factors become [U_1_ref, U_2_ref, U_3_ref] for all other samples.

Output: shards of (U_1, U_2, U_3, C, mesh_id, mesh_path, voxel_size) plus
ref_anchor.pt and manifest.txt.

Usage (smoke):
  python scripts/save_tucker_sdf.py \
    --mesh_list /home/zijian/abc_sdf_experiment/outputs/meshes_100.json \
    --out_dir outputs/tucker_smoke_N128_R24 \
    --N 128 --rank 24 --mu_factor 2 \
    --n_workers 8 --shard_size 50
"""

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "BLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch
import trimesh
import tensorly as tl
from tensorly.decomposition import tucker
from tqdm import tqdm

from diffats.utils import normalize_mesh, compute_sdf, truncate_sdf

REF_SEED = 42


# ---------------------------------------------------------------------------
# Tucker + Procrustes
# ---------------------------------------------------------------------------

def tucker_decompose(arr_f64, rank, n_iter_max=100):
    core, factors = tucker(tl.tensor(arr_f64), rank=rank,
                           n_iter_max=n_iter_max, verbose=False)
    return np.array(core), [np.array(f) for f in factors]


def procrustes_rotation(X, Y):
    """Orthogonal Q* minimising ||X @ Q - Y||_F  via  SVD(X^T @ Y)."""
    U, _, Wh = np.linalg.svd(X.T @ Y)
    return U @ Wh


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_ref_factors = None
_N = None
_R = None
_mu_factor = None
_n_iter_max = None


def _worker_init(ref_factors, N, R, mu_factor, n_iter_max):
    global _ref_factors, _N, _R, _mu_factor, _n_iter_max
    _ref_factors, _N, _R, _mu_factor, _n_iter_max = \
        ref_factors, N, R, mu_factor, n_iter_max


def _process_one(args):
    i, mesh_id, mesh_path = args
    try:
        raw = trimesh.load(mesh_path, force='mesh')
        mesh = normalize_mesh(raw)
        sdf, voxel_size = compute_sdf(mesh, N=_N)
        tsdf = truncate_sdf(sdf, _mu_factor * voxel_size)

        core, factors = tucker_decompose(tsdf.astype(np.float64),
                                         [_R, _R, _R], _n_iter_max)

        Qs, aligned = [], []
        for U, U_ref in zip(factors, _ref_factors):
            Q = procrustes_rotation(U, U_ref)
            Qs.append(Q)
            aligned.append(U @ Q)
        U_1, U_2, U_3 = aligned
        Q_1, Q_2, Q_3 = Qs
        C_aligned = np.einsum('abc,ai,bj,ck->ijk', core, Q_1, Q_2, Q_3)

        return (i, mesh_id, mesh_path,
                U_1.astype(np.float32), U_2.astype(np.float32),
                U_3.astype(np.float32), C_aligned.astype(np.float32),
                float(voxel_size), None)
    except Exception as e:
        import traceback
        return (i, mesh_id, mesh_path, None, None, None, None, None,
                f'{type(e).__name__}: {e}\n{traceback.format_exc()}')


# ---------------------------------------------------------------------------
# Shard I/O
# ---------------------------------------------------------------------------

def save_shard(shard_idx, buf, out_dir, N, R, mu_factor):
    path = os.path.join(out_dir, f'tucker_sdf_shard_{shard_idx:04d}.pt')
    torch.save({
        'U_1':        torch.from_numpy(np.stack(buf['U_1'])),   # (B, N, R)
        'U_2':        torch.from_numpy(np.stack(buf['U_2'])),   # (B, N, R)
        'U_3':        torch.from_numpy(np.stack(buf['U_3'])),   # (B, N, R)
        'C':          torch.from_numpy(np.stack(buf['C'])),     # (B, R, R, R)
        'mesh_id':    buf['mesh_id'],
        'mesh_path':  buf['mesh_path'],
        'voxel_size': torch.tensor(buf['voxel_size'], dtype=torch.float32),
        'N': N, 'rank': R, 'mu_factor': mu_factor,
    }, path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_list',  required=True)
    parser.add_argument('--out_dir',    required=True)
    parser.add_argument('--N',          type=int, default=128)
    parser.add_argument('--rank',       type=int, default=24)
    parser.add_argument('--mu_factor',  type=float, default=2.0)
    parser.add_argument('--n_max',      type=int, default=None)
    parser.add_argument('--shard_size', type=int, default=50)
    parser.add_argument('--seed',       type=int, default=REF_SEED)
    parser.add_argument('--n_workers',  type=int, default=os.cpu_count())
    parser.add_argument('--n_iter_max', type=int, default=100)
    parser.add_argument('--anchor_path', type=str, default=None,
                        help='If set, skip anchor computation and load from here')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.mesh_list) as f:
        meshes = json.load(f)['pilot']
    n_total = len(meshes) if args.n_max is None else min(len(meshes), args.n_max)
    print(f'Mesh list: {args.mesh_list}  ({len(meshes)} total, processing {n_total})')
    print(f'N={args.N}  rank={args.rank}  mu_factor={args.mu_factor}  '
          f'shard_size={args.shard_size}  workers={args.n_workers}')
    print(f'Output: {os.path.abspath(args.out_dir)}\n')

    # ── Anchor ────────────────────────────────────────────────────────────
    if args.anchor_path is not None:
        print(f'Loading anchor from {args.anchor_path}')
        ref = torch.load(args.anchor_path, map_location='cpu', weights_only=False)
        ref_factors = [ref[k].numpy().astype(np.float64)
                       for k in ['U_1_ref', 'U_2_ref', 'U_3_ref']]
        assert ref['N'] == args.N, f'Anchor N {ref["N"]} != requested {args.N}'
        assert ref['rank'] == args.rank, f'Anchor rank {ref["rank"]} != requested {args.rank}'
    else:
        rng = random.Random(args.seed)
        ref_idx = rng.randrange(n_total)
        ref_entry = meshes[ref_idx]
        print(f'Reference mesh: idx={ref_idx}  id={ref_entry["id"]}  '
              f'path={ref_entry["mesh_path"]}  (seed={args.seed})')
        t0 = time.time()
        raw = trimesh.load(ref_entry['mesh_path'], force='mesh')
        ref_mesh = normalize_mesh(raw)
        ref_sdf, ref_voxel = compute_sdf(ref_mesh, N=args.N)
        ref_tsdf = truncate_sdf(ref_sdf, args.mu_factor * ref_voxel)
        ref_core, ref_factors = tucker_decompose(
            ref_tsdf.astype(np.float64), [args.rank]*3, args.n_iter_max)
        print(f'  Anchor Tucker in {time.time()-t0:.1f}s  '
              f'U_k shapes: {[f.shape for f in ref_factors]}  '
              f'C: {ref_core.shape}\n')

        anchor_path = os.path.join(args.out_dir, 'ref_anchor.pt')
        torch.save({
            'U_1_ref':     torch.from_numpy(ref_factors[0].astype(np.float32)),
            'U_2_ref':     torch.from_numpy(ref_factors[1].astype(np.float32)),
            'U_3_ref':     torch.from_numpy(ref_factors[2].astype(np.float32)),
            'C_ref':       torch.from_numpy(ref_core.astype(np.float32)),
            'ref_mesh_id': ref_entry['id'],
            'ref_idx':     ref_idx,
            'voxel_size':  float(ref_voxel),
            'N': args.N, 'rank': args.rank, 'mu_factor': args.mu_factor,
        }, anchor_path)
        print(f'Anchor saved → {anchor_path}\n')

    # ── Process all samples ───────────────────────────────────────────────
    def _empty_buf():
        return {k: [] for k in ('U_1','U_2','U_3','C',
                                'mesh_id','mesh_path','voxel_size')}
    buf = _empty_buf()
    shard_idx_out = 0
    saved_shards  = []
    failures      = []

    tasks = [(i, meshes[i]['id'], meshes[i]['mesh_path'])
             for i in range(n_total)]
    chunksize = max(1, min(10, n_total // max(1, args.n_workers * 4)))

    t_start = time.time()
    pbar = tqdm(total=n_total, desc='Tucker+Align', unit='sample',
                dynamic_ncols=True)

    with Pool(processes=args.n_workers, initializer=_worker_init,
              initargs=(ref_factors, args.N, args.rank,
                        args.mu_factor, args.n_iter_max)) as pool:
        for result in pool.imap_unordered(_process_one, tasks,
                                          chunksize=chunksize):
            i, mesh_id, mesh_path, U_1, U_2, U_3, C, voxel, err = result
            if err is not None:
                failures.append({'mesh_id': mesh_id, 'mesh_path': mesh_path,
                                 'error': err})
                pbar.update(1)
                continue
            buf['U_1'].append(U_1);  buf['U_2'].append(U_2)
            buf['U_3'].append(U_3);  buf['C'].append(C)
            buf['mesh_id'].append(mesh_id)
            buf['mesh_path'].append(mesh_path)
            buf['voxel_size'].append(voxel)
            if len(buf['U_1']) >= args.shard_size:
                p = save_shard(shard_idx_out, buf, args.out_dir,
                               args.N, args.rank, args.mu_factor)
                saved_shards.append(p)
                shard_idx_out += 1
                buf = _empty_buf()
            pbar.update(1)

    pbar.close()

    if buf['U_1']:
        p = save_shard(shard_idx_out, buf, args.out_dir,
                       args.N, args.rank, args.mu_factor)
        saved_shards.append(p)

    elapsed = time.time() - t_start
    n_ok = n_total - len(failures)
    print(f'\n{n_ok}/{n_total} OK  ({len(failures)} failed)  '
          f'in {elapsed:.1f}s ({elapsed/n_total*1000:.1f} ms/sample)')
    print(f'{len(saved_shards)} shards → {args.out_dir}')

    if failures:
        fail_path = os.path.join(args.out_dir, 'failures.jsonl')
        with open(fail_path, 'w') as f:
            for r in failures:
                f.write(json.dumps(r) + '\n')
        print(f'Failures logged → {fail_path}')

    manifest = os.path.join(args.out_dir, 'manifest.txt')
    with open(manifest, 'w') as f:
        for p in saved_shards:
            f.write(os.path.basename(p) + '\n')
    print(f'Manifest → {manifest}')

    # ── Sanity check on first shard, first entry ──────────────────────────
    print('\n─── Sanity check (first shard, first entry) ───')
    s = torch.load(saved_shards[0], map_location='cpu', weights_only=False)
    U_1 = s['U_1'][0].numpy().astype(np.float64)
    U_2 = s['U_2'][0].numpy().astype(np.float64)
    U_3 = s['U_3'][0].numpy().astype(np.float64)
    C   = s['C'][0].numpy().astype(np.float64)
    print(f'  shapes  U_1 {U_1.shape}  U_2 {U_2.shape}  U_3 {U_3.shape}  C {C.shape}')
    for name, U in [('U_1', U_1), ('U_2', U_2), ('U_3', U_3)]:
        orth = np.linalg.norm(U.T @ U - np.eye(U.shape[1]))
        print(f'  ||{name}^T {name} - I|| = {orth:.2e}')

    # Recompute original TSDF and compare to Tucker reconstruction
    mesh_path = s['mesh_path'][0]
    voxel = float(s['voxel_size'][0])
    raw = trimesh.load(mesh_path, force='mesh')
    orig_mesh = normalize_mesh(raw)
    orig_sdf, _ = compute_sdf(orig_mesh, N=args.N)
    orig_tsdf = truncate_sdf(orig_sdf, args.mu_factor * voxel)
    recon = np.einsum('abc,ia,jb,kc->ijk', C, U_1, U_2, U_3)
    mse  = float(np.mean((orig_tsdf - recon) ** 2))
    rel  = (float(np.linalg.norm(orig_tsdf - recon))
            / max(float(np.linalg.norm(orig_tsdf)), 1e-12))
    inter = np.logical_and(orig_tsdf < 0, recon < 0).sum()
    union = np.logical_or (orig_tsdf < 0, recon < 0).sum()
    iou  = float(inter / max(union, 1))
    print(f'  Recon  MSE={mse:.6f}  RelErr={rel:.4f}  IoU(<0)={iou:.4f}  '
          f'mesh_id={s["mesh_id"][0]}')


if __name__ == '__main__':
    main()
