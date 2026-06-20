"""
realign_medoid.py — Re-align random-anchor Tucker shards to medoid anchor.

Mathematically equivalent to redoing Tucker preprocessing with medoid anchor
(up to a uniform global rotation, irrelevant to diffusion).

Per the DiffATs paper (Sec 4.4, l=2 TGP), only modes k in {2, 3} are aligned.
Mode 1 is left unchanged; its information is absorbed into G at training time
(G = C ×_1 U_1).

Pipeline:
  1. Load all (C, U_1, U_2, U_3) from input shards (random-anchor aligned).
  2. Find medoid index per mode k in {2, 3}, using
       i_k* = argmax_i sum_j ||U_{k,i}^T U_{k,j}||_F^2
     This index is invariant to which anchor U is aligned to (Frobenius norm is
     invariant under orthogonal right-rotation).
  3. Take anchor_k = U_{k,i_k*} (the aligned-version of medoid sample's factor)
  4. For each sample: Q_k = OP(U_k, anchor_k), then  Ũ_k = U_k · Q_k
  5. Update core: C_new = C ×_2 Q_2^T ×_3 Q_3^T (so X reconstruction is preserved)
  6. Save new shards in identical (C, U_1, U_2, U_3) format.

Sanity checks:
  - Reconstruction X = C ×_1 U_1 ×_2 U_2 ×_3 U_3 unchanged before/after re-align.
  - Average Procrustes distance ||U_i Q - U_anchor||_F is < random-anchor version.
"""

import argparse
import glob
import os
import time

import torch


# ───────────────────── I/O ─────────────────────

def load_all_shards(in_dirs):
    if isinstance(in_dirs, str):
        in_dirs = [in_dirs]
    shard_files = []
    for d in in_dirs:
        shard_files.extend(sorted(glob.glob(os.path.join(d, 'tucker_sdf_shard_*.pt'))))
    if not shard_files:
        raise FileNotFoundError(f'no shards in {in_dirs}')
    samples = []
    seen_ids = set()
    N = R = None
    for sf in shard_files:
        s = torch.load(sf, map_location='cpu', weights_only=False)
        N = N or s['N']
        R = R or s['rank']
        B = s['U_1'].shape[0]
        mesh_ids = s.get('mesh_id', [None] * B)
        for i in range(B):
            mid = mesh_ids[i] if i < len(mesh_ids) else None
            if mid is not None and mid in seen_ids:
                continue
            if mid is not None:
                seen_ids.add(mid)
            samples.append({
                'C':   s['C'][i].float(),
                'U_1': s['U_1'][i].float(),
                'U_2': s['U_2'][i].float(),
                'U_3': s['U_3'][i].float(),
                'mesh_id':    mid,
                'mesh_path':  s['mesh_path'][i] if 'mesh_path' in s else '',
                'voxel_size': s['voxel_size'][i].item() if 'voxel_size' in s else 0.0,
            })
    return samples, N, R


# ───────────────────── medoid finder ─────────────────────

def compute_medoid_idx(U_all, device='cuda', chunk=200):
    """
    Find i* = argmax_i sum_j ||U_i^T U_j||_F^2.

    Identity: ||U_i^T U_j||_F^2 = <U_i U_i^T, U_j U_j^T>_F
              sum_j M[i,j] = <U_i U_i^T, S>_F  where S = sum_j U_j U_j^T.
              score_i = tr(U_i^T S U_i)

    Returns (i_star, scores_per_sample).
    """
    M, N, R = U_all.shape
    U_dev = U_all.to(device)
    S = torch.zeros(N, N, device=device, dtype=U_dev.dtype)
    for start in range(0, M, chunk):
        Uc = U_dev[start:start + chunk]
        S += (Uc @ Uc.transpose(-1, -2)).sum(dim=0)

    SU = torch.einsum('nm,bmr->bnr', S, U_dev)
    scores = (U_dev * SU).sum(dim=(-1, -2))
    return int(scores.argmax().item()), scores.cpu()


# ───────────────────── OP align ─────────────────────

def op_align(U, U_anchor):
    """
    OP align U (B, N, R) to U_anchor (N, R).

    Solution Q* = L V^T  where  U^T U_anchor = L S V^T  (SVD).
    Returns (U_new = U @ Q*, Q*).
    """
    M = torch.einsum('bnr,ns->brs', U, U_anchor)
    Lc, _, Vh = torch.linalg.svd(M)               # M = Lc · diag(S) · Vh, Vh = V^T
    Q = torch.einsum('brk,bks->brs', Lc, Vh)      # L V^T
    U_new = torch.einsum('bnr,brs->bns', U, Q)
    return U_new, Q


# ───────────────────── main ─────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir',  required=True, nargs='+')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--shard_size', type=int, default=100)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    if device.type == 'cuda':
        print(f'  gpu: {torch.cuda.get_device_name(0)}')

    t0 = time.time()
    print(f'Loading shards from: {args.in_dir}')
    samples, N, R = load_all_shards(args.in_dir)
    M_n = len(samples)
    print(f'  loaded {M_n} unique samples (N={N}, R={R})  in {time.time()-t0:.1f}s')

    U_1_all = torch.stack([s['U_1'] for s in samples], dim=0)  # (M, N, R)
    U_2_all = torch.stack([s['U_2'] for s in samples], dim=0)
    U_3_all = torch.stack([s['U_3'] for s in samples], dim=0)
    C_all   = torch.stack([s['C']   for s in samples], dim=0)  # (M, R, R, R)

    # ── Medoid per mode ─────────────────────────────────────
    print('\n== Medoid selection ==')
    t1 = time.time()
    idx_2, scores_2 = compute_medoid_idx(U_2_all, device=device)
    print(f'  medoid mode 2: idx={idx_2}  mesh_id={samples[idx_2]["mesh_id"]}  '
          f'  score=[{scores_2.min():.2f}, mean={scores_2.mean():.2f}, '
          f'max={scores_2.max():.2f}]   ({time.time()-t1:.1f}s)')
    t1 = time.time()
    idx_3, scores_3 = compute_medoid_idx(U_3_all, device=device)
    print(f'  medoid mode 3: idx={idx_3}  mesh_id={samples[idx_3]["mesh_id"]}  '
          f'  score=[{scores_3.min():.2f}, mean={scores_3.mean():.2f}, '
          f'max={scores_3.max():.2f}]   ({time.time()-t1:.1f}s)')

    anchor_2 = U_2_all[idx_2].to(device)
    anchor_3 = U_3_all[idx_3].to(device)

    # ── OP align ────────────────────────────────────────────
    print('\n== OP align (mode 2, mode 3) ==')
    t1 = time.time()
    U_2_dev = U_2_all.to(device)
    U_3_dev = U_3_all.to(device)
    C_dev   = C_all.to(device)

    U_2_new, Q_2 = op_align(U_2_dev, anchor_2)
    U_3_new, Q_3 = op_align(U_3_dev, anchor_3)
    # Update core: C_new = C ×_2 Q_2^T ×_3 Q_3^T  (so X recon preserved)
    C_new = torch.einsum('Babc,Bbd,Bce->Bade', C_dev, Q_2, Q_3)
    print(f'  done in {time.time()-t1:.1f}s')

    # Procrustes-distance diagnostic
    def avg_proc_dist(U, A):
        diff = U - A.unsqueeze(0)                # (B, N, R)
        return diff.flatten(1).norm(dim=1).mean().item()

    d_pre  = avg_proc_dist(U_2_dev, anchor_2)
    d_post = avg_proc_dist(U_2_new, anchor_2)
    print(f'  mode 2:  avg ||U - anchor||_F   pre-OP={d_pre:.4f}   post-OP={d_post:.4f}')
    d_pre  = avg_proc_dist(U_3_dev, anchor_3)
    d_post = avg_proc_dist(U_3_new, anchor_3)
    print(f'  mode 3:  avg ||U - anchor||_F   pre-OP={d_pre:.4f}   post-OP={d_post:.4f}')
    assert d_post < d_pre + 1e-4, 'OP did not decrease distance — bug!'

    # ── Reconstruction sanity ──────────────────────────────
    print('\n== Reconstruction sanity (sample 0) ==')
    U1_0 = U_1_all[0].to(device)
    X_old = torch.einsum('abc,ia,jb,kc->ijk',
                         C_dev[0],  U1_0, U_2_dev[0], U_3_dev[0])
    X_new = torch.einsum('abc,ia,jb,kc->ijk',
                         C_new[0],  U1_0, U_2_new[0], U_3_new[0])
    rel_err = (X_old - X_new).norm() / X_old.norm()
    print(f'  ||X_old - X_new||_F / ||X_old||_F = {rel_err:.2e}')
    assert rel_err.item() < 1e-4, \
        f'reconstruction error after re-align too large: {rel_err}'

    # Spot-check a few more samples
    for i in (1, M_n // 2, M_n - 1):
        U1_i = U_1_all[i].to(device)
        X_o = torch.einsum('abc,ia,jb,kc->ijk',
                           C_dev[i],  U1_i, U_2_dev[i], U_3_dev[i])
        X_n = torch.einsum('abc,ia,jb,kc->ijk',
                           C_new[i],  U1_i, U_2_new[i], U_3_new[i])
        re = (X_o - X_n).norm() / X_o.norm()
        print(f'  sample {i}: rel err = {re:.2e}')
        assert re.item() < 1e-4

    U_2_new = U_2_new.cpu()
    U_3_new = U_3_new.cpu()
    C_new   = C_new.cpu()

    # ── Save shards (same format as original Tucker preprocessing) ─────
    print(f'\nSaving shards to {args.out_dir}')
    n_shards = (M_n + args.shard_size - 1) // args.shard_size
    for k in range(n_shards):
        i0 = k * args.shard_size
        i1 = min(i0 + args.shard_size, M_n)
        buf = {
            'U_1':   U_1_all[i0:i1],
            'U_2':   U_2_new[i0:i1],
            'U_3':   U_3_new[i0:i1],
            'C':     C_new[i0:i1],
            'mesh_id':    [samples[i]['mesh_id']    for i in range(i0, i1)],
            'mesh_path':  [samples[i]['mesh_path']  for i in range(i0, i1)],
            'voxel_size': torch.tensor(
                [samples[i]['voxel_size'] for i in range(i0, i1)],
                dtype=torch.float32),
            'N': N, 'rank': R, 'mu_factor': 2.0,
            'anchor_mode_2_idx':     idx_2,
            'anchor_mode_2_mesh_id': samples[idx_2]['mesh_id'],
            'anchor_mode_3_idx':     idx_3,
            'anchor_mode_3_mesh_id': samples[idx_3]['mesh_id'],
        }
        out_path = os.path.join(args.out_dir, f'tucker_sdf_shard_{k:04d}.pt')
        torch.save(buf, out_path)
    print(f'  saved {M_n} samples in {n_shards} shards')

    torch.save({
        'idx_2': idx_2, 'mesh_id_2': samples[idx_2]['mesh_id'],
        'idx_3': idx_3, 'mesh_id_3': samples[idx_3]['mesh_id'],
        'scores_2': scores_2, 'scores_3': scores_3,
        'M': M_n, 'N': N, 'R': R,
    }, os.path.join(args.out_dir, 'medoid_info.pt'))
    print(f'\nDONE in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
