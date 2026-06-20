"""
sample.py - DDIM (DDPM ckpt) or ODE (SI ckpt) sampler for v4 / v7.

Reconstructs SDF via X = G x_2 U_2 x_3 U_3, where G has shape (n_1, r_2, r_3).

Auto-detects mode from ckpt['si']:
  - 'none' (or missing): DDIM sampler (DDPM ε/v parameterization).
  - 'linear' / 'trig'  : ODE sampler integrating dx/dt = b_θ(x, t) from t=1 → t=0.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

from diffats.models import SDFDiTConfig, SDFDiT, SDFDiTWrapper
from diffats.utils import extract_mesh


def make_schedule(T=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {'alpha_bar': alpha_bar}


@torch.no_grad()
def _cfg_predict(model, x, t_int, cond, cfg_scale):
    """Classifier-free guidance: blend cond and uncond predictions."""
    if cond is None or cfg_scale == 1.0:
        return model(x, t_int, cond=cond)
    # Concatenate batch for cond + uncond pass in single forward
    x2 = torch.cat([x, x], dim=0)
    t2 = torch.cat([t_int, t_int], dim=0)
    zero_cond = torch.zeros_like(cond)
    cond2 = torch.cat([cond, zero_cond], dim=0)
    pred2 = model(x2, t2, cond=cond2)
    pred_c, pred_u = pred2.chunk(2, dim=0)
    return pred_u + cfg_scale * (pred_c - pred_u)


@torch.no_grad()
def si_sample(model, F, T_scale, n_steps, device, n_samples=1,
              schedule='trig', solver='heun', log_every=10,
              cond=None, cfg_scale=1.0):
    """Stochastic Interpolant ODE sampling with optional CFG.

    Integrates dx/dt = b_θ(x, t) from t=1 (noise) to t=0 (data).
    schedule = 'linear' (α=1-t, β=t) or 'trig' (α=cos(πt/2), β=sin(πt/2)).
    solver   = 'euler' (1st order) or 'heun' (2nd order).
    cfg_scale = 1.0 disables CFG; >1.0 amplifies condition signal.
    """
    x = torch.randn(n_samples, F, device=device)
    dt = -1.0 / n_steps  # negative: integrating backward in t
    t = torch.full((n_samples,), 1.0, device=device)
    for k in range(n_steps):
        t_int = (t.clamp(min=1e-4, max=1.0 - 1e-4) * (T_scale - 1)).long()
        if solver == 'euler':
            v = _cfg_predict(model, x, t_int, cond, cfg_scale)
            x = x + dt * v
        else:  # 'heun' — predictor-corrector (Ralston)
            v1 = _cfg_predict(model, x, t_int, cond, cfg_scale)
            x_pred = x + dt * v1
            t_next = (t + dt).clamp(min=1e-4, max=1.0 - 1e-4)
            t_next_int = (t_next * (T_scale - 1)).long()
            v2 = _cfg_predict(model, x_pred, t_next_int, cond, cfg_scale)
            x = x + 0.5 * dt * (v1 + v2)
        t = t + dt
        if k % log_every == 0 or k == n_steps - 1:
            print(f'  k={k:3d}/{n_steps}  t={t[0].item():+.4f}  '
                  f'x.abs.mean={x.abs().mean().item():.4f}', flush=True)
    return x


@torch.no_grad()
def ddim_sample(model, F, sched, T, n_steps, device, n_samples=1, target='eps',
                log_every=10):
    x = torch.randn(n_samples, F, device=device)
    alpha_bar = sched['alpha_bar']
    ts = torch.linspace(T - 1, 0, n_steps + 1, device=device).long()
    for k in range(n_steps):
        t_cur, t_nxt = int(ts[k]), int(ts[k + 1])
        t_tensor = torch.full((n_samples,), t_cur, dtype=torch.long, device=device)
        pred = model(x, t_tensor)
        a_cur = alpha_bar[t_cur]
        a_nxt = alpha_bar[t_nxt] if t_nxt >= 0 else torch.tensor(1.0, device=device)
        sqrt_a_cur     = torch.sqrt(a_cur)
        sqrt_1m_a_cur  = torch.sqrt(1.0 - a_cur)
        if target == 'v':
            x0_pred  = sqrt_a_cur * x - sqrt_1m_a_cur * pred
            eps_pred = sqrt_1m_a_cur * x + sqrt_a_cur * pred
        else:
            eps_pred = pred
            x0_pred  = (x - sqrt_1m_a_cur * pred) / sqrt_a_cur
        x = torch.sqrt(a_nxt) * x0_pred + torch.sqrt(1.0 - a_nxt) * eps_pred
        if k % log_every == 0 or k == n_steps - 1:
            print(f'  k={k:3d}/{n_steps}  t={t_cur:4d}->{t_nxt:<4d}  '
                  f'x.abs.mean={x.abs().mean().item():.4f}  '
                  f'x0.abs.mean={x0_pred.abs().mean().item():.4f}', flush=True)
    return x


def unnormalize(x_flat, N, R, stats):
    sG  = N * R * R
    sU2 = N * R
    sU3 = N * R
    c0, c1, c2 = x_flat.split([sG, sU2, sU3], dim=1)
    G  = c0.reshape(-1, N, R, R) * stats['G_std']  + stats['G_mean']
    U2 = c1.reshape(-1, N, R)    * stats['U2_std'] + stats['U2_mean']
    U3 = c2.reshape(-1, N, R)    * stats['U3_std'] + stats['U3_mean']
    return G, U2, U3


def reconstruct_sdf(G, U2, U3):
    """X[n,j,k] = sum_{q,r} G[n,q,r] U_2[j,q] U_3[k,r]."""
    return torch.einsum('bnqr,bjq,bkr->bnjk', G, U2, U3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',     required=True)
    p.add_argument('--out_dir',  default='outputs/samples_v4')
    p.add_argument('--n',        type=int, default=4)
    p.add_argument('--T',        type=int, default=1000)
    p.add_argument('--n_steps',  type=int, default=250,
                   help='DDIM steps (DDPM) OR ODE steps (SI). SI default 50.')
    p.add_argument('--voxel',    type=float, default=2.0/255)
    p.add_argument('--seed',     type=int, default=42)
    p.add_argument('--solver',   default='heun', choices=['euler', 'heun'],
                   help='ODE solver for SI mode (ignored for DDPM).')
    p.add_argument('--si_override', default=None, choices=['none', 'linear', 'trig'],
                   help='Force a sampling mode (default: read from ckpt).')
    p.add_argument('--clip_tsdf', action='store_true',
                   help='Clip reconstructed SDF to [-mu*voxel, +mu*voxel] '
                        'before marching cubes (matches training truncation).')
    p.add_argument('--mu_factor', type=float, default=2.0,
                   help='TSDF truncation factor (default 2.0 = training-time).')
    p.add_argument('--qr_correct', action='store_true',
                   help='QR-orthonormalize U_2 and U_3 columns BEFORE TSDF '
                        'reconstruction. Tests if generated factors lost orthonormality.')
    p.add_argument('--use_ema', action='store_true',
                   help='Use EMA weights for sampling (if present in ckpt).')
    # --- v9 conditional sampling ---
    p.add_argument('--cond_bundle', default='',
                   help='Path to bundle.pt with {embs, mesh_ids} for cond sampling.')
    p.add_argument('--cond_mesh_ids', default='',
                   help='Comma-separated mesh_ids (e.g. 00000,00001,...). '
                        'Must exist in cond_bundle.')
    p.add_argument('--cfg_scale', type=float, default=1.0,
                   help='Classifier-free guidance scale (1.0 = off, 2-4 typical).')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f'Loading checkpoint: {args.ckpt}')
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg_d = ck['cfg']
    cfg = SDFDiTConfig(**cfg_d)
    stats = ck['stats']
    print(f'  cfg: {cfg}')
    print(f'  stats: {stats}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  device: {device}')

    model = SDFDiTWrapper(SDFDiT(cfg))
    if args.use_ema and 'ema_state_dict' in ck:
        model.load_state_dict(ck['ema_state_dict'])
        print(f'  loaded EMA weights')
    else:
        model.load_state_dict(ck['state_dict'])
        if args.use_ema:
            print(f'  WARN: --use_ema requested but ckpt has no ema_state_dict; using raw')
    model.eval().to(device)
    print(f'  params: {sum(pp.numel() for pp in model.parameters()):,}')

    F = model.core.flat_total
    target = ck.get('target', 'eps')

    # Load conditional embeddings if requested (v9 image-to-3D mode)
    cond_tensor = None
    cond_mesh_ids = []
    if args.cond_bundle and args.cond_mesh_ids:
        print(f'\n[cond] loading bundle: {args.cond_bundle}')
        b = torch.load(args.cond_bundle, map_location='cpu', weights_only=False)
        all_ids = b['mesh_ids']
        id_to_idx = {m: i for i, m in enumerate(all_ids)}
        cond_mesh_ids = args.cond_mesh_ids.split(',')
        idx = torch.tensor([id_to_idx[m] for m in cond_mesh_ids], dtype=torch.long)
        cond_tensor = b['embs'].index_select(0, idx).to(device).float()
        print(f'[cond] tensor shape: {tuple(cond_tensor.shape)}  '
              f'cfg_scale={args.cfg_scale}')
        # override n to match cond_mesh_ids count
        args.n = len(cond_mesh_ids)

    # Decide DDPM (DDIM) vs SI (ODE) mode
    si_mode = args.si_override if args.si_override is not None else ck.get('si', 'none')
    print(f'\nSampling {args.n} shapes  (mode={si_mode}, F={F})')
    t0 = time.time()
    if si_mode == 'none':
        sched = make_schedule(args.T, device=device)
        print(f'  DDIM {args.n_steps} steps from T={args.T}, target={target}')
        x = ddim_sample(model, F, sched, args.T, args.n_steps, device,
                        n_samples=args.n, target=target, log_every=25)
    else:
        n_steps_si = args.n_steps if args.n_steps != 250 else 50
        print(f'  SI {si_mode}, ODE {args.solver}, {n_steps_si} steps')
        x = si_sample(model, F, args.T, n_steps_si, device,
                      n_samples=args.n, schedule=si_mode,
                      cond=cond_tensor, cfg_scale=args.cfg_scale,
                      solver=args.solver, log_every=5)
    print(f'  sample wall: {time.time()-t0:.1f}s')

    G, U2, U3 = unnormalize(x, cfg.N, cfg.R, stats)
    print(f'  G  stats: mean={G.mean():.4f} std={G.std():.4f}')
    print(f'  U2 stats: mean={U2.mean():.4f} std={U2.std():.4f}')
    print(f'  U3 stats: mean={U3.mean():.4f} std={U3.std():.4f}')

    if args.qr_correct:
        # Diagnostic: how far are generated U_2/U_3 from orthonormal manifold?
        print('\n[qr_correct] Per-sample factor deviation from orthonormality:')
        I = torch.eye(cfg.R, device=U2.device)
        for i in range(min(4, args.n)):
            d2 = (U2[i].T @ U2[i] - I).norm().item()
            d3 = (U3[i].T @ U3[i] - I).norm().item()
            n2 = U2[i].norm(dim=0)
            print(f'  sample {i}: ||U2^T U2 - I||={d2:.4f}  ||U3^T U3 - I||={d3:.4f}  '
                  f'U2 col-norms min/max=[{n2.min():.3f}, {n2.max():.3f}]')
        # Project to orthonormal manifold by replacing U with Q from QR.
        # This DROPS the R factor → SDF reconstruction CHANGES (intended).
        # Training factors were orthonormal (HOOI); if model generates non-ortho
        # factors, magnitude drift comes from inflated column norms / cross-correlations.
        U2 = torch.linalg.qr(U2)[0]
        U3 = torch.linalg.qr(U3)[0]
        d2_post = (U2[0].T @ U2[0] - I).norm().item()
        d3_post = (U3[0].T @ U3[0] - I).norm().item()
        print(f'  Post-QR ||U2^T U2 - I||={d2_post:.6f}  ||U3^T U3 - I||={d3_post:.6f}')
        print(f'  (G unchanged; SDF magnitude will adjust to new factor scale)')

    sdf = reconstruct_sdf(G, U2, U3).cpu().numpy()
    print(f'\nReconstructed SDF: {sdf.shape}  '
          f'  min={sdf.min():.4f}  max={sdf.max():.4f}  mean={sdf.mean():.4f}')

    if args.clip_tsdf:
        mu_h = args.mu_factor * args.voxel
        print(f'\n[clip_tsdf] Truncating to ±{mu_h:.5f} '
              f'(mu_factor={args.mu_factor}, voxel={args.voxel:.5f})')
        n_clipped = int(((sdf < -mu_h) | (sdf > mu_h)).sum())
        n_total = sdf.size
        print(f'  fraction out-of-range pre-clip: {n_clipped/n_total*100:.2f}%')
        sdf = np.clip(sdf, -mu_h, mu_h)
        print(f'  after clip: min={sdf.min():.5f}, max={sdf.max():.5f}')

    summary = []
    for i in range(args.n):
        s = sdf[i]
        in_frac = float((s < 0).mean())
        print(f'\n[sample {i}] inside_frac={in_frac:.4%}  '
              f'sdf range=[{s.min():.4f}, {s.max():.4f}]')
        if in_frac < 1e-4 or in_frac > 0.999:
            print(f'  -> degenerate (all outside or all inside), skipping mesh')
            summary.append((i, in_frac, None, None))
            continue
        mesh = extract_mesh(s, args.voxel, bbox_min=-1.0)
        if mesh is None:
            print(f'  -> marching cubes failed')
            summary.append((i, in_frac, None, None))
            continue
        # If conditioning on specific mesh_ids, name outputs by mesh_id (predicted)
        if cond_mesh_ids:
            out_path = os.path.join(args.out_dir, f'pred_{cond_mesh_ids[i]}.obj')
        else:
            out_path = os.path.join(args.out_dir, f'sample_{i:02d}.obj')
        mesh.export(out_path)
        nV, nF = len(mesh.vertices), len(mesh.faces)
        print(f'  -> mesh: V={nV}  F={nF}  saved {out_path}')
        summary.append((i, in_frac, nV, nF))

    print('\n=== summary ===')
    for i, ifrac, nV, nF in summary:
        if nV is None:
            print(f'sample {i}: inside={ifrac:.2%}  DEGENERATE')
        else:
            print(f'sample {i}: inside={ifrac:.2%}  V={nV:>10}  F={nF:>10}')


if __name__ == '__main__':
    main()
