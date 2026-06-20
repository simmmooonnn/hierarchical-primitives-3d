"""
train_sdf_dit_v4.py - Paper-faithful DiffATs training: l=2 TGP + AdaLN-Zero DiT.

Shards still store (U_1, U_2, U_3, C) per-sample (already OP-aligned to anchor).
At load time we form G = C x_1 U_1 -> shape (B, N, R, R) and DISCARD U_1.
Diffusion is then on (G, U_2, U_3) per the paper's l=2 TGP.
"""

import argparse
import copy
import glob
import math
import os
import sys
import time

import numpy as np
import torch

from diffats.models import SDFDiTConfig, SDFDiT, SDFDiTWrapper


def load_shards(shard_dirs):
    if isinstance(shard_dirs, str):
        shard_dirs = [shard_dirs]
    shard_files = []
    for d in shard_dirs:
        shard_files.extend(sorted(glob.glob(os.path.join(d, 'tucker_sdf_shard_*.pt'))))
    if not shard_files:
        raise FileNotFoundError(f'no shards in {shard_dirs}')
    U1, U2, U3, C = [], [], [], []
    kept_mesh_ids = []
    seen_ids = set()
    N = R = None
    for sf in shard_files:
        s = torch.load(sf, map_location='cpu', weights_only=False)
        N = N or s['N']
        R = R or s['rank']
        mesh_ids = s.get('mesh_id', [None] * s['U_1'].shape[0])
        keep = [i for i, mid in enumerate(mesh_ids) if mid is None or mid not in seen_ids]
        for i in keep:
            kept_mesh_ids.append(mesh_ids[i])
        for mid in mesh_ids:
            if mid is not None:
                seen_ids.add(mid)
        U1.append(s['U_1'].float()[keep])
        U2.append(s['U_2'].float()[keep])
        U3.append(s['U_3'].float()[keep])
        C .append(s['C'  ].float()[keep])
    U1 = torch.cat(U1, dim=0)        # (B, N, R)
    U2 = torch.cat(U2, dim=0)        # (B, N, R)
    U3 = torch.cat(U3, dim=0)        # (B, N, R)
    C  = torch.cat(C,  dim=0)        # (B, R, R, R)
    return U1, U2, U3, C, N, R, kept_mesh_ids


def load_embeddings(emb_dir, mesh_ids, expected_shape):
    """Pre-load DINOv2 embeddings, ordered to match mesh_ids of the latent shards.

    Two layouts auto-detected:
      1) bundle .pt file: contains {'embs': (N, ...), 'mesh_ids': [...]}.
         Fast (~seconds); built by bundle_embeddings.py.
      2) directory of per-mesh .pt files (slow ~10 min for 10k files).
    """
    n = len(mesh_ids)
    bundle_path = (emb_dir if emb_dir.endswith('.pt')
                   else os.path.join(emb_dir, 'bundle.pt'))
    if os.path.isfile(bundle_path):
        print(f'  [load_embeddings] using bundle: {bundle_path}')
        b = torch.load(bundle_path, map_location='cpu', weights_only=False)
        index = {m: i for i, m in enumerate(b['mesh_ids'])}
        missing = [m for m in mesh_ids if m not in index]
        if missing:
            raise FileNotFoundError(f'{len(missing)} embeddings missing, e.g. {missing[:5]}')
        idx = torch.tensor([index[m] for m in mesh_ids], dtype=torch.long)
        embs = b['embs'].index_select(0, idx).contiguous()
        if embs.dtype != torch.float16:
            embs = embs.to(torch.float16)
        return embs
    print(f'  [load_embeddings] per-file fallback ({n} files, slow) ...')
    embs = torch.empty((n, *expected_shape), dtype=torch.float16)
    missing = []
    for i, mid in enumerate(mesh_ids):
        p = os.path.join(emb_dir, f'{mid}.pt')
        if not os.path.exists(p):
            missing.append(mid)
            continue
        e = torch.load(p, map_location='cpu', weights_only=False)
        if e.dtype != torch.float16:
            e = e.to(torch.float16)
        assert tuple(e.shape) == expected_shape, (
            f'embedding {p} shape {tuple(e.shape)} != expected {expected_shape}'
        )
        embs[i] = e
    if missing:
        raise FileNotFoundError(f'{len(missing)} embeddings missing, e.g. {missing[:5]}')
    return embs


def compute_G(C, U1):
    """G = C x_1 U_1.  C: (B, p, q, r)  U_1: (B, n, p)  ->  G: (B, n, q, r)."""
    return torch.einsum('bpqr,bnp->bnqr', C, U1)


def compute_stats(G, U2, U3):
    return {
        'G_mean':  G .mean().item(), 'G_std':  max(G .std().item(), 1e-6),
        'U2_mean': U2.mean().item(), 'U2_std': max(U2.std().item(), 1e-6),
        'U3_mean': U3.mean().item(), 'U3_std': max(U3.std().item(), 1e-6),
    }


def normalize(t, mean, std):
    return (t - mean) / std


def get_lr(step, base_lr, warmup, total, schedule):
    """Linear warmup → optional cosine decay."""
    if warmup > 0 and step < warmup:
        return base_lr * step / warmup
    if schedule == 'cosine':
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(max(progress, 0.0), 1.0)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for ep, p in zip(ema_model.parameters(), model.parameters()):
        ep.mul_(decay).add_(p.data, alpha=1.0 - decay)
    for eb, b in zip(ema_model.buffers(), model.buffers()):
        eb.copy_(b)


def eikonal_and_sign_loss(x0_hat_flat, x0_flat, N, R, stats, K, tau,
                           t_weight, device,
                           compute_eikonal=True, compute_sign=False,
                           tau_sign=0.005):
    """Combined TSDF-aware Eikonal + sign-consistency loss with shared voxel sampling.

    x0_hat_flat: (B, F) PREDICTED clean latent
    x0_flat:     (B, F) GROUND-TRUTH clean latent
    tau:         near-surface threshold for Eikonal mask (coord-space).
    tau_sign:    temperature for sign BCE (smaller = sharper sign enforcement).
    t_weight:    (B,) per-sample signal-strength weight (√ᾱ_t or cos(πt/2)).
    compute_eikonal: whether to compute |∇|→1 loss (near-surface masked).
    compute_sign:    whether to compute BCE sign-consistency loss (global, attacks
                     phantom zero-crossings → reduces n_components in MC).

    Returns dict: {'eikonal': L_eik?, 'sign': L_sign?, 'surf_frac': float, ...}
    """
    B = x0_hat_flat.shape[0]
    sG  = N * R * R
    sU2 = N * R
    sU3 = N * R

    def unflatten(flat):
        c0, c1, c2 = flat.split([sG, sU2, sU3], dim=1)
        G_  = c0.reshape(B, N, R, R) * stats['G_std']  + stats['G_mean']
        U2_ = c1.reshape(B, N, R)    * stats['U2_std'] + stats['U2_mean']
        U3_ = c2.reshape(B, N, R)    * stats['U3_std'] + stats['U3_mean']
        return G_, U2_, U3_

    G_hat, U2_hat, U3_hat = unflatten(x0_hat_flat)
    G_gt,  U2_gt,  U3_gt  = unflatten(x0_flat)

    # Shared sampling: K random voxels per shape (interior)
    pos = torch.randint(1, N - 1, (B, K, 3), device=device)
    ix, iy, iz = pos.unbind(-1)
    bi = torch.arange(B, device=device).unsqueeze(1).expand(-1, K)

    def eval_sdf(G_, U2_, U3_, i, j, k):
        g  = G_ [bi, i]
        u2 = U2_[bi, j]
        u3 = U3_[bi, k]
        return torch.einsum('bkqr,bkq,bkr->bk', g, u2, u3)

    # Target SDF at center: used for both Eikonal mask AND sign label
    with torch.no_grad():
        target_sdf = eval_sdf(G_gt, U2_gt, U3_gt, ix, iy, iz)

    out = {}
    w_sum = t_weight.sum().clamp(min=1e-6)

    # --- Sign consistency loss (global, BCE) ---
    if compute_sign:
        pred_sdf_c = eval_sdf(G_hat, U2_hat, U3_hat, ix, iy, iz)
        sign_label = (target_sdf < 0).float()                  # 1 = inside
        # logit = -pred/τ: positive → predicting inside
        logit = -pred_sdf_c / tau_sign
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, sign_label, reduction='none')               # (B, K)
        sign_per_sample = bce.mean(dim=1)
        out['sign'] = (sign_per_sample * t_weight).sum() / w_sum
        # Diagnostic: fraction of voxels where sign agrees
        with torch.no_grad():
            sign_correct = ((pred_sdf_c < 0) == (target_sdf < 0)).float()
            out['sign_acc'] = sign_correct.mean().item()

    # --- Eikonal loss (near-surface, |∇|→1) ---
    if compute_eikonal:
        with torch.no_grad():
            mask = (target_sdf.abs() < tau).float()
        h = 2.0 / (N - 1)
        gx = (eval_sdf(G_hat, U2_hat, U3_hat, ix + 1, iy, iz)
            - eval_sdf(G_hat, U2_hat, U3_hat, ix - 1, iy, iz)) / (2.0 * h)
        gy = (eval_sdf(G_hat, U2_hat, U3_hat, ix, iy + 1, iz)
            - eval_sdf(G_hat, U2_hat, U3_hat, ix, iy - 1, iz)) / (2.0 * h)
        gz = (eval_sdf(G_hat, U2_hat, U3_hat, ix, iy, iz + 1)
            - eval_sdf(G_hat, U2_hat, U3_hat, ix, iy, iz - 1)) / (2.0 * h)
        grad_norm = torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-8).clamp(max=10.0)
        err = (grad_norm - 1.0) ** 2
        num = (err * mask).sum(dim=1)
        den = mask.sum(dim=1).clamp(min=1.0)
        eik_per_sample = num / den
        out['eikonal'] = (eik_per_sample * t_weight).sum() / w_sum
        out['surf_frac'] = mask.mean().item()

    return out


def make_schedule(T=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {
        'alpha_bar': alpha_bar,
        'sqrt_alpha_bar': torch.sqrt(alpha_bar),
        'sqrt_one_minus_alpha_bar': torch.sqrt(1.0 - alpha_bar),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shard_dir',   required=True, nargs='+',
                   help='one or more shard directories; mesh_id dedup applied')
    p.add_argument('--out_dir',     required=True)
    p.add_argument('--steps',       type=int, default=50000)
    p.add_argument('--batch_size',  type=int, default=8)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--log_every',   type=int, default=200)
    p.add_argument('--ckpt_every',  type=int, default=2000)
    p.add_argument('--T',           type=int, default=1000)
    p.add_argument('--hidden_size', type=int, default=512)
    p.add_argument('--depth',       type=int, default=12)
    p.add_argument('--num_heads',   type=int, default=8)
    p.add_argument('--optimizer',   default='adamw', choices=['sgd', 'adamw', 'lion'])
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--betas',       default='0.9,0.99',
                   help='AdamW: paper uses (0.9, 0.99); modern FM/SI uses (0.9, 0.999). '
                        'Lion: (0.95, 0.98) is the original recommendation.')
    p.add_argument('--adam_eps', type=float, default=1e-8,
                   help='AdamW epsilon. Modern FM/SI uses 1e-15 for numerical stability.')
    p.add_argument('--target',      default='eps', choices=['eps', 'v'],
                   help='paper uses eps')
    p.add_argument('--resume',      default='auto')
    p.add_argument('--bf16',        action='store_true')
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--min_snr_gamma', type=float, default=0.0,
                   help='Min-SNR-γ loss weighting (Hang et al. 2023). 0 = off, paper uses 5.')
    p.add_argument('--si', default='none', choices=['none', 'linear', 'trig'],
                   help='Stochastic Interpolant mode (Albergo & Vanden-Eijnden 2023). '
                        'linear = Rectified Flow (α=1-t, β=t). '
                        'trig = variance-preserving (α=cos(πt/2), β=sin(πt/2)). '
                        'none (default) = standard DDPM.')
    p.add_argument('--grad_clip', type=float, default=1.0,
                   help='Max grad norm for clipping (prevents AdamW state pollution).'
                        ' 0 = off.')
    # --- v7.2 SI stability tricks ---
    p.add_argument('--ema_decay', type=float, default=0.0,
                   help='EMA decay for sampling model (0=off, recommend 0.9999)')
    p.add_argument('--lr_schedule', default='constant',
                   choices=['constant', 'cosine'])
    p.add_argument('--warmup_steps', type=int, default=0,
                   help='Linear warmup steps before cosine decay')
    p.add_argument('--t_sampling', default='uniform',
                   choices=['uniform', 'logit_normal'],
                   help='Time sampling for SI (logit_normal concentrates around t=0.5)')
    p.add_argument('--t_clamp', type=float, default=1e-4,
                   help='SI time clamp margin from {0,1} (default 1e-4, recommend 0.01)')
    # --- v8 Eikonal regularization ---
    p.add_argument('--eikonal_weight', type=float, default=0.0,
                   help='Eikonal loss weight (0=off, recommend 0.05-0.1)')
    p.add_argument('--eikonal_k_points', type=int, default=8192,
                   help='K random voxel sample points per shape for Eikonal '
                        '(after near-surface masking only ~5%% survive at default tau)')
    p.add_argument('--eikonal_tau_factor', type=float, default=0.8,
                   help='Near-surface mask threshold: |target_sdf| < tau_factor * mu * h.'
                        ' Default 0.8 keeps strictly inside TSDF truncation band.')
    p.add_argument('--eikonal_mu', type=float, default=2.0,
                   help='TSDF truncation factor used during preprocessing (default 2.0)')
    # --- v7.4 SI modern recipe additions ---
    p.add_argument('--qk_norm', action='store_true',
                   help='Add QK-RMSNorm in DiT attention (SD3/TRELLIS standard).'
                        ' Prevents exploding attention norms in long FM/SI training.')
    # --- v9 conditional generation ---
    p.add_argument('--cond_dir', default='',
                   help='Directory with DINOv2 .pt embeddings (one per mesh_id). '
                        'Enables conditional training.')
    p.add_argument('--cond_dim',      type=int, default=1024,  help='DINOv2-L = 1024')
    p.add_argument('--cond_n_views',  type=int, default=8)
    p.add_argument('--cond_n_tokens', type=int, default=257,   help='DINOv2-L/14 224px')
    p.add_argument('--cond_dropout',  type=float, default=0.1,
                   help='Classifier-free guidance dropout: fraction of batches where '
                        'cond is replaced with all-zeros. Standard 0.1.')
    p.add_argument('--init_from', default='',
                   help='Warm-start backbone weights from this checkpoint (e.g. v7.4a). '
                        'Cross-attn / cond projections init from random (zero-init proj).')
    # --- v8' Sign-consistency loss ---
    p.add_argument('--sign_weight', type=float, default=0.0,
                   help='Sign-consistency BCE loss weight (0=off, recommend 0.1-0.3).'
                        ' Directly attacks spurious zero-crossings → reduces MC n_components.')
    p.add_argument('--tau_sign', type=float, default=0.005,
                   help='Temperature for sign BCE: logit = -pred_sdf/tau_sign.'
                        ' Smaller = sharper (default 0.005 ~ μh/3)')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    if device.type == 'cuda':
        print(f'  gpu: {torch.cuda.get_device_name(0)}')
        print(f'  mem: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    print('Loading shards from', args.shard_dir)
    U1, U2, U3, C, N, R, kept_mesh_ids = load_shards(args.shard_dir)
    n_total = U1.shape[0]
    print(f'  loaded {n_total} samples  U1 {U1.shape}  U2 {U2.shape}  U3 {U3.shape}  C {C.shape}')

    # Conditional: pre-load DINOv2 embeddings in the SAME order as flat latents
    use_cond = bool(args.cond_dir)
    embs = None
    if use_cond:
        emb_shape = (args.cond_n_views, args.cond_n_tokens, args.cond_dim)
        print(f'Loading {n_total} embeddings from {args.cond_dir} '
              f'(shape per mesh: {emb_shape}) ...')
        t_emb = time.time()
        embs = load_embeddings(args.cond_dir, kept_mesh_ids, emb_shape)
        print(f'  embs: {tuple(embs.shape)}  dtype={embs.dtype}  '
              f'load time {time.time()-t_emb:.1f}s')

    print('Building G = C x_1 U_1 ...')
    G = compute_G(C, U1)             # (B, N, R, R)
    print(f'  G {G.shape}  (discarding U_1; not in primitive per l=2 TGP)')
    del U1, C

    stats = compute_stats(G, U2, U3)
    print(f'  stats: {stats}')
    G  = normalize(G,  stats['G_mean'],  stats['G_std'])
    U2 = normalize(U2, stats['U2_mean'], stats['U2_std'])
    U3 = normalize(U3, stats['U3_mean'], stats['U3_std'])

    flat = torch.cat([G .reshape(n_total, -1),
                      U2.reshape(n_total, -1),
                      U3.reshape(n_total, -1)], dim=1)
    F_total = flat.shape[1]
    print(f'  flat shape: {flat.shape}  (F={F_total})')

    cfg = SDFDiTConfig(N=N, R=R,
                         hidden_size=args.hidden_size,
                         depth=args.depth,
                         num_heads=args.num_heads,
                         qk_norm=args.qk_norm,
                         cond_dim     =(args.cond_dim     if use_cond else 0),
                         cond_n_views =(args.cond_n_views if use_cond else 0),
                         cond_n_tokens=(args.cond_n_tokens if use_cond else 0))
    print(f'\ncfg: {cfg}')
    model = SDFDiTWrapper(SDFDiT(cfg))
    n_params = sum(p.numel() for p in model.parameters())
    print(f'params: {n_params:,}  flat: {model.core.flat_total}  seq_len: {model.core.seq_len}')
    assert F_total == model.core.flat_total
    model = model.to(device).train()

    # --- Optional warm-start from unconditional v7.4 backbone ---
    if args.init_from and os.path.exists(args.init_from):
        print(f'[init_from] loading backbone weights from {args.init_from}')
        init_state = torch.load(args.init_from, map_location='cpu', weights_only=False)
        src_sd = init_state['state_dict']
        own_sd = model.state_dict()
        matched, mismatched, missing = 0, [], 0
        for k, v in src_sd.items():
            if k in own_sd and own_sd[k].shape == v.shape:
                own_sd[k].copy_(v)
                matched += 1
            elif k in own_sd:
                mismatched.append((k, tuple(v.shape), tuple(own_sd[k].shape)))
        for k in own_sd:
            if k not in src_sd:
                missing += 1
        model.load_state_dict(own_sd)
        print(f'  matched {matched}/{len(src_sd)} src; '
              f'{missing} dest keys absent in src (these stay random-init, '
              f'e.g. cross-attn + cond_proj layers).')
        if mismatched:
            print(f'  WARN: {len(mismatched)} shape mismatches, first:')
            for k, ss, ds in mismatched[:3]:
                print(f'    {k}: src {ss} != dest {ds}')

    ema_model = None
    if args.ema_decay > 0:
        ema_model = copy.deepcopy(model).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        print(f'EMA enabled: decay={args.ema_decay}')

    start_step = 0
    resume_state = None
    if args.resume not in ('none', None, ''):
        if args.resume == 'auto':
            cands = sorted(glob.glob(os.path.join(args.out_dir, 'ckpt_step*.pt')))
            ckpt_path = cands[-1] if cands else None
        else:
            ckpt_path = args.resume if os.path.exists(args.resume) else None
        if ckpt_path:
            print(f'[resume] loading {ckpt_path}')
            resume_state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            start_step = int(resume_state.get('step', 0))
            model.load_state_dict(resume_state['state_dict'])
            if ema_model is not None:
                if 'ema_state_dict' in resume_state:
                    ema_model.load_state_dict(resume_state['ema_state_dict'])
                    print('[resume] EMA state restored')
                else:
                    ema_model.load_state_dict(resume_state['state_dict'])
                    print('[resume] EMA initialized from model state (no prior EMA)')
            print(f'[resume] starting from step {start_step}')

    if args.optimizer == 'adamw':
        b1, b2 = [float(x) for x in args.betas.split(',')]
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                betas=(b1, b2), eps=args.adam_eps,
                                weight_decay=args.weight_decay)
        print(f'optimizer: AdamW lr={args.lr} betas=({b1},{b2}) eps={args.adam_eps} wd={args.weight_decay}')
    elif args.optimizer == 'lion':
        from lion_pytorch import Lion
        b1, b2 = [float(x) for x in args.betas.split(',')]
        opt = Lion(model.parameters(), lr=args.lr,
                   betas=(b1, b2), weight_decay=args.weight_decay)
        print(f'optimizer: Lion lr={args.lr} betas=({b1},{b2}) wd={args.weight_decay}')
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        print(f'optimizer: SGD lr={args.lr} momentum=0.9')

    if resume_state and 'opt_state_dict' in resume_state:
        try:
            opt.load_state_dict(resume_state['opt_state_dict'])
            print('[resume] optimizer state restored')
        except Exception as e:
            print(f'[resume] WARN: could not load opt state: {e}')

    sched = make_schedule(args.T, device=device)
    sqrt_ab    = sched['sqrt_alpha_bar']
    sqrt_1m_ab = sched['sqrt_one_minus_alpha_bar']

    amp_dtype = torch.bfloat16 if args.bf16 else None

    print(f'\n--- training steps {start_step}..{args.steps}  '
          f'(batch_size={args.batch_size}, bf16={args.bf16}, target={args.target}) ---')
    t_start = time.time()
    losses = list(resume_state['losses']) if resume_state else []
    eik_losses = []
    target_rms_list = []
    pred_rms_list = []
    grad_norm_list = []
    for step in range(start_step, args.steps):
        t0 = time.time()
        # LR schedule
        lr_now = get_lr(step, args.lr, args.warmup_steps, args.steps, args.lr_schedule)
        for pg in opt.param_groups:
            pg['lr'] = lr_now

        idx = torch.randint(0, n_total, (args.batch_size,))
        x0 = flat[idx].to(device, non_blocking=True)
        noise = torch.randn_like(x0)

        # Conditional: select image embeddings for this batch, apply CFG dropout.
        cond_batch = None
        if use_cond:
            cond_batch = embs[idx].to(device, non_blocking=True).float()
            if args.cond_dropout > 0:
                drop = torch.rand(args.batch_size, device=device) < args.cond_dropout
                if drop.any():
                    cond_batch = cond_batch.clone()
                    cond_batch[drop] = 0.0

        if args.si == 'none':
            # ===== Standard DDPM path =====
            t = torch.randint(0, args.T, (args.batch_size,), device=device)
            a_sqrt    = sqrt_ab.index_select(0, t).unsqueeze(1)
            a_1m_sqrt = sqrt_1m_ab.index_select(0, t).unsqueeze(1)
            xt = a_sqrt * x0 + a_1m_sqrt * noise

            if args.min_snr_gamma > 0:
                snr = (a_sqrt ** 2) / (a_1m_sqrt ** 2 + 1e-12)
                if args.target == 'v':
                    snr_w = torch.clamp(snr, max=args.min_snr_gamma) / (snr + 1.0)
                else:
                    snr_w = torch.clamp(snr, max=args.min_snr_gamma) / (snr + 1e-12)
            else:
                snr_w = None

            if args.target == 'v':
                target = a_sqrt * noise - a_1m_sqrt * x0
            else:
                target = noise
            t_model = t
        else:
            # ===== Stochastic Interpolant path (Albergo & Vanden-Eijnden 2023) =====
            # SD3 (Esser+ 2024) uses logit-normal to concentrate t around 0.5 and
            # downweight unstable endpoint regions. Wider clamp also helps.
            if args.t_sampling == 'logit_normal':
                u = torch.randn(args.batch_size, device=device)
                t = torch.sigmoid(u).clamp_(args.t_clamp, 1.0 - args.t_clamp)
            else:
                t = torch.rand(args.batch_size, device=device).clamp_(args.t_clamp, 1.0 - args.t_clamp)

            if args.si == 'linear':
                # Rectified Flow: x_t = (1-t)*x_0 + t*x_1
                alpha     = (1.0 - t).unsqueeze(1)
                beta      = t.unsqueeze(1)
                alpha_dot = torch.full_like(alpha, -1.0)
                beta_dot  = torch.full_like(beta,   1.0)
            else:  # 'trig'
                # Variance-preserving: x_t = cos(πt/2)*x_0 + sin(πt/2)*x_1
                half_pi_t = (0.5 * math.pi) * t
                alpha     = torch.cos(half_pi_t).unsqueeze(1)
                beta      = torch.sin(half_pi_t).unsqueeze(1)
                alpha_dot = -(0.5 * math.pi) * beta
                beta_dot  =  (0.5 * math.pi) * alpha

            xt = alpha * x0 + beta * noise
            # Velocity field target: dx_t/dt = α̇(t)*x_0 + β̇(t)*x_1
            target = alpha_dot * x0 + beta_dot * noise
            # Scale continuous t to integer range matching DDPM embedder.
            t_model = (t * (args.T - 1)).long()
            snr_w = None  # Min-SNR-γ doesn't apply to SI

        if amp_dtype is not None:
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                pred = model(xt, t_model, cond=cond_batch)
                sq = (pred.float() - target.float()) ** 2
                if snr_w is not None:
                    sq = sq * snr_w
                loss_diff = sq.mean()
        else:
            pred = model(xt, t_model, cond=cond_batch)
            sq = (pred - target) ** 2
            if snr_w is not None:
                sq = sq * snr_w
            loss_diff = sq.mean()

        # v7.4 diagnostic: target/pred magnitude tracking (locate outlier batches)
        with torch.no_grad():
            target_rms = float(target.float().pow(2).mean().sqrt())
            pred_rms   = float(pred.detach().float().pow(2).mean().sqrt())

        # TSDF-aware Eikonal + Sign-consistency on predicted clean x_0
        loss_eik_val = 0.0
        loss_sign_val = 0.0
        eik_surf_frac = 0.0
        sign_acc = 0.0
        if args.eikonal_weight > 0 or args.sign_weight > 0:
            pred_f = pred.float()
            xt_f   = xt.float()
            if args.si == 'none':
                if args.target == 'v':
                    x0_hat = a_sqrt * xt_f - a_1m_sqrt * pred_f
                else:  # eps
                    x0_hat = (xt_f - a_1m_sqrt * pred_f) / a_sqrt
                t_weight = a_sqrt.squeeze(1).float()
            elif args.si == 'trig':
                x0_hat = alpha * xt_f - (2.0 / math.pi) * beta * pred_f
                t_weight = alpha.squeeze(1).float()
            else:
                x0_hat = xt_f - beta * pred_f
                t_weight = alpha.squeeze(1).float()
            voxel = 2.0 / (N - 1)
            tau = args.eikonal_tau_factor * args.eikonal_mu * voxel
            out = eikonal_and_sign_loss(
                x0_hat, x0.float(), N, R, stats,
                args.eikonal_k_points, tau, t_weight, device,
                compute_eikonal=(args.eikonal_weight > 0),
                compute_sign=(args.sign_weight > 0),
                tau_sign=args.tau_sign)
            extra = 0.0
            if 'eikonal' in out:
                extra = extra + args.eikonal_weight * out['eikonal']
                loss_eik_val = float(out['eikonal'].detach().cpu())
                eik_surf_frac = out['surf_frac']
            if 'sign' in out:
                extra = extra + args.sign_weight * out['sign']
                loss_sign_val = float(out['sign'].detach().cpu())
                sign_acc = out['sign_acc']
            loss = loss_diff + extra
        else:
            loss = loss_diff

        opt.zero_grad(set_to_none=True)

        # Skip non-finite loss to protect AdamW state
        if not torch.isfinite(loss):
            print(f'  WARN step {step}: non-finite loss ({loss.item()}), skipping update', flush=True)
            losses.append(float('nan'))
            continue

        loss.backward()
        # Gradient clipping (prevents AdamW state pollution from outlier batches)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        opt.step()

        if ema_model is not None:
            update_ema(ema_model, model, args.ema_decay)

        loss_val = float(loss_diff.detach().cpu())
        losses.append(loss_val)
        eik_losses.append(loss_eik_val)
        target_rms_list.append(target_rms)
        pred_rms_list.append(pred_rms)
        grad_norm_list.append(float(grad_norm))

        if step % args.log_every == 0 or step == args.steps - 1:
            dt = time.time() - t0
            elapsed = time.time() - t_start
            mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0
            extra = f'  trms={target_rms:.3f}  prms={pred_rms:.3f}  gn={float(grad_norm):.3f}'
            if args.eikonal_weight > 0:
                extra += f'  eik={loss_eik_val:.4f}  surf={eik_surf_frac:.3f}'
            if args.sign_weight > 0:
                extra += f'  sgn={loss_sign_val:.4f}  acc={sign_acc:.3f}'
            if args.lr_schedule != 'constant':
                extra += f'  lr={lr_now:.2e}'
            print(f'step {step:>6d}/{args.steps}  loss={loss_val:.4f}{extra}  '
                  f'step_dt={dt*1000:.0f}ms  elapsed={elapsed:.0f}s  '
                  f'gpu_mem={mem:.1f}GB', flush=True)

        if (step + 1) % args.ckpt_every == 0:
            ckpt_path = os.path.join(args.out_dir, f'ckpt_step{step+1:06d}.pt')
            payload = {'step': step+1,
                       'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                       'stats': stats,
                       'cfg':   cfg.__dict__,
                       'target': args.target,
                       'si': args.si,
                       'opt_state_dict': opt.state_dict(),
                       'losses': losses}
            if ema_model is not None:
                payload['ema_state_dict'] = {k: v.cpu() for k, v in ema_model.state_dict().items()}
            torch.save(payload, ckpt_path)
            print(f'  -> checkpoint {ckpt_path}', flush=True)

    ckpt_path = os.path.join(args.out_dir, f'ckpt_step{args.steps:06d}.pt')
    if not os.path.exists(ckpt_path):
        payload = {'step': args.steps,
                   'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                   'stats': stats,
                   'cfg':   cfg.__dict__,
                   'target': args.target,
                   'si': args.si,
                   'opt_state_dict': opt.state_dict(),
                   'losses': losses}
        if ema_model is not None:
            payload['ema_state_dict'] = {k: v.cpu() for k, v in ema_model.state_dict().items()}
        torch.save(payload, ckpt_path)
        print(f'  -> checkpoint {ckpt_path}')

    if losses:
        print(f'\nLast loss: {losses[-1]:.4f}  mean(last100): {np.mean(losses[-100:]):.4f}')
    np.save(os.path.join(args.out_dir, 'losses.npy'), np.array(losses))
    if target_rms_list:
        np.save(os.path.join(args.out_dir, 'target_rms.npy'), np.array(target_rms_list))
        np.save(os.path.join(args.out_dir, 'pred_rms.npy'),   np.array(pred_rms_list))
        np.save(os.path.join(args.out_dir, 'grad_norm.npy'),  np.array(grad_norm_list))
    print(f'DONE step={args.steps}')


if __name__ == '__main__':
    main()
