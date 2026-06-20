"""
model.py - Paper-faithful DiffATs DiT.

Differences from v3:
  v3: l=1 TGP — diffuses on (U_1, U_2, U_3, C). Paper requires l >= 2.
  v3: time prepended as a token (no per-block conditioning).
  v3: v-prediction.

  v4: l=2 TGP — diffuses on (G, U_2, U_3), where
        G = X x_2 U_2^T x_3 U_3^T = C x_1 U_1  (mode-1 absorbed into G).
      U_1 is dropped (mode-1 is left "unaligned" per paper Sec 4.4 Step 2).
  v4: DiT AdaLN-Zero blocks (per-block modulation by t-embedding).
  v4: eps-prediction default (paper Sec C.4).

Sequence layout (N=256, R=24):
  G:   R*R = 576 tokens, each token = one (a,b) slice of shape (N,)
  U_2: R   = 24  tokens, each token = one rank column of shape (N,)
  U_3: R   = 24  tokens, each token = one rank column of shape (N,)
  Total seq_len = R*R + 2R = 576 + 48 = 624
Flat latent dim = N*R*R + 2*N*R = 256*576 + 2*256*24 = 147456 + 12288 = 159744
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embed(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t):
        return self.mlp(sinusoidal_embed(t, self.frequency_embedding_size))


def modulate(x, shift, scale):
    # x: (B, L, H), shift/scale: (B, H) -> broadcast on L
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class CrossAttention(nn.Module):
    """Image-condition cross-attention used in conditional DiT blocks.

    Q comes from the diffusion latent tokens (size hidden); K, V come from the
    flattened image-condition tokens (size cond_dim). Output projection is
    zero-init so a fresh conditional block starts as a no-op, allowing reuse
    of unconditional checkpoint weights.
    """

    def __init__(self, hidden, cond_dim, num_heads, qk_norm=False):
        super().__init__()
        assert hidden % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden // num_heads
        self.q   = nn.Linear(hidden,   hidden)
        self.kv  = nn.Linear(cond_dim, hidden * 2)
        self.proj = nn.Linear(hidden,   hidden)
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, cond):
        B, L, H = x.shape
        _, M, _ = cond.shape
        q = self.q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(cond).view(B, M, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = self.q_norm(q.float()).to(q.dtype)
        k = self.k_norm(k.float()).to(k.dtype)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, H)
        return self.proj(a)


class DiTBlock(nn.Module):
    """Standard DiT block with AdaLN-Zero (Peebles & Xie 2023).

    Optional QK-RMSNorm (SD3 / TRELLIS / FLUX) prevents exploding attention
    logit norms during mixed-precision long training, the standard fix for the
    flow-matching low-noise pathology (arXiv:2509.20952).
    """

    def __init__(self, hidden, num_heads, mlp_ratio=4.0, qk_norm=False,
                 cond_dim=0):
        super().__init__()
        assert hidden % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.qk_norm = qk_norm
        self.has_cond = cond_dim > 0

        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.qkv   = nn.Linear(hidden, hidden * 3)
        self.proj  = nn.Linear(hidden, hidden)

        # QK-RMSNorm on head_dim — applies per-head after q/k projection.
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # Image-condition cross-attention. Zero-init projection keeps block
        # identical to unconditional behavior at init → can warm-start from
        # v7.4 weights.
        if self.has_cond:
            self.norm_cross = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
            self.cross_attn = CrossAttention(hidden, cond_dim, num_heads,
                                             qk_norm=qk_norm)
            nn.init.zeros_(self.cross_attn.proj.weight)
            nn.init.zeros_(self.cross_attn.proj.bias)

        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        inner = int(hidden * mlp_ratio)
        self.fc1 = nn.Linear(hidden, inner)
        self.fc2 = nn.Linear(inner, hidden)

        # produces (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.ada_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, 6 * hidden, bias=True),
        )

    def _attn(self, x):
        B, L, H = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        # QK-RMSNorm (no-op when qk_norm=False, runs in fp32 cast for safety).
        q = self.q_norm(q.float()).to(q.dtype)
        k = self.k_norm(k.float()).to(k.dtype)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, H)
        return self.proj(a)

    def _mlp(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate='tanh'))

    def forward(self, x, c, cond=None):
        sh1, sc1, gt1, sh2, sc2, gt2 = self.ada_mod(c).chunk(6, dim=-1)
        x = x + gt1.unsqueeze(1) * self._attn(modulate(self.norm1(x), sh1, sc1))
        if self.has_cond and cond is not None:
            x = x + self.cross_attn(self.norm_cross(x), cond)
        x = x + gt2.unsqueeze(1) * self._mlp(modulate(self.norm2(x), sh2, sc2))
        return x


class FinalLayer(nn.Module):
    """AdaLN-Zero final layer projecting tokens back to feature dim."""

    def __init__(self, hidden, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden, out_dim)
        self.ada_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, 2 * hidden, bias=True),
        )

    def forward(self, x, c):
        sh, sc = self.ada_mod(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), sh, sc)
        return self.linear(x)


@dataclass
class SDFDiTConfig:
    N:             int = 256
    R:             int = 24
    hidden_size:   int = 512
    depth:         int = 12
    num_heads:     int = 8
    mlp_ratio:     float = 4.0
    qk_norm:       bool  = False  # SD3/TRELLIS attention stability for SI/FM
    # --- v9 conditional generation ---
    cond_dim:      int = 0        # 0 = unconditional; 1024 = DINOv2-L
    cond_n_views:  int = 0        # # of views in condition (e.g., 8)
    cond_n_tokens: int = 0        # # of tokens per view (e.g., 257 for DINOv2-L)


class SDFDiT(nn.Module):
    """Diffuse on (G, U_2, U_3) per DiffATs paper TGP with l=2."""

    def __init__(self, cfg: SDFDiTConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size
        N, R = cfg.N, cfg.R

        # token feature dim is always N (each token is a length-N slice/column)
        self.proj_G  = nn.Linear(N, H)
        self.proj_U2 = nn.Linear(N, H)
        self.proj_U3 = nn.Linear(N, H)

        # 3 token types: G, U_2, U_3
        self.type_embed = nn.Embedding(3, H)

        # G has R*R tokens indexed by (a, b)
        self.pe_G_a = nn.Embedding(R, H)
        self.pe_G_b = nn.Embedding(R, H)
        # U_2, U_3 each have R tokens indexed by rank
        self.pe_U2  = nn.Embedding(R, H)
        self.pe_U3  = nn.Embedding(R, H)

        self.t_embedder = TimestepEmbedder(H)

        # Image-condition projection + per-view / per-token positional embeddings.
        self.has_cond = cfg.cond_dim > 0
        if self.has_cond:
            self.cond_proj      = nn.Linear(cfg.cond_dim, H)
            self.cond_view_pe   = nn.Embedding(cfg.cond_n_views,  H)
            self.cond_token_pe  = nn.Embedding(cfg.cond_n_tokens, H)

        self.blocks = nn.ModuleList(
            [DiTBlock(H, cfg.num_heads, cfg.mlp_ratio, qk_norm=cfg.qk_norm,
                      cond_dim=H if self.has_cond else 0)
             for _ in range(cfg.depth)]
        )

        # per-stream final layer (token dim H -> token feature dim N)
        self.final_G  = FinalLayer(H, N)
        self.final_U2 = FinalLayer(H, N)
        self.final_U3 = FinalLayer(H, N)

        rank_idx = torch.arange(R)
        self.register_buffer('rank_idx', rank_idx, persistent=False)
        self.register_buffer('g_a_idx', rank_idx.repeat_interleave(R), persistent=False)
        self.register_buffer('g_b_idx', rank_idx.repeat(R),             persistent=False)

        self.N_G  = R * R
        self.N_U2 = R
        self.N_U3 = R
        self.seq_len = self.N_G + self.N_U2 + self.N_U3

        self.flat_G  = R * R * N
        self.flat_U2 = R * N
        self.flat_U3 = R * N
        self.flat_total = self.flat_G + self.flat_U2 + self.flat_U3

        self._init_weights()

    def _init_weights(self):
        # default Xavier for Linear, normal for Embedding
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        # Zero out the AdaLN modulation projection at init (AdaLN-Zero).
        for block in self.blocks:
            nn.init.zeros_(block.ada_mod[-1].weight)
            nn.init.zeros_(block.ada_mod[-1].bias)
        for final in (self.final_G, self.final_U2, self.final_U3):
            nn.init.zeros_(final.ada_mod[-1].weight)
            nn.init.zeros_(final.ada_mod[-1].bias)
            nn.init.zeros_(final.linear.weight)
            nn.init.zeros_(final.linear.bias)

    def forward(self, G, U2, U3, t, cond=None):
        # G:  (B, N, R, R)  -> tokens (B, R*R, N) via permute
        # U2: (B, N, R)     -> tokens (B, R,   N) via transpose
        # U3: (B, N, R)     -> tokens (B, R,   N) via transpose
        # cond: (B, n_views, n_tokens, cond_dim) — DINOv2 image-condition tokens
        B = G.shape[0]
        R = self.cfg.R

        G_tok  = G.permute(0, 2, 3, 1).reshape(B, R * R, -1)   # (B, R*R, N)
        U2_tok = U2.transpose(1, 2).contiguous()               # (B, R,   N)
        U3_tok = U3.transpose(1, 2).contiguous()               # (B, R,   N)

        h_G  = (self.proj_G(G_tok)
                + self.pe_G_a(self.g_a_idx)
                + self.pe_G_b(self.g_b_idx)
                + self.type_embed.weight[0])
        h_U2 = (self.proj_U2(U2_tok)
                + self.pe_U2(self.rank_idx)
                + self.type_embed.weight[1])
        h_U3 = (self.proj_U3(U3_tok)
                + self.pe_U3(self.rank_idx)
                + self.type_embed.weight[2])

        seq = torch.cat([h_G, h_U2, h_U3], dim=1)
        c   = self.t_embedder(t)

        # Project condition + add view/token positional embeddings → flat (B, V*T, H).
        cond_seq = None
        if self.has_cond and cond is not None:
            Bc, V, T, D = cond.shape
            assert V == self.cfg.cond_n_views and T == self.cfg.cond_n_tokens, (
                f'cond shape mismatch: got V={V},T={T} expect '
                f'{self.cfg.cond_n_views},{self.cfg.cond_n_tokens}')
            cp = self.cond_proj(cond)                                  # (B, V, T, H)
            v_idx = torch.arange(V, device=cond.device)
            t_idx = torch.arange(T, device=cond.device)
            cp = cp + self.cond_view_pe(v_idx)[None, :, None, :]
            cp = cp + self.cond_token_pe(t_idx)[None, None, :, :]
            cond_seq = cp.reshape(Bc, V * T, -1)                       # (B, V*T, H)

        for blk in self.blocks:
            seq = blk(seq, c, cond_seq)

        s_G  = seq[:, :self.N_G]
        s_U2 = seq[:, self.N_G:self.N_G + R]
        s_U3 = seq[:, self.N_G + R:]

        pred_G  = self.final_G(s_G,  c)                          # (B, R*R, N)
        pred_U2 = self.final_U2(s_U2, c)                         # (B, R,   N)
        pred_U3 = self.final_U3(s_U3, c)                         # (B, R,   N)

        pred_G  = pred_G.reshape(B, R, R, -1).permute(0, 3, 1, 2)  # (B, N, R, R)
        pred_U2 = pred_U2.transpose(1, 2).contiguous()             # (B, N, R)
        pred_U3 = pred_U3.transpose(1, 2).contiguous()             # (B, N, R)
        return pred_G, pred_U2, pred_U3


class SDFDiTWrapper(nn.Module):
    """Flat-vector adapter: (B, F) <-> (G, U_2, U_3)."""

    def __init__(self, core: SDFDiT):
        super().__init__()
        self.core = core
        N, R = core.cfg.N, core.cfg.R
        self._splits = [N * R * R, N * R, N * R]
        self._shapes = [(N, R, R), (N, R), (N, R)]

    def forward(self, x_flat, t, cond=None):
        B = x_flat.shape[0]
        c0, c1, c2 = x_flat.split(self._splits, dim=1)
        G  = c0.reshape(B, *self._shapes[0])
        U2 = c1.reshape(B, *self._shapes[1])
        U3 = c2.reshape(B, *self._shapes[2])
        pG, pU2, pU3 = self.core(G, U2, U3, t, cond=cond)
        return torch.cat([pG.reshape(B, -1),
                          pU2.reshape(B, -1),
                          pU3.reshape(B, -1)], dim=1)


if __name__ == '__main__':
    cfg = SDFDiTConfig(N=256, R=24, hidden_size=512, depth=12, num_heads=8)
    m = SDFDiTWrapper(SDFDiT(cfg))
    n_params = sum(p.numel() for p in m.parameters())
    print(f'cfg: {cfg}')
    print(f'params: {n_params:,}')
    print(f'seq_len: {m.core.seq_len}  flat: {m.core.flat_total}')
    B = 2
    x = torch.randn(B, m.core.flat_total)
    t = torch.randint(0, 1000, (B,))
    out = m(x, t)
    assert out.shape == x.shape, f'{out.shape} != {x.shape}'
    print(f'IO check OK: {tuple(x.shape)} -> {tuple(out.shape)}')
