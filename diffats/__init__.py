"""DiffATs — Tucker-latent diffusion transformers for 3D CAD shape generation.

Public modules:
    diffats.models    — the SDF-DiT (l=2 TGP, AdaLN-Zero, optional cross-attention)
    diffats.sampling  — ODE / DDIM samplers and SDF reconstruction
    diffats.render    — multi-view headless rendering (conditioning + viz)
    diffats.utils     — TSDF, Tucker decomposition, marching cubes, metrics

Heavy submodules (torch, pyrender, transformers) are imported lazily on use, so
`import diffats` itself stays light.
"""

__version__ = "0.1.0"
