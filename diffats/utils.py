"""
Shared utilities for the ABC SDF compression experiment.

Sign convention throughout: SDF inside negative, outside positive.
(pysdf returns inside-positive; we flip the sign at the source.)
"""

import numpy as np
import trimesh
try:
    from pysdf import SDF
except ImportError:
    SDF = None  # not needed for sampling
from skimage.measure import marching_cubes
from scipy.spatial import cKDTree
import tensorly as tl
from tensorly.decomposition import tucker, tensor_train


# ---------------------------------------------------------------------------
# Mesh & SDF
# ---------------------------------------------------------------------------

def normalize_mesh(mesh, margin=0.9):
    """Center mesh and scale so its largest extent fits in [-margin, margin]."""
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.centroid)
    max_extent = float(np.max(mesh.extents))
    if max_extent <= 0:
        raise ValueError("Invalid mesh extent.")
    mesh.apply_scale(2.0 * margin / max_extent)
    return mesh


def make_grid(N, bbox_min=-1.0, bbox_max=1.0):
    """Regular voxel grid points + voxel_size."""
    lin = np.linspace(bbox_min, bbox_max, N)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    voxel_size = (bbox_max - bbox_min) / (N - 1)
    return points, voxel_size


def compute_sdf(mesh, N=128, bbox_min=-1.0, bbox_max=1.0, batch_size=200000):
    """Compute SDF on a regular N^3 grid. Inside negative, outside positive."""
    points, voxel_size = make_grid(N, bbox_min, bbox_max)
    sdf_func = SDF(mesh.vertices, mesh.faces)
    out = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), batch_size):
        end = min(start + batch_size, len(points))
        out[start:end] = -sdf_func(points[start:end]).astype(np.float32)
    return out.reshape(N, N, N), voxel_size


def truncate_sdf(sdf, mu):
    """Clamp to [-mu, mu]."""
    return np.clip(sdf, -mu, mu)


def extract_mesh(sdf, voxel_size, bbox_min=-1.0):
    """Marching cubes at level 0. Returns trimesh.Trimesh or None."""
    try:
        verts, faces, _, _ = marching_cubes(
            sdf, level=0.0,
            spacing=(voxel_size, voxel_size, voxel_size),
        )
    except (ValueError, RuntimeError):
        return None
    if len(verts) == 0 or len(faces) == 0:
        return None
    verts = verts + np.array([bbox_min, bbox_min, bbox_min])
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


# ---------------------------------------------------------------------------
# Decompositions
# ---------------------------------------------------------------------------

def tt_compress(tensor, max_rank):
    """Tensor-train decomposition with uniform max rank.

    Returns
    -------
    cores : list of np.ndarray
    n_params : int
    """
    arr = tl.tensor(tensor.astype(np.float64))
    tt = tensor_train(arr, rank=max_rank)
    # tensorly versions differ: tt may be TTTensor with .factors, or already a list
    cores = list(tt) if not hasattr(tt, "factors") else list(tt.factors)
    n_params = int(sum(c.size for c in cores))
    return cores, n_params


def tt_reconstruct(cores):
    return np.asarray(tl.tt_to_tensor(cores))


def tucker_compress(tensor, ranks):
    """Tucker decomposition with given per-mode ranks (tuple)."""
    arr = tl.tensor(tensor.astype(np.float64))
    core, factors = tucker(arr, rank=list(ranks))
    n_params = int(core.size + sum(f.size for f in factors))
    return (core, factors), n_params


def tucker_reconstruct(decomp):
    core, factors = decomp
    return np.asarray(tl.tucker_to_tensor((core, factors)))


def tucker_rank_matching_tt(N, tt_rank):
    """Pick a (uniform) Tucker rank R so Tucker params >= TT params.

    TT params for 3-core uniform-rank R_TT, mode N:
        N*R_TT + R_TT*N*R_TT + R_TT*N = N*R_TT^2 + 2*N*R_TT
    Tucker params for rank (R,R,R), mode N:
        R^3 + 3*N*R
    """
    tt_params = N * tt_rank * tt_rank + 2 * N * tt_rank
    R = 1
    while R**3 + 3 * N * R < tt_params and R < N:
        R += 1
    return R


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def volume_rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def iou_volume(sdf_gt, sdf_pred):
    a = sdf_gt < 0
    b = sdf_pred < 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def _surface_samples(mesh, n=30000, rng=None):
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return np.asarray(pts)


def boundary_metrics(mesh_a, mesh_b, n=30000):
    """Symmetric Chamfer (mean d^2, mean d) and Hausdorff (normalized by bbox diagonal of a)."""
    inf = float("inf")
    out = {"chamfer_l2_sq": inf, "chamfer_l1": inf, "hausdorff_rel": inf}
    if (mesh_a is None or mesh_b is None
            or len(mesh_a.vertices) == 0 or len(mesh_b.vertices) == 0
            or len(mesh_a.faces) == 0 or len(mesh_b.faces) == 0):
        return out
    pa = _surface_samples(mesh_a, n)
    pb = _surface_samples(mesh_b, n)
    tree_a = cKDTree(pa)
    tree_b = cKDTree(pb)
    d_ab, _ = tree_b.query(pa)
    d_ba, _ = tree_a.query(pb)
    out["chamfer_l2_sq"] = float((d_ab ** 2).mean() + (d_ba ** 2).mean())
    out["chamfer_l1"] = float(d_ab.mean() + d_ba.mean())
    hd = float(max(d_ab.max(), d_ba.max()))
    diag = float(np.linalg.norm(mesh_a.bounds[1] - mesh_a.bounds[0]))
    out["hausdorff_rel"] = hd / diag if diag > 0 else inf
    return out
