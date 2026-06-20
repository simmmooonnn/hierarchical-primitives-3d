"""
recon_vs_original.py
比较 Tucker 压缩-还原后的 mesh 与原始 ABC mesh 的差距。

数据布局:
  shard_dir 下是 tucker_sdf_shard_*.pt,每个 shard 含 100 个 mesh 批:
    U_1, U_2, U_3   shape (B, N, R)
    C               shape (B, R, R, R)
    mesh_id         list of B str
    mesh_path       list of B absolute paths to original .obj
    voxel_size      (B,)

参数:
  --shard_dir  Tucker shard 目录
  --out_dir    输出目录
  --n          抽几个样本(默认 8)

输出:
  out_dir/{idx}_{mesh_id}_orig.obj
  out_dir/{idx}_{mesh_id}_recon.obj
  out_dir/metrics.json
  out_dir/summary.txt
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


def reconstruct_voxel(C, U1, U2, U3):
    """从单样本 Tucker 因子还原 X = C ×_1 U_1 ×_2 U_2 ×_3 U_3"""
    X = torch.einsum("abc,ia,jb,kc->ijk", C.float(), U1.float(), U2.float(), U3.float())
    return X.cpu().numpy()


def mesh_from_voxel(X, level=0.0):
    """复刻 preprocessing 的 grid:linspace(-1, 1, N) → spacing = 2/(N-1)。"""
    N = X.shape[0]
    h = 2.0 / (N - 1)
    try:
        v, f, _, _ = marching_cubes(X, level=level, spacing=(h, h, h))
        v -= 1.0  # 平移到 [-1, 1]
        return trimesh.Trimesh(vertices=v, faces=f, process=False)
    except (ValueError, RuntimeError):
        return None


def normalize_to_unit_box(mesh, margin=0.9):
    """跟 utils.normalize_mesh 完全一致:体心 + margin=0.9。"""
    m = mesh.copy()
    m.apply_translation(-m.centroid)
    max_extent = float(np.max(m.extents))
    if max_extent <= 0:
        return mesh
    m.apply_scale(2.0 * margin / max_extent)
    return m


def chamfer_l2(p1, p2):
    t1 = cKDTree(p1); t2 = cKDTree(p2)
    d12, _ = t2.query(p1); d21, _ = t1.query(p2)
    return float((d12.mean() + d21.mean()) / 2)


def hausdorff(p1, p2):
    t1 = cKDTree(p1); t2 = cKDTree(p2)
    d12, _ = t2.query(p1); d21, _ = t1.query(p2)
    return float(max(d12.max(), d21.max()))


def inside_frac(X, level=0.0):
    return float((X < level).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--n_points", type=int, default=5000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 列出所有 shard
    shard_paths = sorted(glob.glob(os.path.join(args.shard_dir, "tucker_sdf_shard_*.pt")))
    if not shard_paths:
        print(f"[error] no shards in {args.shard_dir}", file=sys.stderr)
        sys.exit(1)

    # 先看第一个 shard 的 batch size 估算总数
    first = torch.load(shard_paths[0], map_location="cpu", weights_only=False)
    B = len(first["mesh_id"])
    total = B * len(shard_paths)  # 近似
    print(f"\n{'='*72}")
    print(f"  Tucker (N={first['N']}, R={first['rank']}) 重构 vs 原始 ABC mesh")
    print(f"  shard 数: {len(shard_paths)},  每个 {B} 个 mesh,  约 {total} 个总样本")
    print(f"  抽样: {args.n} 个均匀分布")
    print(f"{'='*72}\n")

    # 均匀挑 n 个全局 index → 转成 (shard_idx, within_idx)
    global_idxs = np.linspace(0, total - 1, args.n, dtype=int)

    metrics = []
    cached_shard = None
    cached_data = None

    for k, gi in enumerate(global_idxs):
        si = int(gi // B)
        wi = int(gi % B)
        if si != cached_shard:
            cached_data = torch.load(shard_paths[si], map_location="cpu", weights_only=False)
            cached_shard = si
        d = cached_data
        if wi >= len(d["mesh_id"]):
            wi = len(d["mesh_id"]) - 1
        mesh_id = str(d["mesh_id"][wi])
        mesh_path = d["mesh_path"][wi]

        # 1) 还原 voxel
        X_recon = reconstruct_voxel(d["C"][wi], d["U_1"][wi], d["U_2"][wi], d["U_3"][wi])
        m_recon = mesh_from_voxel(X_recon)
        if m_recon is None or len(m_recon.vertices) == 0:
            print(f"[{k+1}/{args.n}] {mesh_id}: MC 失败,跳过")
            continue

        # 2) 加载原始
        if not os.path.exists(mesh_path):
            print(f"[{k+1}/{args.n}] {mesh_id}: 原始 mesh 不存在 {mesh_path}")
            continue
        try:
            m_orig = trimesh.load(mesh_path, force="mesh")
        except Exception as e:
            print(f"[{k+1}/{args.n}] {mesh_id}: 加载失败 {e}")
            continue
        if len(m_orig.vertices) == 0:
            print(f"[{k+1}/{args.n}] {mesh_id}: 空 mesh,跳过")
            continue
        m_orig = normalize_to_unit_box(m_orig)

        # 3) 度量
        p_orig = m_orig.sample(args.n_points)
        p_recon = m_recon.sample(args.n_points)
        cd = chamfer_l2(p_orig, p_recon)
        hd = hausdorff(p_orig, p_recon)
        ifr = inside_frac(X_recon)

        rec = {
            "mesh_id": mesh_id,
            "mesh_path": mesh_path,
            "chamfer_l2": cd,
            "hausdorff": hd,
            "inside_frac_recon": ifr,
            "n_verts_orig": len(m_orig.vertices),
            "n_verts_recon": len(m_recon.vertices),
            "n_faces_orig": int(len(m_orig.faces)),
            "n_faces_recon": int(len(m_recon.faces)),
            "is_watertight_orig": bool(m_orig.is_watertight),
            "is_watertight_recon": bool(m_recon.is_watertight),
        }
        metrics.append(rec)

        m_orig.export(os.path.join(args.out_dir, f"{k:03d}_{mesh_id}_orig.obj"))
        m_recon.export(os.path.join(args.out_dir, f"{k:03d}_{mesh_id}_recon.obj"))

        print(f"[{k+1}/{args.n}] {mesh_id:>8s}  CD={cd:.5f}  HD={hd:.5f}  "
              f"inside={ifr*100:.1f}%  vts(orig/recon)={len(m_orig.vertices)}/{len(m_recon.vertices)}")

    if not metrics:
        print("\n[error] 没有成功对比")
        sys.exit(1)

    h = 2 / 256
    cd_mean = float(np.mean([m["chamfer_l2"] for m in metrics]))
    cd_std = float(np.std([m["chamfer_l2"] for m in metrics]))
    cd_min = float(np.min([m["chamfer_l2"] for m in metrics]))
    cd_max = float(np.max([m["chamfer_l2"] for m in metrics]))
    hd_mean = float(np.mean([m["hausdorff"] for m in metrics]))
    hd_max_all = float(np.max([m["hausdorff"] for m in metrics]))
    ifr_mean = float(np.mean([m["inside_frac_recon"] for m in metrics]))

    summary = {
        "n_compared": len(metrics),
        "voxel_size_h": h,
        "chamfer_l2": {"mean": cd_mean, "std": cd_std, "min": cd_min, "max": cd_max,
                       "mean_in_voxels": cd_mean / h},
        "hausdorff": {"mean": hd_mean, "max": hd_max_all, "mean_in_voxels": hd_mean / h},
        "inside_frac_recon": {"mean": ifr_mean, "abc_reference": 0.074},
        "samples": metrics,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    txt = f"""
{'='*72}
  汇总 (n={len(metrics)} 个样本)
{'='*72}

  Chamfer L2  均值: {cd_mean:.6f}  ± {cd_std:.6f}
              范围: [{cd_min:.6f}, {cd_max:.6f}]
              换算: {cd_mean/h:.2f} × 体素边长 h ({h:.5f})

  Hausdorff   均值: {hd_mean:.6f}  ({hd_mean/h:.2f} × h)
              最大: {hd_max_all:.6f}

  inside_frac 均值: {ifr_mean*100:.2f}%   (ABC 训练集 ~7.4%)

参考标尺(CD 用体素边长 h 度量):
  CD ≈ 1 × h    几乎看不出区别(平均偏移 1 体素)
  CD ≈ 3 × h    明显但能用(尖角圆化)
  CD > 10 × h   压缩损失明显
{'='*72}

文件:
  *_orig.obj   原始 ABC mesh(归一化到 [-1,1]³)
  *_recon.obj  Tucker R=24 还原 mesh
  metrics.json 完整数值

在本地用 MeshLab 同时打开两个文件,按 T 平移一个 Y 轴即可并排对比。
"""
    print(txt)
    with open(os.path.join(args.out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
