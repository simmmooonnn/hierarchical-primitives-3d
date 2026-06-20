"""
build_dataset.py
扩充版数据集 manifest 构建器,过滤极端 inside_frac。

输入: abc_obj/obj/ 下所有 .obj
过滤链:
  1. faces ∈ [500, 2M]
  2. aspect_ratio >= 0.005(最薄边 / 最厚边)
  3. inside_frac ∈ [inside_frac_min, inside_frac_max]  ← 新增

inside_frac 用 trimesh proximity 在 N=32 grid 上估算(慢但稳)。
输出:meshes_v2.json,格式跟 meshes_relaxed_anvil.json 兼容。
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import trimesh
from pysdf import SDF

MIN_FACES = 500
MAX_FACES = 2_000_000
MIN_ASPECT = 0.005


def _make_grid(N=32, margin=0.9):
    """[-margin, margin]^3 的 N^3 grid,跟 utils.normalize_mesh 用同 margin。"""
    lin = np.linspace(-margin, margin, N)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    return np.stack([X, Y, Z], axis=-1).reshape(-1, 3)


GRID = None


def _init_worker(N, margin):
    global GRID
    GRID = _make_grid(N=N, margin=margin)


def _normalize(mesh, margin=0.9):
    m = mesh.copy()
    m.apply_translation(-m.centroid)
    e = float(np.max(m.extents))
    if e <= 0:
        return None
    m.apply_scale(2.0 * margin / e)
    return m


def _check_one(mesh_path):
    try:
        mesh = trimesh.load(mesh_path, force="mesh", skip_materials=True)
        if mesh is None or len(mesh.faces) == 0:
            return {"ok": False, "reason": "empty", "mesh_path": mesh_path}
        nv, nf = int(len(mesh.vertices)), int(len(mesh.faces))
        if nf < MIN_FACES or nf > MAX_FACES:
            return {"ok": False, "reason": "faces", "n_faces": nf, "mesh_path": mesh_path}
        ext = [float(x) for x in mesh.extents]
        aspect = float(min(ext) / max(ext))
        if aspect < MIN_ASPECT:
            return {"ok": False, "reason": "aspect", "aspect": aspect, "mesh_path": mesh_path}

        m = _normalize(mesh)
        if m is None:
            return {"ok": False, "reason": "norm_fail", "mesh_path": mesh_path}

        # pysdf 对非 watertight 也稳健;raw 输出 "inside = positive"
        sdf_func = SDF(np.asarray(m.vertices, dtype=np.float32),
                       np.asarray(m.faces, dtype=np.uint32))
        sdf_vals = sdf_func(GRID.astype(np.float32))
        inside = int((sdf_vals > 0).sum())
        ifr = float(inside / GRID.shape[0])

        return {
            "ok": True,
            "mesh_path": mesh_path,
            "vertices": nv,
            "faces": nf,
            "aspect_ratio": aspect,
            "extents": ext,
            "inside_frac": ifr,
            "watertight": bool(mesh.is_watertight),
        }
    except Exception as e:
        return {"ok": False, "reason": f"exc:{type(e).__name__}:{str(e)[:80]}", "mesh_path": mesh_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj_dir", required=True, help="abc_obj/obj/")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--inside_frac_min", type=float, default=0.02)
    ap.add_argument("--inside_frac_max", type=float, default=0.25)
    ap.add_argument("--grid_N", type=int, default=32)
    ap.add_argument("--margin", type=float, default=0.9)
    ap.add_argument("--n_workers", type=int, default=32)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.obj_dir, "*/*.obj")))
    print(f"Found {len(paths)} .obj files in {args.obj_dir}", flush=True)
    if not paths:
        print("[error] no .obj found", file=sys.stderr); sys.exit(1)

    print(f"Processing with {args.n_workers} workers, grid {args.grid_N}³,"
          f" inside_frac filter [{args.inside_frac_min}, {args.inside_frac_max}]", flush=True)

    with Pool(args.n_workers, initializer=_init_worker,
              initargs=(args.grid_N, args.margin)) as pool:
        results = []
        from tqdm import tqdm
        for r in tqdm(pool.imap_unordered(_check_one, paths, chunksize=4),
                      total=len(paths), desc="checking"):
            results.append(r)

    ok = [r for r in results if r["ok"]]
    valid = [r for r in ok
             if args.inside_frac_min <= r["inside_frac"] <= args.inside_frac_max]
    valid.sort(key=lambda r: r["mesh_path"])
    for i, r in enumerate(valid):
        r["id"] = f"{i:05d}"
        r["idx"] = i

    print()
    print(f"  Total checked:           {len(results)}")
    print(f"  Pass sanity (faces+asp): {len(ok)} ({len(ok)/len(results)*100:.1f}%)")
    print(f"  + inside_frac filter:    {len(valid)} ({len(valid)/len(results)*100:.1f}%)")

    if ok:
        inside_arr = np.array([r["inside_frac"] for r in ok])
        print(f"\n  inside_frac stats (pre-filter):")
        print(f"    mean: {inside_arr.mean():.4f}")
        print(f"    median: {np.median(inside_arr):.4f}")
        print(f"    in [0.02, 0.25]: {((inside_arr >= 0.02) & (inside_arr <= 0.25)).mean()*100:.1f}%")
        print(f"    in [0.01, 0.30]: {((inside_arr >= 0.01) & (inside_arr <= 0.30)).mean()*100:.1f}%")
        print(f"    in [0.005, 0.40]: {((inside_arr >= 0.005) & (inside_arr <= 0.40)).mean()*100:.1f}%")

    out = {
        "config": {
            "abc_root": args.obj_dir,
            "n_total_obj": len(paths),
            "n_passed_sanity": len(ok),
            "n_passed_full": len(valid),
            "MIN_FACES": MIN_FACES,
            "MAX_FACES": MAX_FACES,
            "MIN_ASPECT": MIN_ASPECT,
            "INSIDE_FRAC_MIN": args.inside_frac_min,
            "INSIDE_FRAC_MAX": args.inside_frac_max,
            "grid_N": args.grid_N,
        },
        "pilot": valid,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved {len(valid)} valid meshes → {args.out_json}")


if __name__ == "__main__":
    main()
