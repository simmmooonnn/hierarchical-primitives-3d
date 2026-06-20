"""render_batch.py — Render all 10k ABC meshes from the mesh_id_to_path.json map.

Output layout:
  <out_root>/<mesh_id>/view{i:02d}_rgb.png
                      /view{i:02d}_normal.png
                      /view{i:02d}_depth.png

where <mesh_id> is the 5-digit zero-padded sequential index matching Tucker shards.

Skips meshes whose output dir already has all 24 expected PNGs (resumable).
"""
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import argparse
import glob
import json
import sys
import time
import numpy as np
import trimesh
import pyrender
from PIL import Image


# ----- Camera + rendering helpers -----

def look_at_matrix(eye, target, up):
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-9)
    up_c = np.cross(right, forward)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up_c
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye
    return c2w


def spherical_camera_pose(azim_deg, elev_deg, distance):
    a = np.radians(azim_deg)
    e = np.radians(elev_deg)
    eye = np.array([distance * np.cos(e) * np.sin(a),
                    distance * np.sin(e),
                    distance * np.cos(e) * np.cos(a)])
    return look_at_matrix(eye, np.zeros(3), np.array([0., 1., 0.]))


def normalize_mesh(mesh):
    mesh.apply_translation(-mesh.centroid)
    r = float(np.max(np.linalg.norm(mesh.vertices, axis=1)))
    if r > 1e-9:
        mesh.apply_scale(1.0 / r)
    return mesh


def render_one(mesh_path, out_dir, renderer, n_views, image_size, distance, elevation):
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f'Bad mesh: {mesh_path}')
    mesh = normalize_mesh(mesh)
    mesh.fix_normals()
    vn = mesh.vertex_normals

    lit_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    nm_mesh_t = mesh.copy()
    nm_colors = (np.clip(vn, -1.0, 1.0) + 1.0) * 0.5
    nm_mesh_t.visual.vertex_colors = (nm_colors * 255).astype(np.uint8)
    normal_mesh = pyrender.Mesh.from_trimesh(nm_mesh_t, smooth=True)

    scene_lit = pyrender.Scene(bg_color=[255, 255, 255, 0],
                               ambient_light=[0.5, 0.5, 0.5])
    scene_lit.add(lit_mesh)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.0)

    scene_nm = pyrender.Scene(bg_color=[128, 128, 128, 0],
                              ambient_light=[1.0, 1.0, 1.0])
    scene_nm.add(normal_mesh)

    azimuths = np.linspace(0, 360, n_views, endpoint=False)
    for i, azim in enumerate(azimuths):
        pose = spherical_camera_pose(azim, elevation, distance)
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

        cn = scene_lit.add(cam, pose=pose)
        ln = scene_lit.add(light, pose=pose)
        rgb, depth = renderer.render(scene_lit)
        scene_lit.remove_node(cn)
        scene_lit.remove_node(ln)
        Image.fromarray(rgb).save(os.path.join(out_dir, f'view{i:02d}_rgb.png'))

        valid = depth > 0
        if valid.any():
            d_min, d_max = float(depth[valid].min()), float(depth[valid].max())
            depth_norm = np.zeros_like(depth)
            depth_norm[valid] = (depth[valid] - d_min) / max(d_max - d_min, 1e-9)
            depth_u16 = (depth_norm * 65535).astype(np.uint16)
        else:
            depth_u16 = np.zeros_like(depth, dtype=np.uint16)
        Image.fromarray(depth_u16).save(os.path.join(out_dir, f'view{i:02d}_depth.png'))

        cn = scene_nm.add(cam, pose=pose)
        nrm, _ = renderer.render(scene_nm)
        scene_nm.remove_node(cn)
        Image.fromarray(nrm).save(os.path.join(out_dir, f'view{i:02d}_normal.png'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mesh_id_map', required=True,
                   help='Path to mesh_id_to_path.json')
    p.add_argument('--out_root', required=True)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--n_views', type=int, default=8)
    p.add_argument('--distance', type=float, default=1.6)
    p.add_argument('--elevation', type=float, default=30.0)
    p.add_argument('--start', type=int, default=0,
                   help='Inclusive start index (in sorted mesh_id order)')
    p.add_argument('--end', type=int, default=-1,
                   help='Exclusive end index; -1 = all')
    p.add_argument('--log_every', type=int, default=100)
    args = p.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    with open(args.mesh_id_map) as f:
        m = json.load(f)
    ids = sorted(m.keys())
    if args.end < 0:
        args.end = len(ids)
    ids = ids[args.start:args.end]
    print(f'Total mesh_ids to render: {len(ids)} (range [{args.start}, {args.end}))', flush=True)

    r = pyrender.OffscreenRenderer(viewport_width=args.size, viewport_height=args.size)
    expected_files = 3 * args.n_views  # rgb + normal + depth per view

    t_start = time.time()
    done = 0
    skipped = 0
    errors = []
    for k, mid in enumerate(ids):
        out_dir = os.path.join(args.out_root, mid)
        if os.path.isdir(out_dir):
            existing = len([f for f in os.listdir(out_dir) if f.endswith('.png')])
            if existing >= expected_files:
                skipped += 1
                continue
        os.makedirs(out_dir, exist_ok=True)
        try:
            render_one(m[mid], out_dir, r, args.n_views, args.size,
                       args.distance, args.elevation)
            done += 1
        except Exception as e:
            errors.append((mid, str(e)))
            print(f'  ERR {mid}: {e}', flush=True)

        if (k + 1) % args.log_every == 0:
            el = time.time() - t_start
            rate = (k + 1) / el if el > 0 else 0
            remain_s = (len(ids) - k - 1) / max(rate, 1e-6)
            print(f'  {k+1}/{len(ids)}  done={done} skip={skipped} err={len(errors)}  '
                  f'{rate:.1f} mesh/s  ETA {remain_s/60:.1f} min', flush=True)

    r.delete()
    total = time.time() - t_start
    print()
    print(f'DONE: {done} rendered, {skipped} skipped, {len(errors)} errors  '
          f'(elapsed {total/60:.1f} min)', flush=True)
    if errors:
        print('Error sample:')
        for mid, e in errors[:5]:
            print(f'  {mid}: {e}')


if __name__ == '__main__':
    main()
