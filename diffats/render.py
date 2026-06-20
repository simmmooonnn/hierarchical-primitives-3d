"""render_views.py — Multi-view rendering of ABC meshes for conditional 3D-gen.

Output per mesh (in <out_dir>/<mesh_id>/):
  view{i:02d}_rgb.png     — Lambertian shaded RGB
  view{i:02d}_normal.png  — world-space normal map (R=nx+1)/2, etc.
  view{i:02d}_depth.png   — 16-bit depth (normalized per-image)

Default: 8 azimuth views (0/45/.../315°) at elevation 30°, distance 1.5.
"""
import os
# Try EGL first (GPU headless), fall back to OSMesa
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import argparse
import sys
import time
import numpy as np
import trimesh
import pyrender
from PIL import Image


def look_at_matrix(eye, target, up):
    """Camera-to-world transform (pyrender convention: -Z forward, +Y up)."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-9)
    up_c = np.cross(right, forward)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up_c
    c2w[:3, 2] = -forward          # -Z faces target in pyrender
    c2w[:3, 3] = eye
    return c2w


def spherical_camera_pose(azimuth_deg, elevation_deg, distance):
    azim = np.radians(azimuth_deg)
    elev = np.radians(elevation_deg)
    eye = np.array([
        distance * np.cos(elev) * np.sin(azim),
        distance * np.sin(elev),
        distance * np.cos(elev) * np.cos(azim),
    ])
    return look_at_matrix(eye, np.zeros(3), np.array([0., 1., 0.]))


def normalize_mesh(mesh):
    """Center at origin and scale to unit bounding sphere."""
    mesh.apply_translation(-mesh.centroid)
    radius = float(np.max(np.linalg.norm(mesh.vertices, axis=1)))
    if radius > 1e-9:
        mesh.apply_scale(1.0 / radius)
    return mesh


def render_one(mesh_path, out_dir, n_views=8, image_size=224, distance=1.6,
               elevation=30.0, log=True):
    os.makedirs(out_dir, exist_ok=True)

    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f'Bad mesh: {mesh_path}')
    mesh = normalize_mesh(mesh)
    mesh.fix_normals()
    vn = mesh.vertex_normals  # (V, 3) world-space normals

    # Two pyrender meshes:
    #  (a) lit: default gray material for RGB shading
    #  (b) normal: vertex-colored by (n+1)/2 → RGB encodes world normal
    lit_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)

    nm_mesh = mesh.copy()
    nm_colors = (np.clip(vn, -1.0, 1.0) + 1.0) * 0.5  # [0, 1]
    nm_mesh.visual.vertex_colors = (nm_colors * 255).astype(np.uint8)
    normal_mesh = pyrender.Mesh.from_trimesh(nm_mesh, smooth=True)

    scene_lit = pyrender.Scene(bg_color=[255, 255, 255, 0],
                               ambient_light=[0.3, 0.3, 0.3])
    scene_lit.add(lit_mesh)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.0)

    scene_nm = pyrender.Scene(bg_color=[128, 128, 128, 0],
                              ambient_light=[1.0, 1.0, 1.0])
    scene_nm.add(normal_mesh)

    r = pyrender.OffscreenRenderer(viewport_width=image_size,
                                   viewport_height=image_size)
    azimuths = np.linspace(0, 360, n_views, endpoint=False)

    t0 = time.time()
    for i, azim in enumerate(azimuths):
        pose = spherical_camera_pose(azim, elevation, distance)
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

        # RGB (lit) + depth in one pass
        cn = scene_lit.add(cam, pose=pose)
        ln = scene_lit.add(light, pose=pose)
        rgb, depth = r.render(scene_lit)
        scene_lit.remove_node(cn)
        scene_lit.remove_node(ln)

        Image.fromarray(rgb).save(os.path.join(out_dir, f'view{i:02d}_rgb.png'))

        # Depth → 16-bit PNG (normalized per-image so far-pixel ≠ zero)
        valid = depth > 0
        if valid.any():
            d_min, d_max = depth[valid].min(), depth[valid].max()
            depth_norm = np.zeros_like(depth)
            depth_norm[valid] = (depth[valid] - d_min) / max(d_max - d_min, 1e-9)
            depth_u16 = (depth_norm * 65535).astype(np.uint16)
        else:
            depth_u16 = np.zeros_like(depth, dtype=np.uint16)
        Image.fromarray(depth_u16).save(os.path.join(out_dir, f'view{i:02d}_depth.png'))

        # Normal map (unlit, vertex-colored)
        cn = scene_nm.add(cam, pose=pose)
        nrm, _ = r.render(scene_nm)
        scene_nm.remove_node(cn)
        Image.fromarray(nrm).save(os.path.join(out_dir, f'view{i:02d}_normal.png'))

    r.delete()
    t = time.time() - t0
    if log:
        print(f'{os.path.basename(mesh_path)}: {n_views} views in {t:.3f}s '
              f'({t/n_views*1000:.0f} ms/view)', flush=True)
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mesh', nargs='+', required=True)
    p.add_argument('--out_root', required=True)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--n_views', type=int, default=8)
    p.add_argument('--distance', type=float, default=1.6)
    p.add_argument('--elevation', type=float, default=30.0)
    args = p.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    total = 0.0
    for mp in args.mesh:
        # Use the file stem as the per-mesh out dir
        stem = os.path.splitext(os.path.basename(mp))[0]
        out = os.path.join(args.out_root, stem)
        total += render_one(mp, out, n_views=args.n_views, image_size=args.size,
                            distance=args.distance, elevation=args.elevation)
    print(f'\nTOTAL: {len(args.mesh)} meshes, '
          f'{total:.2f}s, {total/max(len(args.mesh),1):.2f}s/mesh avg')


if __name__ == '__main__':
    main()
