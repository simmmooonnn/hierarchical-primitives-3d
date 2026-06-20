"""
decimate_samples.py — Decimate selected v4 sample meshes for download.

Picks 3 samples spanning inside_frac (low/mid/high), decimates to a target
face count, saves as PLY (binary, ~10x smaller than OBJ).
"""

import argparse
import os
import sys

import trimesh


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir',  required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--samples', type=int, nargs='+', default=[0, 4, 13],
                   help='sample indices to decimate (default low/mid/low)')
    p.add_argument('--target_faces', type=int, default=100_000)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for idx in args.samples:
        in_path  = os.path.join(args.in_dir,  f'sample_{idx:02d}.obj')
        out_path = os.path.join(args.out_dir, f'sample_{idx:02d}_dec.ply')
        if not os.path.exists(in_path):
            print(f'  skip sample {idx}: {in_path} not found')
            continue
        print(f'Loading {in_path} ...')
        m = trimesh.load(in_path, force='mesh')
        nV, nF = len(m.vertices), len(m.faces)
        print(f'  original: V={nV:>10}  F={nF:>10}')

        if nF > args.target_faces:
            ratio = args.target_faces / nF
            try:
                m_dec = m.simplify_quadric_decimation(face_count=args.target_faces)
                print(f'  decimated (quadric, target {args.target_faces}):'
                      f'  V={len(m_dec.vertices):>10}  F={len(m_dec.faces):>10}')
            except Exception as e:
                print(f'  quadric decimation failed: {e}')
                print(f'  falling back to random face subsample')
                import numpy as np
                keep = np.random.choice(nF, args.target_faces, replace=False)
                m_dec = trimesh.Trimesh(vertices=m.vertices, faces=m.faces[keep])
                m_dec.remove_unreferenced_vertices()
                print(f'  subsampled: V={len(m_dec.vertices):>10}  F={len(m_dec.faces):>10}')
        else:
            m_dec = m

        m_dec.export(out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f'  saved {out_path}  ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
