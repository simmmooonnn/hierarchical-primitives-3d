"""cache_dinov2_embeddings.py — Encode 10k mesh × 8 view normal maps via DINOv2-L.

Output per mesh: <out_root>/<mesh_id>.pt
  shape (8, 257, 1024) float16 — (view, token, dim)
  Tokens: 1 CLS + 16×16 = 256 patches (patch_size=14, input=224×224).

DINOv2 reference: Oquab et al. 2023, https://huggingface.co/facebook/dinov2-large
"""
import os
import argparse
import glob
import time
import math

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--render_root', required=True,
                   help='Directory with one subfolder per mesh_id, each with viewNN_normal.png')
    p.add_argument('--out_root', required=True)
    p.add_argument('--model', default='facebook/dinov2-large')
    p.add_argument('--batch_meshes', type=int, default=8,
                   help='# meshes per batch (each mesh = 8 view images)')
    p.add_argument('--n_views', type=int, default=8)
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end',   type=int, default=-1)
    p.add_argument('--log_every', type=int, default=100)
    args = p.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}', flush=True)
    print(f'loading {args.model} ...', flush=True)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    model = model.to(torch.bfloat16)
    print(f'  params: {sum(p.numel() for p in model.parameters()):,}', flush=True)

    # ImageNet-standard preprocessing (DINOv2 uses this)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    mesh_dirs = sorted([
        d for d in glob.glob(os.path.join(args.render_root, '*'))
        if os.path.isdir(d)
    ])
    if args.end < 0:
        args.end = len(mesh_dirs)
    mesh_dirs = mesh_dirs[args.start:args.end]
    print(f'meshes: {len(mesh_dirs)} (range [{args.start}, {args.end}))', flush=True)

    # Probe one forward to learn token / dim
    probe = Image.open(os.path.join(mesh_dirs[0], 'view00_normal.png')).convert('RGB')
    probe = probe.resize((args.image_size, args.image_size))
    pa = torch.tensor(np.array(probe)).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        x = ((pa.to(device).unsqueeze(0) - mean) / std).to(torch.bfloat16)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(pixel_values=x).last_hidden_state
    n_tokens, d_model = out.shape[1], out.shape[2]
    print(f'  output tokens={n_tokens}, dim={d_model}', flush=True)

    expected_files = len(mesh_dirs)
    t_start = time.time()
    done = 0
    skipped = 0
    err_count = 0

    n_batches = math.ceil(len(mesh_dirs) / args.batch_meshes)
    for bi in range(n_batches):
        batch_dirs = mesh_dirs[bi*args.batch_meshes:(bi+1)*args.batch_meshes]
        # Skip if all outputs already exist
        out_paths = [os.path.join(args.out_root,
                                  os.path.basename(d) + '.pt')
                     for d in batch_dirs]
        if all(os.path.exists(p) for p in out_paths):
            skipped += len(batch_dirs)
            continue

        # Load 8 view normal maps per mesh
        imgs = []
        valid_mask = []
        for d in batch_dirs:
            try:
                for v in range(args.n_views):
                    pth = os.path.join(d, f'view{v:02d}_normal.png')
                    img = Image.open(pth).convert('RGB')
                    if img.size != (args.image_size, args.image_size):
                        img = img.resize((args.image_size, args.image_size))
                    arr = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
                    imgs.append(arr)
                valid_mask.append(True)
            except Exception as e:
                print(f'  ERR loading {d}: {e}', flush=True)
                valid_mask.append(False)
                # pad with zeros
                for _ in range(args.n_views):
                    imgs.append(torch.zeros(3, args.image_size, args.image_size))
                err_count += 1

        x = torch.stack(imgs).to(device, non_blocking=True)        # (B*8, 3, H, W)
        x = ((x - mean) / std).to(torch.bfloat16)
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(pixel_values=x).last_hidden_state          # (B*8, T, D)
        out = out.view(len(batch_dirs), args.n_views, n_tokens, d_model)
        out = out.detach().to(torch.float16).cpu()

        for d, op, ok, emb in zip(batch_dirs, out_paths, valid_mask, out):
            if not ok:
                continue
            torch.save(emb, op)
            done += 1

        if (bi + 1) % max(1, args.log_every // args.batch_meshes) == 0:
            el = time.time() - t_start
            processed = done + skipped
            rate = processed / max(el, 1e-6)
            remain = (len(mesh_dirs) - processed) / max(rate, 1e-6)
            print(f'  {processed}/{len(mesh_dirs)}  done={done} skip={skipped} err={err_count}  '
                  f'{rate:.1f} mesh/s  ETA {remain/60:.1f} min', flush=True)

    total = time.time() - t_start
    print()
    print(f'DONE: {done} cached, {skipped} skipped, {err_count} errors  '
          f'(elapsed {total/60:.1f} min)', flush=True)


if __name__ == '__main__':
    main()
