"""Bundle 10k per-mesh DINOv2 .pt files into a single tensor for fast load.

Output: outputs/dinov2_embeddings_bundle.pt
   - 'embs':     (N, 8, 257, 1024) fp16
   - 'mesh_ids': list of N strings (5-digit zero-padded)
"""
import os, glob, time, argparse
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir',  required=True)
    p.add_argument('--out_file', required=True)
    p.add_argument('--shape',   default='8,257,1024',
                   help='comma-separated expected per-mesh shape')
    args = p.parse_args()

    shape = tuple(int(x) for x in args.shape.split(','))
    files = sorted(glob.glob(os.path.join(args.in_dir, '*.pt')))
    print(f'files: {len(files)}', flush=True)

    n = len(files)
    out = torch.empty((n, *shape), dtype=torch.float16)
    mesh_ids = []
    t0 = time.time()
    for i, fp in enumerate(files):
        mid = os.path.splitext(os.path.basename(fp))[0]
        e = torch.load(fp, map_location='cpu', weights_only=False)
        if e.dtype != torch.float16:
            e = e.to(torch.float16)
        out[i] = e
        mesh_ids.append(mid)
        if (i + 1) % 1000 == 0:
            print(f'  {i+1}/{n}  elapsed {time.time()-t0:.0f}s', flush=True)

    print(f'saving {args.out_file} ({n * shape[0] * shape[1] * shape[2] * 2 / 1e9:.1f} GB) ...', flush=True)
    torch.save({'embs': out, 'mesh_ids': mesh_ids}, args.out_file)
    print(f'DONE in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
