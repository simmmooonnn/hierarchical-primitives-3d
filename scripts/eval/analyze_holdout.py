"""Analyze hold-out CFG sweep results: CD per CFG → find optimal.

Reads:
  outputs/holdout/mesh_list.txt           — list of (gt_path, V, F)
  outputs/holdout/samples_cfg*/pred_*.obj — predictions per CFG value
"""
import os, sys, trimesh, numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ.get('DIFFATS_ROOT', os.getcwd())
H = os.path.join(ROOT, 'outputs', 'holdout')
LIST = os.path.join(H, 'mesh_list.txt')
CFGS = ['1.0', '1.5', '2.0', '3.0', '5.0', '7.0']

def normalize(m):
    m = m.copy()
    m.apply_translation(-m.centroid)
    r = float(np.max(np.linalg.norm(m.vertices, axis=1)))
    if r > 1e-9: m.apply_scale(1.0 / r)
    return m

def chamfer(a, b, n=2000):
    pa = a.sample(n); pb = b.sample(n)
    ta, tb = cKDTree(pa), cKDTree(pb)
    return 0.5 * (tb.query(pa)[0].mean() + ta.query(pb)[0].mean())

def big_comp(m):
    cs = m.split(only_watertight=False)
    return normalize(max(cs, key=lambda c: len(c.vertices))) if len(cs) > 1 else m

# Parse list
mids = []
gts  = []
with open(LIST) as f:
    for ln in f:
        parts = ln.strip().split('\t')
        rel = parts[0]
        mid = os.path.splitext(os.path.basename(rel))[0]
        mids.append(mid)
        gts.append(os.path.join(ROOT, rel))
print(f'hold-out meshes: {len(mids)}', flush=True)

gt_meshes = [normalize(trimesh.load(p, force='mesh')) for p in gts]

results = {}
for cfg in CFGS:
    d = os.path.join(H, f'samples_cfg{cfg}')
    cds = []
    for mid, gt in zip(mids, gt_meshes):
        pf = os.path.join(d, f'pred_{mid}.obj')
        if not os.path.exists(pf):
            cds.append(np.nan)
            continue
        try:
            m = trimesh.load(pf, force='mesh')
            cds.append(chamfer(big_comp(m), gt))
        except Exception as e:
            cds.append(np.nan)
    results[cfg] = np.array(cds)
    valid = ~np.isnan(cds)
    print(f'CFG={cfg:>4}: mean CD = {np.nanmean(cds):.4f}  '
          f'median = {np.nanmedian(cds):.4f}  '
          f'valid {valid.sum()}/{len(cds)}',
          flush=True)

print()
print('=== Per-mesh CD across CFG ===')
print(f'{"mesh_id":>40} | ' + ' '.join([f'{c:>6}' for c in CFGS]) + ' | best')
print('-' * (50 + 7*len(CFGS) + 10))
for i, mid in enumerate(mids):
    row = [results[c][i] for c in CFGS]
    best_idx = int(np.nanargmin(row))
    short = mid[:38]
    print(f'{short:>40} | ' + ' '.join([f'{r:>6.4f}' for r in row])
          + f' | CFG={CFGS[best_idx]}')

print()
print('=== Best CFG by per-mesh vote ===')
best_cfg_per_mesh = []
for i in range(len(mids)):
    row = [results[c][i] for c in CFGS]
    best_cfg_per_mesh.append(CFGS[int(np.nanargmin(row))])
from collections import Counter
cnt = Counter(best_cfg_per_mesh)
for c in CFGS:
    print(f'  CFG={c}: best for {cnt.get(c, 0)} meshes')
