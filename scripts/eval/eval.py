"""SOTA eval v2 — fixes Volume IoU (uses contains() for non-watertight) and adds 1-NNA.

Metrics:
  - Chamfer Distance (CD)
  - F-Score @ τ ∈ {0.01, 0.05}
  - Volume IoU              (FIXED: mesh.contains() instead of signed_distance)
  - CLIP-Score              (text vs render)
  - 1-NNA-CD                (PointFlow ICLR'19; lower is better, 50% optimal)

Output: outputs/sota_eval/metrics_v2.json
"""
import os, sys, json, time
import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import pyrender

from diffats.render import normalize_mesh, spherical_camera_pose
from transformers import CLIPModel, CLIPProcessor

ROOT     = os.environ.get('DIFFATS_ROOT', os.getcwd())
H        = os.path.join(ROOT, 'outputs', 'holdout')
OUT_DIR  = os.path.join(ROOT, 'outputs', 'sota_eval')
os.makedirs(OUT_DIR, exist_ok=True)


def normalize(m):
    m = m.copy()
    m.apply_translation(-m.centroid)
    r = float(np.max(np.linalg.norm(m.vertices, axis=1)))
    if r > 1e-9: m.apply_scale(1.0 / r)
    return m

def big_comp(m):
    cs = m.split(only_watertight=False)
    return normalize(max(cs, key=lambda c: len(c.vertices))) if len(cs) > 1 else m

_POINT_CACHE = {}
def cached_points(mesh, mesh_id, n=2000):
    if mesh_id not in _POINT_CACHE:
        _POINT_CACHE[mesh_id] = mesh.sample(n)
    return _POINT_CACHE[mesh_id]

def chamfer(p_a, p_b):
    """Chamfer Distance from pre-sampled point clouds."""
    ta, tb = cKDTree(p_a), cKDTree(p_b)
    return 0.5 * (tb.query(p_a)[0].mean() + ta.query(p_b)[0].mean())


def chamfer_and_fscore(p_pred, p_gt, taus=(0.01, 0.05)):
    ta, tb = cKDTree(p_pred), cKDTree(p_gt)
    d_p2g = tb.query(p_pred)[0]
    d_g2p = ta.query(p_gt)[0]
    cd = 0.5 * (d_p2g.mean() + d_g2p.mean())
    fs = {}
    for t in taus:
        p = float((d_p2g < t).mean())
        r = float((d_g2p < t).mean())
        fs[t] = 2 * p * r / max(p + r, 1e-9)
    return cd, fs


def volume_iou(pred, gt, grid_res=48):
    """Volume IoU via ray-casting contains() — robust on non-watertight meshes."""
    coords = np.linspace(-1.0, 1.0, grid_res)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    try:
        in_p = pred.contains(pts)
        in_g = gt.contains(pts)
    except Exception as e:
        return float('nan')
    union = int((in_p | in_g).sum())
    if union == 0:
        return 0.0
    inter = int((in_p & in_g).sum())
    return inter / union


def one_nna_cd(gen_points_list, real_points_list):
    """1-NNA via Chamfer. labels = 'gen' or 'real'. Lower better; 50% optimal."""
    n = len(gen_points_list); m = len(real_points_list)
    N = n + m
    all_pts = gen_points_list + real_points_list
    labels = ['gen'] * n + ['real'] * m

    D = np.full((N, N), np.inf)
    for i in range(N):
        for j in range(i + 1, N):
            d = chamfer(all_pts[i], all_pts[j])
            D[i, j] = d; D[j, i] = d

    correct = 0
    for i in range(N):
        j_nn = int(np.argmin(D[i]))
        if labels[j_nn] == labels[i]:
            correct += 1
    return correct / N

class ClipRenderer:
    def __init__(self, device, img_size=224, n_views=4, distance=1.6, elevation=20.0):
        self.device = device; self.img_size = img_size
        self.n_views = n_views; self.distance = distance; self.elevation = elevation
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_size,
                                                  viewport_height=img_size)
        self.azimuths = np.linspace(0, 360, n_views, endpoint=False)
        self.clip = CLIPModel.from_pretrained(
            'openai/clip-vit-large-patch14', torch_dtype=torch.float32).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')

    def _render(self, mesh):
        m = normalize_mesh(mesh.copy()); m.fix_normals()
        pm = pyrender.Mesh.from_trimesh(m, smooth=True)
        scene = pyrender.Scene(bg_color=[255, 255, 255, 0],
                               ambient_light=[0.5, 0.5, 0.5])
        scene.add(pm)
        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.0)
        imgs = []
        for a in self.azimuths:
            pose = spherical_camera_pose(a, self.elevation, self.distance)
            cam = pyrender.PerspectiveCamera(yfov=np.pi/3.0, aspectRatio=1.0)
            cn = scene.add(cam, pose=pose); ln = scene.add(light, pose=pose)
            rgb, _ = self.renderer.render(scene)
            scene.remove_node(cn); scene.remove_node(ln)
            imgs.append(Image.fromarray(rgb))
        return imgs

    @torch.no_grad()
    def clip_text_score(self, mesh, prompt):
        imgs = self._render(mesh)
        inp = self.proc(text=[prompt], images=imgs, return_tensors='pt',
                        padding=True, truncation=True).to(self.device)
        out = self.clip(**inp)
        ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        te = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
        return float((ie @ te.T).squeeze(1).mean().cpu())

def main():
    LIST = os.path.join(H, 'mesh_list.txt')
    mids, gt_paths = [], []
    with open(LIST) as f:
        for ln in f:
            rel = ln.strip().split('\t')[0]
            mid = os.path.splitext(os.path.basename(rel))[0]
            mids.append(mid)
            gt_paths.append(os.path.join(ROOT, rel))

    print(f'loading {len(mids)} GT meshes ...', flush=True)
    gt_meshes = [normalize(trimesh.load(p, force='mesh')) for p in gt_paths]
    gt_points = [cached_points(g, f'gt_{i}', n=2000) for i, g in enumerate(gt_meshes)]

    with open(os.path.join(ROOT, 'outputs', 'captions', 'captions.json')) as f:
        captions = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('building CLIP-L + PyRender ...', flush=True)
    clip_r = ClipRenderer(device)

    MODELS = [
        ('v9_image_cfg3',   os.path.join(H, 'samples_cfg3.0')),
        ('v10_text_cfg3',   os.path.join(H, 'v10_samples_cfg3.0')),
    ]

    results = {m: {} for m, _ in MODELS}
    summary_per_model = {}

    for mname, mdir in MODELS:
        print(f'\n========== {mname} ==========', flush=True)
        gen_points = []     # for 1-NNA
        for i, (mid, gt) in enumerate(zip(mids, gt_meshes)):
            t0 = time.time()
            pred_p = os.path.join(mdir, f'pred_{mid}.obj')
            if not os.path.exists(pred_p):
                print(f'  [{i+1:>2}] MISSING', flush=True); continue
            pred = big_comp(trimesh.load(pred_p, force='mesh'))

            p_pred = pred.sample(2000)
            gen_points.append(p_pred)
            p_gt = gt_points[i]

            cd, fs = chamfer_and_fscore(p_pred, p_gt)

            # Volume IoU via contains()
            try:
                viou = volume_iou(pred, gt, grid_res=48)
            except Exception as e:
                viou = float('nan')

            # CLIP text score
            try:
                clip_text = clip_r.clip_text_score(pred, captions.get(mid, mid))
            except Exception:
                clip_text = float('nan')

            results[mname][mid] = {
                'CD':         cd,
                'F@0.01':     fs[0.01],
                'F@0.05':     fs[0.05],
                'VolIoU':     viou,
                'CLIP-text':  clip_text,
            }
            print(f'  [{i+1:>2}] {mid[:30]:<30}  CD={cd:.4f}  F@1%={fs[0.01]:.3f}  '
                  f'F@5%={fs[0.05]:.3f}  VIoU={viou:.3f}  CLIP={clip_text:.3f}  '
                  f'[{time.time()-t0:.1f}s]', flush=True)

        # 1-NNA: compare 16 generated vs 16 GT
        print(f'\n  computing 1-NNA-CD ...', flush=True)
        t0 = time.time()
        nna = one_nna_cd(gen_points, gt_points)
        print(f'  1-NNA-CD = {nna*100:.1f}%  (50% = optimal, lower better)  '
              f'[{time.time()-t0:.1f}s]', flush=True)
        summary_per_model[mname] = nna

    out_payload = {'metrics': results, '1-NNA-CD': summary_per_model}
    with open(os.path.join(OUT_DIR, 'metrics_v2.json'), 'w') as f:
        json.dump(out_payload, f, indent=2)
    print(f'\nwrote {OUT_DIR}/metrics_v2.json', flush=True)

    print('\n========== FINAL SUMMARY ==========', flush=True)
    keys = ['CD', 'F@0.01', 'F@0.05', 'VolIoU', 'CLIP-text']
    print(f'{"model":<18} {"CD↓":>8} {"F@1%↑":>8} {"F@5%↑":>8} {"VIoU↑":>8} {"CLIP↑":>8} {"1-NNA":>8}', flush=True)
    print('-' * 75, flush=True)
    for mname, _ in MODELS:
        rows = list(results[mname].values())
        if not rows: continue
        means = {k: np.nanmean([r[k] for r in rows]) for k in keys}
        print(f'{mname:<18} '
              f'{means["CD"]:>8.4f} '
              f'{means["F@0.01"]:>8.3f} '
              f'{means["F@0.05"]:>8.3f} '
              f'{means["VolIoU"]:>8.3f} '
              f'{means["CLIP-text"]:>8.3f} '
              f'{summary_per_model[mname]*100:>7.1f}%', flush=True)


if __name__ == '__main__':
    main()
