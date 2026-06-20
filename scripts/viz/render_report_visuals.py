"""Render report visuals v2 — better aspect ratios + unconditional baseline samples.

Layouts redesigned for Word embedding (wide-ish, < 2× page height):
  - holdout_pairs_wide.png    : 2 rows (GT, Pred) × 5 cols           (best for Word column)
  - phase3_demo_wide.png      : 3 cols × 2 rows (6 demos)
  - qwen_vs_claude_wide.png   : 3 cols × 2 rows × 2 sub-cells
  - unconditional_grid.png    : 3 cols × 2 rows (6 v7.4b samples, NEW)
  - claude_failure_demo.png   : (keep from v1 — already good)
"""
import os, sys, json
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np
import trimesh
import pyrender
from PIL import Image, ImageDraw, ImageFont

from diffats.render import normalize_mesh, spherical_camera_pose

ROOT    = os.environ.get('DIFFATS_ROOT', os.getcwd())
H       = os.path.join(ROOT, 'outputs', 'holdout')
OUT_DIR = os.path.join(ROOT, 'outputs', 'report_figs')
os.makedirs(OUT_DIR, exist_ok=True)

CELL = 320


def big_comp(m):
    cs = m.split(only_watertight=False)
    return max(cs, key=lambda c: len(c.vertices)) if len(cs) > 1 else m


def render_single(mesh, renderer, azim=45, elev=25, distance=1.7):
    m = normalize_mesh(mesh.copy()); m.fix_normals()
    pm = pyrender.Mesh.from_trimesh(m, smooth=True)
    scene = pyrender.Scene(bg_color=[250, 250, 250, 0], ambient_light=[0.35, 0.35, 0.4])
    scene.add(pm)
    pose = spherical_camera_pose(azim, elev, distance)
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.5, aspectRatio=1.0)
    scene.add(cam, pose=pose)
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.0),
              pose=spherical_camera_pose(azim - 30, elev + 20, distance))
    scene.add(pyrender.DirectionalLight(color=[0.9, 0.9, 1.0], intensity=2.0),
              pose=spherical_camera_pose(azim + 60, elev + 10, distance))
    rgb, _ = renderer.render(scene)
    return Image.fromarray(rgb)


def label(img, text, h=28, bg='#1a4b84', size=14):
    w = img.size[0]
    full = Image.new('RGB', (w, img.size[1] + h), 'white')
    full.paste(img, (0, 0))
    draw = ImageDraw.Draw(full)
    draw.rectangle([0, img.size[1], w, img.size[1] + h], fill=bg)
    try: font = ImageFont.truetype('arial.ttf', size)
    except IOError: font = ImageFont.load_default()
    if len(text) > 40: text = text[:38] + '...'
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((w - tw) // 2, img.size[1] + 4), text, fill='white', font=font)
    return full


def title_strip(text, w, h=42, bg='#0a2a5c', size=20):
    s = Image.new('RGB', (w, h), bg)
    draw = ImageDraw.Draw(s)
    try: font = ImageFont.truetype('arial.ttf', size)
    except IOError: font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((w - tw) // 2, (h - bbox[3] + bbox[1]) // 2), text, fill='white', font=font)
    return s


def hstack(imgs, gap=6, bg='white'):
    w = sum(i.size[0] for i in imgs) + gap * (len(imgs) - 1)
    h = max(i.size[1] for i in imgs)
    out = Image.new('RGB', (w, h), bg)
    x = 0
    for i in imgs:
        out.paste(i, (x, 0)); x += i.size[0] + gap
    return out


def vstack(imgs, gap=6, bg='white'):
    w = max(i.size[0] for i in imgs)
    h = sum(i.size[1] for i in imgs) + gap * (len(imgs) - 1)
    out = Image.new('RGB', (w, h), bg)
    y = 0
    for i in imgs:
        out.paste(i, (0, y)); y += i.size[1] + gap
    return out


# ----------  Job A: hold-out pairs in 2-row 5-col layout  ----------
def fig_holdout_pairs(renderer):
    LIST = os.path.join(H, 'mesh_list.txt')
    paths_by_mid = {}
    with open(LIST) as f:
        for ln in f:
            rel = ln.strip().split('\t')[0]
            mid = os.path.splitext(os.path.basename(rel))[0]
            paths_by_mid[mid] = os.path.join(ROOT, rel)
    PICKS = [
        ('00008395_479bdb6fb2da47daafc80f85_trimesh_005', 'Plate (CD=0.021)'),
        ('00023720_bf6380c0d20f4ad099e5c48e_trimesh_000', 'Chamfered (CD=0.025)'),
        ('00026529_a306940ce32e4b0db2460dc2_trimesh_003', 'Bracket (CD=0.034)'),
        ('00005865_0abc50e0d63140d68efa0c5f_trimesh_003', 'Thin (CD=0.034)'),
        ('00026437_f9d63f278d034bf7b9324470_trimesh_016', 'Bracket var (CD=0.040)'),
    ]
    gt_imgs, pred_imgs, captions = [], [], []
    for mid, cap in PICKS:
        gt = trimesh.load(paths_by_mid[mid])
        pred = big_comp(trimesh.load(
            os.path.join(H, 'samples_cfg3.0', f'pred_{mid}.obj')))
        gt_imgs.append(render_single(gt, renderer))
        pred_imgs.append(render_single(pred, renderer))
        captions.append(cap)

    # Build header strip + 5 column labels + 2 rows
    col_labels = []
    for cap in captions:
        s = Image.new('RGB', (CELL, 28), '#bfbfbf')
        d = ImageDraw.Draw(s)
        try: f = ImageFont.truetype('arial.ttf', 13)
        except IOError: f = ImageFont.load_default()
        b = d.textbbox((0, 0), cap, font=f)
        d.text(((CELL - (b[2] - b[0])) // 2, 5), cap, fill='black', font=f)
        col_labels.append(s)
    col_labels_row = hstack(col_labels)

    gt_row = hstack(gt_imgs)
    pred_row = hstack(pred_imgs)
    # left side labels "Ground Truth" / "Image-Cond Prediction"
    gt_lbl = Image.new('RGB', (100, CELL), '#2a6cb0')
    d = ImageDraw.Draw(gt_lbl)
    try: f = ImageFont.truetype('arial.ttf', 14)
    except IOError: f = ImageFont.load_default()
    d.text((6, CELL // 2 - 24), 'Ground', fill='white', font=f)
    d.text((6, CELL // 2 - 6),  'Truth',  fill='white', font=f)
    pred_lbl = Image.new('RGB', (100, CELL), '#0c6043')
    d = ImageDraw.Draw(pred_lbl)
    d.text((6, CELL // 2 - 30), 'Image', fill='white', font=f)
    d.text((6, CELL // 2 - 12), 'Cond',  fill='white', font=f)
    d.text((6, CELL // 2 + 6),  'Predicted', fill='white', font=f)

    # blank left for column labels row
    blank = Image.new('RGB', (100, 28), 'white')
    col_row = hstack([blank, col_labels_row])
    gt_full   = hstack([gt_lbl, gt_row])
    pred_full = hstack([pred_lbl, pred_row])
    body = vstack([col_row, gt_full, pred_full])
    title = title_strip('Image-Driven Generation: 5 Hold-out Examples (GT vs Predicted)',
                        body.size[0])
    final = vstack([title, body])
    out = os.path.join(OUT_DIR, 'holdout_pairs_wide.png')
    final.save(out)
    print(f'wrote {out}  ({final.size[0]}x{final.size[1]})', flush=True)


# ----------  Job B: Phase 3 demos in 3x2 grid  ----------
def fig_phase3_demo(renderer):
    P3 = os.path.join(ROOT, 'outputs', 'phase3_demo')
    with open(os.path.join(P3, 'summary.json')) as f:
        summary = json.load(f)
    cells = []
    for i, s in enumerate(summary[:6]):
        m = big_comp(trimesh.load(os.path.join(P3, f'demo_{i:02d}.obj')))
        img = render_single(m, renderer)
        # caption shows user prompt (truncated)
        prompt = s['user_prompt']
        if len(prompt) > 38: prompt = prompt[:35] + '...'
        cell = label(img, prompt, h=32, bg='#0c6043', size=12)
        cells.append(cell)
    # 3 cols, 2 rows
    row1 = hstack(cells[:3])
    row2 = hstack(cells[3:6])
    body = vstack([row1, row2])
    title = title_strip(
        'Text-Driven Generation via Local Agent: 6 User Prompts',
        body.size[0])
    final = vstack([title, body])
    out = os.path.join(OUT_DIR, 'phase3_demo_wide.png')
    final.save(out)
    print(f'wrote {out}  ({final.size[0]}x{final.size[1]})', flush=True)


# ----------  Job C: Qwen vs Claude 3x2 grid of pairs  ----------
def fig_qwen_vs_claude(renderer):
    Q = os.path.join(ROOT, 'outputs', 'phase3_demo')
    C = os.path.join(ROOT, 'outputs', 'phase3_claude')
    with open(os.path.join(Q, 'summary.json')) as f: qs = json.load(f)
    with open(os.path.join(C, 'summary.json')) as f: cs = json.load(f)
    cells = []
    SMALL = 240
    for i in range(6):
        qm = big_comp(trimesh.load(os.path.join(Q, f'demo_{i:02d}.obj')))
        cm = big_comp(trimesh.load(os.path.join(C, f'demo_{i:02d}.obj')))
        q_img = render_single(qm, renderer); q_img.thumbnail((SMALL, SMALL))
        c_img = render_single(cm, renderer); c_img.thumbnail((SMALL, SMALL))
        q_lab = label(q_img, 'Qwen', h=24, bg='#0c6043', size=12)
        c_lab = label(c_img, 'Claude', h=24, bg='#2a6cb0', size=12)
        pair = hstack([q_lab, c_lab])
        prompt = qs[i]['user_prompt']
        if len(prompt) > 50: prompt = prompt[:48] + '...'
        cap_strip = Image.new('RGB', (pair.size[0], 26), '#8a4a00')
        d = ImageDraw.Draw(cap_strip)
        try: f = ImageFont.truetype('arial.ttf', 12)
        except IOError: f = ImageFont.load_default()
        d.text((8, 5), f'#{i+1}: "{prompt}"', fill='white', font=f)
        cells.append(vstack([cap_strip, pair]))
    row1 = hstack(cells[:3])
    row2 = hstack(cells[3:6])
    body = vstack([row1, row2])
    title = title_strip('Agent Ablation: Qwen2-VL (left) vs Claude Sonnet 4.6 (right)',
                        body.size[0])
    final = vstack([title, body])
    out = os.path.join(OUT_DIR, 'qwen_vs_claude_wide.png')
    final.save(out)
    print(f'wrote {out}  ({final.size[0]}x{final.size[1]})', flush=True)


# ----------  Job D: Unconditional baseline samples (NEW)  ----------
def fig_unconditional(renderer):
    """v7.4b unconditional baseline — 6 representative samples."""
    BASE = os.path.join(ROOT, 'outputs', 'samples_v7_4b')
    # The samples_v7_4b/ has 16 .obj files. Pick 6 that show variety.
    PICKS = [
        (4,  '#1: Plate variant'),
        (6,  '#2: Block w/ holes'),
        (10, '#3: Bracket'),
        (11, '#4: Cylindrical'),
        (12, '#5: Plate'),
        (15, '#6: Bracket variant'),
    ]
    cells = []
    for idx, cap in PICKS:
        p = os.path.join(BASE, f'sample_{idx:02d}.obj')
        m = big_comp(trimesh.load(p))
        img = render_single(m, renderer)
        cells.append(label(img, cap, h=28, bg='#5a3a1a', size=13))
    row1 = hstack(cells[:3])
    row2 = hstack(cells[3:6])
    body = vstack([row1, row2])
    title = title_strip(
        'Unconditional Baseline: 6 Random Samples from the Stable Flow-Matching Model',
        body.size[0])
    final = vstack([title, body])
    out = os.path.join(OUT_DIR, 'unconditional_grid.png')
    final.save(out)
    print(f'wrote {out}  ({final.size[0]}x{final.size[1]})', flush=True)


def main():
    print('loading PyRender ...', flush=True)
    r = pyrender.OffscreenRenderer(CELL, CELL)
    print('Job A: hold-out wide ...', flush=True);     fig_holdout_pairs(r)
    print('Job B: Phase 3 wide ...', flush=True);      fig_phase3_demo(r)
    print('Job C: Qwen vs Claude wide ...', flush=True); fig_qwen_vs_claude(r)
    print('Job D: Unconditional grid ...', flush=True);  fig_unconditional(r)
    r.delete()
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
