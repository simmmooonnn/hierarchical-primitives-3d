"""Pick 16 ABC meshes NOT seen during training, with reasonable polygon counts.

Reads:
  data/mesh_id_to_path.json — training mesh paths (to EXCLUDE)
  abc_obj/obj/                — all available 22617 mesh paths

Filters: 3k <= n_faces <= 100k (typical CAD size), well-formed
Output: outputs/holdout/mesh_list.txt + per-id symlinks
"""
import os, json, glob, random
import trimesh

ROOT = os.environ.get('DIFFATS_ROOT', os.getcwd())
TRAIN_MAP = os.path.join(ROOT, 'data', 'mesh_id_to_path.json')
ABC_DIR   = os.path.join(ROOT, 'abc_obj', 'obj')
OUT_DIR   = os.path.join(ROOT, 'outputs', 'holdout')
os.makedirs(OUT_DIR, exist_ok=True)

with open(TRAIN_MAP) as f:
    train_paths = set(json.load(f).values())
print(f'training paths: {len(train_paths)}', flush=True)

all_paths = sorted(glob.glob(os.path.join(ABC_DIR, '*/*.obj')))

all_rel = [os.path.relpath(p, ROOT) for p in all_paths]
unseen = [(p, rp) for p, rp in zip(all_paths, all_rel) if rp not in train_paths]
print(f'unseen candidates: {len(unseen)}', flush=True)

random.seed(123)
random.shuffle(unseen)

# Filter for reasonable mesh size
picks = []
i = 0
while len(picks) < 16 and i < len(unseen):
    full_p, rel_p = unseen[i]
    i += 1
    try:
        m = trimesh.load(full_p, force='mesh', process=False)
        if not isinstance(m, trimesh.Trimesh):
            continue
        nf = len(m.faces); nv = len(m.vertices)
        if 3000 <= nf <= 100000 and nv >= 1500:
            picks.append((rel_p, nv, nf))
            print(f'  picked {len(picks):>2}: {rel_p}  V={nv} F={nf}', flush=True)
    except Exception:
        continue
print(f'\nPicked {len(picks)} holdout meshes', flush=True)

out_list = os.path.join(OUT_DIR, 'mesh_list.txt')
with open(out_list, 'w') as f:
    for rel_p, nv, nf in picks:
        f.write(f'{rel_p}\t{nv}\t{nf}\n')
print(f'wrote {out_list}', flush=True)
