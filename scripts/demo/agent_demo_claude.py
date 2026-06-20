"""Phase 3 demo with Claude as the agent (vs Qwen2-VL in phase3_agent_demo.py).

Reads API key from ANTHROPIC_API_KEY env var — never logs it.
Saves outputs/phase3_claude/ alongside outputs/phase3_demo/ for direct comparison.
"""
import os, sys, time, json
import numpy as np
import torch
import anthropic

from diffats.models import SDFDiTConfig, SDFDiT, SDFDiTWrapper
from diffats.sampling import si_sample, unnormalize, reconstruct_sdf
from diffats.utils import extract_mesh
from transformers import CLIPTextModel, CLIPTokenizer

ROOT = os.environ.get('DIFFATS_ROOT', os.getcwd())
OUT_DIR = os.path.join(ROOT, 'outputs', 'phase3_claude')
os.makedirs(OUT_DIR, exist_ok=True)

USER_PROMPTS = [
    "a flat plate with two corner holes",
    "a cylindrical housing with a central hole and a flange at the base",
    "a bracket with three mounting slots and rounded corners",
    "a thick rectangular plate with a chamfered top edge",
    "a hollow tube with a fillet at one end",
    "a triangular plate with a single central hole",
]

AGENT_SYSTEM = (
    "You are a 3D CAD geometry describer. The user gives you a casual description of a "
    "mechanical part. Rewrite it as a single 60-120-word geometric caption that matches "
    "the training distribution of a text-to-3D model.\n\n"
    "STRICT RULES:\n"
    "1. Describe ONLY the physical object. Never mention images, views, renderings, normals, "
    "colors, shading, brightness, or anything green/red/blue/yellow.\n"
    "2. Use DECLARATIVE language. Forbidden: 'appears', 'seems', 'likely', 'possibly', "
    "'might', 'could', 'suggests'. State features as facts.\n"
    "3. State concrete features: shape category (bracket / plate / cylinder / housing / "
    "shaft / gear / flange / clamp), number of holes / slots / ribs / fillets / chamfers, "
    "axis of symmetry, rough proportions.\n"
    "4. Start with the shape category, e.g. 'A flat plate with...', 'A cylindrical housing "
    "featuring...'.\n\n"
    "Reply with ONLY the rewritten caption, no preamble or explanation."
)


def refine_with_claude(client, user_text, model='claude-sonnet-4-6'):
    msg = client.messages.create(
        model=model,
        max_tokens=300,
        system=AGENT_SYSTEM,
        messages=[{'role': 'user', 'content': f'User request: "{user_text}"\n\nRewrite as the geometric caption:'}],
    )
    return msg.content[0].text.strip()


def load_v10(device):
    ck_path = os.path.join(ROOT, 'outputs', 'train_v10', 'ckpt_step200000.pt')
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    cfg = SDFDiTConfig(**ck['cfg'])
    model = SDFDiTWrapper(SDFDiT(cfg))
    sd = ck.get('ema_state_dict', ck['state_dict'])
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, ck['stats'], cfg


def encode_text(tok, clip_text, captions, device):
    inputs = tok(captions, padding='max_length', truncation=True,
                 max_length=77, return_tensors='pt').to(device)
    with torch.no_grad():
        out = clip_text(**inputs).last_hidden_state
    return out.float()


@torch.no_grad()
def sample_v10(model, cfg, stats, cond, cfg_scale, device, n_steps=50):
    F = model.core.flat_total
    x = si_sample(model, F, T_scale=1000, n_steps=n_steps, device=device,
                  n_samples=cond.shape[0], schedule='trig', solver='heun',
                  cond=cond, cfg_scale=cfg_scale, log_every=999)
    G, U2, U3 = unnormalize(x, cfg.N, cfg.R, stats)
    sdf = reconstruct_sdf(G, U2, U3).cpu().numpy()
    return sdf


def main():
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print('ERROR: ANTHROPIC_API_KEY env var not set', flush=True)
        sys.exit(2)
    # Never print the key.
    print(f'Claude SDK ready (key configured: {bool(api_key)})', flush=True)
    client = anthropic.Anthropic()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\n[1/2] loading CLIP-L text encoder ...', flush=True)
    clip_tok = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    clip_text = CLIPTextModel.from_pretrained(
        'openai/clip-vit-large-patch14', torch_dtype=torch.bfloat16).to(device).eval()

    print('\n[2/2] loading v10 model ...', flush=True)
    v10, stats, v10_cfg = load_v10(device)
    print(f'  v10 params: {sum(p.numel() for p in v10.parameters()):,}', flush=True)

    summary = []
    voxel = 2.0 / 255

    for i, user_text in enumerate(USER_PROMPTS):
        print(f'\n========== prompt {i+1}/{len(USER_PROMPTS)} ==========')
        print(f'USER     : {user_text}', flush=True)

        t0 = time.time()
        refined = refine_with_claude(client, user_text)
        t_agent = time.time() - t0
        print(f'REFINED  : {refined}', flush=True)
        print(f'  [Claude: {t_agent:.1f}s]', flush=True)

        emb = encode_text(clip_tok, clip_text, [refined], device)
        emb = emb.unsqueeze(1)

        t0 = time.time()
        sdf = sample_v10(v10, v10_cfg, stats, emb, cfg_scale=3.0, device=device)
        t_sample = time.time() - t0
        in_frac = float((sdf[0] < 0).mean())
        print(f'  SDF range=[{sdf[0].min():.4f}, {sdf[0].max():.4f}]  inside_frac={in_frac:.4%}', flush=True)
        print(f'  [sample: {t_sample:.1f}s]', flush=True)

        if 1e-4 < in_frac < 0.999:
            mesh = extract_mesh(sdf[0], voxel, bbox_min=-1.0)
            if mesh is not None:
                out_obj = os.path.join(OUT_DIR, f'demo_{i:02d}.obj')
                mesh.export(out_obj)
                nV, nF = len(mesh.vertices), len(mesh.faces)
                print(f'  -> {out_obj}  V={nV}  F={nF}', flush=True)
                summary.append({
                    'user_prompt':   user_text,
                    'refined':       refined,
                    'inside_frac':   in_frac,
                    'V': nV, 'F': nF,
                    'agent_time_s':  round(t_agent, 2),
                    'sample_time_s': round(t_sample, 2),
                    'out_obj':       out_obj,
                })

    with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\nwrote {OUT_DIR}/summary.json', flush=True)
    print('\n=== TIMING (per prompt avg) ===')
    if summary:
        print(f'  Claude: {np.mean([s["agent_time_s"]  for s in summary]):.1f}s')
        print(f'  sample: {np.mean([s["sample_time_s"] for s in summary]):.1f}s')


if __name__ == '__main__':
    main()
