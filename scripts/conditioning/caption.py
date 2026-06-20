"""Batch-caption all 10k training meshes + 16 hold-out meshes via Qwen2-VL-7B.

Output: outputs/captions/captions.json  {mesh_id: caption, ...}
Resumable: skips meshes whose mesh_id is already in the output JSON.
"""
import os, sys, json, time, glob
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

ROOT = os.environ.get('DIFFATS_ROOT', os.getcwd())
OUT_FILE = os.path.join(ROOT, 'outputs', 'captions', 'captions.json')

# Same prompt as the validated smoke run
PROMPT = (
    "These 8 inputs show one 3D mechanical part from different angles. Write a single "
    "60-120-word paragraph describing the part's geometry, to be used as a text "
    "condition for training a generative 3D model.\n\n"
    "STRICT RULES (violating any of these makes the caption unusable):\n"
    "1. Describe ONLY the physical object — never mention images, views, renderings, "
    "normals, shading, colors, brightness, highlights, or anything green/red/blue/"
    "yellow/etc.\n"
    "2. Use DECLARATIVE language. Forbidden words: 'appears', 'seems', 'likely', "
    "'possibly', 'might', 'could', 'suggests'. State features as facts.\n"
    "3. State concrete features: shape category (bracket / plate / cylinder / "
    "housing / shaft / gear / flange / clamp / etc.), number of holes / slots / "
    "ribs / fillets / chamfers, axis of symmetry if any, rough proportions "
    "(long-thin, short-stubby, equal-sided, ...).\n"
    "4. Start the paragraph with the shape category, e.g. 'A flat plate with...', "
    "'A cylindrical housing featuring...'.\n\n"
    "Write the caption now:"
)


def caption_one(model, proc, render_dir, n_views=8):
    imgs = [Image.open(os.path.join(render_dir, f'view{v:02d}_normal.png')).convert('RGB')
            for v in range(n_views)]
    content = [{'type': 'image', 'image': im} for im in imgs] + \
              [{'type': 'text',  'text': PROMPT}]
    messages = [{'role': 'user', 'content': content}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=imgs, padding=True,
                  return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    gen = out[:, inputs.input_ids.shape[1]:]
    return proc.batch_decode(gen, skip_special_tokens=True)[0].strip()


def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    # Load existing captions for resumability
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE) as f:
            captions = json.load(f)
        print(f'[resume] {len(captions)} captions already in {OUT_FILE}', flush=True)
    else:
        captions = {}

    # Build work list: training meshes + hold-out meshes
    work = []
    train_root = os.path.join(ROOT, 'outputs', 'renders')
    for d in sorted(glob.glob(os.path.join(train_root, '*'))):
        if os.path.isdir(d):
            mid = os.path.basename(d)
            if mid not in captions:
                work.append((mid, d))
    holdout_root = os.path.join(ROOT, 'outputs', 'holdout', 'renders')
    for d in sorted(glob.glob(os.path.join(holdout_root, '*'))):
        if os.path.isdir(d):
            mid = os.path.basename(d)
            if mid not in captions:
                work.append((mid, d))
    print(f'to caption: {len(work)} meshes', flush=True)
    if not work:
        print('nothing to do.')
        return

    print('loading Qwen2-VL-7B-Instruct ...', flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        'Qwen/Qwen2-VL-7B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    proc = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct')

    t_start = time.time()
    errors = []
    SAVE_EVERY = 100
    for i, (mid, d) in enumerate(work):
        try:
            cap = caption_one(model, proc, d)
            captions[mid] = cap
        except Exception as e:
            errors.append((mid, str(e)))
            print(f'  ERR {mid}: {e}', flush=True)
            continue

        if (i + 1) % SAVE_EVERY == 0:
            with open(OUT_FILE, 'w') as f:
                json.dump(captions, f, indent=1, ensure_ascii=False)
            el = time.time() - t_start
            rate = (i + 1) / el
            remain = (len(work) - i - 1) / max(rate, 1e-9)
            print(f'  {i+1}/{len(work)} ({rate:.2f} mesh/s) '
                  f'ETA {remain/60:.1f} min  errors={len(errors)}', flush=True)

    with open(OUT_FILE, 'w') as f:
        json.dump(captions, f, indent=1, ensure_ascii=False)
    print(f'\nDONE: {len(captions)} captions, {len(errors)} errors  '
          f'(elapsed {(time.time()-t_start)/60:.1f} min)', flush=True)


if __name__ == '__main__':
    main()
