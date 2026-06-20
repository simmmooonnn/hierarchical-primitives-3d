"""Encode captions via CLIP-L text encoder → bundle.pt for v10 training.

Input:  outputs/captions/captions.json  {mesh_id: caption}
Output: outputs/clip_caption_embs/bundle.pt
         {'embs': (N, 1, 77, 768) fp16, 'mesh_ids': [...]}

The (1, 77, 768) shape matches the existing conditional model's expected
(n_views, n_tokens, cond_dim) layout (with n_views=1 for text-only).
"""
import os, json, time
import torch
from transformers import CLIPTextModel, CLIPTokenizer

ROOT = os.environ.get('DIFFATS_ROOT', os.getcwd())
CAPTIONS_JSON = os.path.join(ROOT, 'outputs', 'captions', 'captions.json')
OUT_FILE      = os.path.join(ROOT, 'outputs', 'clip_caption_embs', 'bundle.pt')


def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    with open(CAPTIONS_JSON) as f:
        captions = json.load(f)
    mesh_ids = sorted(captions.keys())
    texts = [captions[m] for m in mesh_ids]
    print(f'loaded {len(texts)} captions', flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('loading openai/clip-vit-large-patch14 text encoder ...', flush=True)
    tok = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    model = CLIPTextModel.from_pretrained(
        'openai/clip-vit-large-patch14',
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    D = model.config.hidden_size           # 768
    T = tok.model_max_length               # 77
    print(f'  dim={D}  max_tokens={T}', flush=True)

    N = len(texts)
    embs = torch.empty((N, 1, T, D), dtype=torch.float16)

    t0 = time.time()
    BATCH = 32
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        batch_texts = texts[start:end]
        inputs = tok(batch_texts, padding='max_length', truncation=True,
                     max_length=T, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inputs).last_hidden_state    # (B, T, D)
        embs[start:end, 0] = out.detach().to(torch.float16).cpu()
        if (end // BATCH) % 10 == 0:
            rate = end / (time.time() - t0)
            remain = (N - end) / max(rate, 1e-9)
            print(f'  {end}/{N}  {rate:.0f} cap/s  ETA {remain:.0f}s', flush=True)

    print(f'\nembs shape={tuple(embs.shape)}  dtype={embs.dtype}  '
          f'size={embs.element_size() * embs.numel() / 1e9:.2f} GB', flush=True)
    torch.save({'embs': embs, 'mesh_ids': mesh_ids}, OUT_FILE)
    print(f'wrote {OUT_FILE}', flush=True)


if __name__ == '__main__':
    main()
