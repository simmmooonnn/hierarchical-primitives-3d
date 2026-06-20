# Repository Notes

Internal notes on how this repo was assembled, what was cleaned, and what still needs your
attention before publishing. **Not part of the public-facing docs** — delete or trim before release.

## Provenance

Code was pulled from the Anvil GPU project `/anvil/scratch/x-zsu7/diffats_gpu/` (the live experiment
tree) on 2026-06-19 and reorganized into a standard installable-package layout:

- **`diffats/` package** — the importable library: `models.py` (DiT), `sampling.py` (samplers),
  `render.py` (shared rendering), `utils.py` (TSDF/Tucker/metrics). Installed via `pip install -e .`
  (`pyproject.toml`). Run directly with e.g. `python -m diffats.sampling`.
- **`scripts/`** — runnable entry points that `import diffats` (train, eval, data prep, conditioning,
  agent demos). Cross-script `sys.path` hacks were removed; everything imports from the package.
- **`scripts/slurm/`** — the original SLURM submitters, paths normalized to `scripts/` / `diffats`.
- **Filenames normalized** — version suffixes dropped; this is the **latest** working code (v7/v9/v10),
  not drafts. Module map: `model_sdf_dit_v4.py`→`diffats/models.py`, `sample_sdf_dit_v4.py`→
  `diffats/sampling.py`, `render_views.py`→`diffats/render.py`, `train_sdf_dit_v4.py`→`scripts/train.py`,
  `eval_sota_metrics_v2.py`→`scripts/eval/eval.py`, `save_tucker_sdf.py`→`scripts/data/preprocess_tucker.py`,
  `phase3_claude_demo.py`→`scripts/demo/agent_demo_claude.py`, `phase3_agent_demo.py`→`scripts/demo/agent_demo_qwen.py`.
  Classes `SDFDiTV4/ConfigV4/WrapperV4` → `SDFDiT/SDFDiTConfig/SDFDiTWrapper`.
- **Removed (outdated/dev-only):** `eval_sota_metrics.py`, `render_report_visuals.py` (superseded by
  `_v2`), plus dev scripts `diag_v4.py`, `caption_smoke.py`, `smoke_test_cond_dit.py`, and their orphaned
  SLURM submitters. All recoverable from the first commit.

Verified after reorg: `pip install -e .` succeeds; `import diffats`, `from diffats.models import SDFDiT`,
`from diffats.sampling import si_sample` all resolve; `python -m diffats.models` self-test passes
(59,524,864 params, seq_len 624, IO check OK); all `diffats/` + `scripts/` files byte-compile.

## What was cleaned

1. **Hardcoded cluster paths removed (17 occurrences, 11 files).**
   - `ROOT = '/anvil/scratch/x-zsu7/diffats_gpu'` → `ROOT = os.environ.get('DIFFATS_ROOT', os.getcwd())`
     (run scripts from the repo root, or point `DIFFATS_ROOT` at your data).
   - `sys.path.insert(0, '/anvil/.../scripts')` hacks removed entirely — imports now resolve through the
     installed `diffats` package.

   No `/anvil/scratch`, `x-zsu7`, or server IPs remain in `diffats/` or `scripts/`. (The `scripts/slurm/`
   scripts still contain `$SCRATCH/...` cluster paths by design — they are cluster-specific reference.)

2. **API-key print removed.** `phase3_claude_demo.py` previously printed the first 5 chars of the key;
   now it only prints whether a key is configured. The key is still read from `ANTHROPIC_API_KEY` only.

## Security

- **No secrets are committed.** `.gitignore` blocks `.env`, `*.key`, `*_ed25519`, `*.pem`.
- The Anthropic API key is read from the `ANTHROPIC_API_KEY` environment variable at runtime and is
  never written to any file. **Rotate the key** you used for the live demo if you have not already.
- Before pushing public, double-check git history is clean: this repo is a **fresh `git init`** with no
  prior history, so no old secrets/checkpoints are carried in.

## TODO before publishing

- [ ] Fill in author/name/username placeholders in `README.md`, `LICENSE`, `CITATION.cff`.
- [ ] Upload pretrained weights (HF Hub or GitHub Releases) and link in README → *Weights & Data*.
- [ ] Provide an ABC download/prepare helper, or document the exact subset
      (`meshes_v2_filtered_10k.json` is gitignored — host it or regenerate).
- [ ] Verify `train.py` / `sample.py` argument names match the README examples
      (README commands were taken from the `slurm/` scripts; spot-check `--betas` parsing etc.).
- [ ] Run one clean-environment smoke test (fresh venv → install → `python -m diffats.models`),
      ideally on a machine that is *not* the dev box, to catch missing deps.
- [ ] Consider trimming `slurm/` to a few representative scripts (49 files is a lot of ablation noise).

## Known doc discrepancy

`docs/Tokenization_Report.docx` was written from the **older Trainium** model (`model_sdf_dit_v2.py`,
which diffused on `(U₁, U₃, G)` with **U₂** folded into the core). The **current** model here
(`model.py`) is the paper-faithful *l=2* scheme: it diffuses on `(G, U₂, U₃)` with
`G = 𝒞 ×₁ U₁` (**U₁** folded into the core, U₁ then dropped), seq_len = R²+2R = 624.
Regenerate that report against `model.py` before using it in the paper.
