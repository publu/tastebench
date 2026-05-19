# LLM.md

Guidance for LLM coding agents working in this repo.

Treat this as shared project guidance, not tool-specific ownership. Keep
assistant-specific state, scheduled tasks, credentials, memories, lockfiles,
and other local runtime artifacts out of tracked repo files unless the user
explicitly asks for them.

## What this is

`tastebench` learns the *taste signature* of reference media you admire,
then grades your own draft against it and lists ranked edits. Two layers
run per file:

- **Craft** — model-free, sub-second. Audio uses librosa; still images and
  video use the PIL-derived visual path. The default and graceful fallback.
  Never needs a download.
- **Brain** — Meta's TRIBE fMRI-encoder (~20 GB, gated Llama-3.2). Optional;
  degrades cleanly to craft when absent.

It measures *distance to a taste the user defined* — never predicts hits.

## Run / test

```bash
make            # venv (core deps) + launch the worker on ./workspace
make test       # model-free smoke suite (synthesizes its own audio)
python -m tastebench [worker|compare|…]   # no install needed
.venv/bin/tastebench --help               # all CLI verbs
```

Tests are model-free and must stay that way — they synthesize audio and
assert the engine degrades clearly with no model present. Keep `make test`
green; add a smoke test for new surfaces.

## Primary UX: the worker (not the CLI)

The product's main surface is **`tastebench/worker.py`**, launched by bare
`tastebench` / `python -m tastebench`. It watches
`<root>/references/<name>/{refs,draft}/` — each `<name>` is one experiment;
`refs/` defines a taste, `draft/` is graded against it, and a
`<draft>.report.md` is written next to the draft. Poll-based and
settle-aware (no partial reads). The CLI verbs (`compare`/`optimize`/…) and
`tastebench drop` are kept for scripting/legacy — do not re-make the CLI
the default.

## Architecture (one job per module)

- `engine.py` — pure TRIBE inference (`predict`); raises a clear error, no
  network/credentials/server. `signature.py` — one file → craft+brain sig.
- `profile.py` — refs → centroid+spread. `compare.py` — demo vs taste,
  spread-normalized deltas. `optimize.py` — ranked, confidence-labeled edits.
- `report.py` — markdown / json / verdict / LLM bundle (producer voice via
  `identity.py`). `worker.py` / `flow.py` / `tui.py` / `cli.py` — surfaces.
- `features/` — craft extractors. `explainers/explainers.json` — the
  first-class glossary; every compare/optimize line attaches an entry.
- `native.py` / `fast_text.py` — the speed layer over vanilla `tribev2`
  (same pipeline + numerics, ~15–40× on the audio path: bf16 Llama,
  one forward per unique sentence, sdpa, MPS, spawn-safe DataLoader,
  cached in-process ASR). All opt-out via env; CUDA path unchanged. Don't
  alter the byte-identical slice math in `fast_text._get_data_fast`.
  `timing.py` prints per-stage wall time (`TRIBE_TIMING`).

## Conventions

- Python 3.11–3.12. Modules are docstring-heavy and single-purpose; match
  that voice. Lazy imports in `cli.py`/surfaces keep startup fast and the
  package importable with no model.
- Keep repo instructions agent-neutral. Do not introduce tool-specific
  automation, background tasks, or lockfiles into tracked files without an
  explicit user request.
- Never commit media, weights, secrets, or large caches (see `.gitignore`).
  No model weights are vendored — they're declared deps the user installs.
- License: wrapper is MIT but the tool *as a whole* is non-commercial
  (CC-BY-NC TRIBE + Llama community license). Don't add anything implying
  commercial grant. See `LICENSE` / `NOTICE` / `ATTRIBUTION.md`.
