```
╭───────────────────────────────────────╮
│    █████   ███    ████  █████  █████  │
│      █    █   █  █        █    █      │
│      █    █████   ███     █    ███    │
│      █    █   █      █    █    █      │
│      █    █   █  ████     █    █████  │
│                                       │
│    ████   █████  █   █   ████  █   █  │
│    █   █  █      ██  █  █      █   █  │
│    ████   ███    █ █ █  █      █████  │
│    █   █  █      █  ██  █      █   █  │
│    ████   █████  █   █   ████  █   █  │
╰───────────────────────────────────────╯
```

# Your own focus group — for a track, a cut, or a web page

You have a draft and a few things you wish it felt like. Drop them in a
folder together. It learns what those things have in common and tells
you, the second your draft lands, where yours is off and what to move
first — then keeps doing it on everything you drop after.

No "is this good in the abstract." No "will it go viral." Just: how far
is this from the taste I picked, and which lever closes the gap.

```
── synthwave · demo.wav ──────────────────

  DIFFERENT RECORD — 11% taste match

  TASTE MATCH  [███·····················]  11%

  ▸ the one thing: your hook shows up at 11.3s — the refs land theirs by
    ~4s; most listeners are gone before your best moment.
  · closest to ref_b.wav of your set
```

That's the craft read. Underneath it, the same drop gets a neural read —
the predicted brain response, the part no waveform shows you. Here it is
on a real released track, computed locally, no GPU:

```
 level-up-v4.mp3 · 2:38 · full neural read in ~4m40s on an M1, no GPU
 predicted 12-network response — z-scored across its 12 networks
 ───────────────────────────────────────────────────────────────────
  Visual2            scene / motion     ████████████████████████  +1.9 ✦
  Cingulo-Opercular  attention/effort   ████████████████████      +1.5 ✦
  Visual1            low-level visual    █████████████████         +1.1 ✦
  Somatomotor        embodied / motion   ████████████████          +1.1 ✦
  8 other networks   no differential     ▏                         −0.7
 ───────────────────────────────────────────────────────────────────
 ✦ stable across the whole track (reliability ~0.98)
 a predicted response pattern, not a verdict · full read → demo/
```

This track reads as motion / attention / body, stable end to end, with
the rest flat — that *shape* is the taste. Your references define the
shape to hit; your draft is scored by how far its shape sits from theirs.

## Run one thing

```bash
git clone https://github.com/publu/tastebench && cd tastebench
make     # model-free venv in seconds (nothing gated), then it starts watching a folder
```

That's the whole setup. `make` needs only `python3`. It starts a worker
on a folder tree — **one folder per taste:**

```
tastebench/references/
  my-sound/
    refs/    ← a few tracks / videos / images / live URLs you ADMIRE
    draft/   ← drop your draft here → graded the instant it lands
```

Drop work you admire into `refs/` — it learns the taste they share. Drop
a draft into `draft/` — it's graded against that taste, live in the
terminal, with a full `<draft>.report.md` written beside it. Change
anything and **it re-grades on its own**: settle-aware (a half-copied or
multi-file drag never triggers a partial run), profile-cached (the heavy
model never re-runs on unchanged inputs). As many `references/<name>/`
folders as you want, all independent.

You run one command. It loops from there.

## What you can benchmark

- **Music** — *"hit like these three records — where am I off?"* → hook
  lands 11s in (theirs at 4s), doesn't loop clean, dynamics flat —
  *fix the hook timing first.*
- **Video** — match the cut pacing, motion energy, palette and contrast
  of an edit you admire.
- **Websites** — a page *is* a video here. It drives a real browser,
  autoscrolls, and grades the recording's visual signature against pages
  you like. Drop a `.url` into `refs/`/`draft/`, or
  `tastebench web https://your.site --like good1.mp4`.
- **A/B a decision** — two mixes / cuts / versions, same references:
  which is actually closer, by how much, on which signals.
- **Brief an LLM** — `--llm` emits a self-contained bundle (raw numbers +
  full glossary) you paste into any model for a grounded, no-hallucination
  read.

Mix modalities freely — one model spans all three, so a track and a
video land in the *same* space.

> **It does not predict hits.** Those are irreducibly noisy (Salganik et
> al., *Science* 2006). It measures *near vs. far from the taste you
> gave it* — nothing else. That honesty is the point.

See the full neural + craft read on a **real released track**, computed
locally on an M1 in ~4m40s with no GPU: [`demo/`](demo/).

> **Licensing.** This wrapper is MIT, but it runs on Meta's **TRIBE v2**
> (CC-BY-NC-4.0) and **Llama 3.2**. The tool *as a whole* is
> **non-commercial — research / personal creative use only**. See
> [LICENSE](LICENSE), [NOTICE](NOTICE), [ATTRIBUTION.md](ATTRIBUTION.md).

## How it works

Every file — audio, video, image, or a recorded page — runs through two
layers:

| Layer | What it measures | Needs the model? | Modalities |
|---|---|---|---|
| **Brain** | TRIBE (Meta's fMRI-encoder) → a 12-network *Cole-Anticevic* signature: how strongly the work drives auditory, reward, default-mode, frontoparietal… | Yes (~20 GB) | audio · video · image — one model |
| **Craft** | concrete, fixable features — librosa for audio (hook timing, loopability, chorus lift, tempo/key stability, dynamics); a PIL path for video/image (palette, contrast, composition, cut pacing, motion) | No — sub-second | audio · video · image |

References define a target signature; your draft is scored by how far it
sits from it, network by network and feature by feature. The craft layer
**needs no download and answers instantly**; the brain layer **turns on
by itself** once its weights are present (until then you get the full
craft read). The brain layer is a *hypothesis view* — a predicted neural
response, not a validated outcome — and the tool says so wherever it
appears.

## The CLI — every step is also a plain command

```
tastebench profile  ref1 ref2 ref3               # what you like
tastebench compare  ref1 ref2 --to demo          # how you diverge
tastebench optimize demo --toward ref1 ref2      # ranked edits
tastebench web      https://site --like good.mp4 # grade a live URL
tastebench glossary [TERM]                        # the explainer dictionary
tastebench tui      ref1 ref2 --demo demo         # the visual view
```

`--llm` → bundle for any model · `--format json` → machine output ·
`-o FILE` → write to disk · `--no-brain` → craft only.

The explainer dictionary (`tastebench/explainers/explainers.json`) is a
first-class deliverable: one entry per craft feature, brain network, ROI
group and edit type — plain sentence, what it measures, how it's computed,
units, how to act. Every `compare`/`optimize` line carries its entry;
`report --llm` embeds the whole dictionary. Browse it with `tastebench
glossary`.

## The brain layer (optional, heavier)

```bash
make brain                                  # core + torch + Meta's tribev2 stack
huggingface-cli login                       # accept Meta's Llama 3.2 license (gated)
.venv/bin/python scripts/download_models.py # ~20 GB → ~/.cache/tastebench
```

~20 GB cache (fMRI encoder + Llama-3.2-3B + Whisper + wav2vec2). Runs on
**Apple Silicon (MPS) or CUDA**, auto-detected — minutes per clip on a
Mac, fast on a GPU. The brain stack wants **Python 3.11–3.12** (`make`
auto-picks one; the model-free core has no such limit). Nothing here is
required: without the cache the package still imports and falls back to
the instant craft read.

**No GPU?** Run the brain layer on your own [Modal](https://modal.com):

```bash
pip install -e ".[modal]"
modal setup                                       # your account
modal secret create huggingface HF_TOKEN=hf_xxx   # gated Llama-3.2
modal run tastebench/modal_app.py::download       # warm the ~20 GB Volume
modal run tastebench/modal_app.py --demo demo.wav --refs ref_a.wav,ref_b.wav
```

Self-serve: your account, your cache Volume, your bill. Same engine as
local, full upstream fidelity on a CUDA box. `TASTEBENCH_MODAL_GPU=A100`
if a big video OOMs the default A10G.

### Video on Apple Silicon auto-scales to your RAM

Upstream runs video at 64 frames/clip full-res — an unbounded set that
kernel-panics a 32 GB Mac on the first clip. So on Apple Silicon the
engine **auto-caps the video extractor by total RAM** (16 GB→4 frames,
32→8, 48→16, 64→24, 96→48); a 12 s / 720p clip then runs ~30 s on a
32 GB M1 instead of never finishing. It's a speed/fidelity trade —
fewer frames means coarser motion, so Mac video predictions aren't
numerically identical to a GPU run (the audio/text speed layer *is*
byte-identical). CUDA, ≥128 GiB, or `TRIBE_VIDEO_AUTO=0` → untouched,
full fidelity.

| env | default | effect |
|---|---|---|
| `TRIBE_VIDEO_AUTO` | `1` | `0` → upstream defaults (will OOM small Macs) |
| `TRIBE_VIDEO_FRAMES` | auto | force `num_frames` |
| `TRIBE_VIDEO_IMSIZE` | auto | force `max_imsize` (`0` = no cap) |

## Why it's faster than vanilla TRIBE v2

Upstream `tribev2` assumes a single CUDA box; as-is on a Mac it's slow or
fails outright. `native.py` + `fast_text.py` run the same pipeline with
the same numerics — every change opt-out via env vars, so the CUDA path
is unchanged:

- **Llama embeddings** — ~15–40× on the audio path: bf16 not fp32 (2–3×),
  one pass per *unique sentence* not per *word* (5–10×), `sdpa`/
  `flash_attn` not eager (1.3–2×), optional cross-sentence batching.
  Per-token math is byte-identical; `TRIBE_FAST_TEXT_BATCH=0` reverts
  exactly.
- **Apple-Silicon execution** — retargets the extractors + brain model
  onto MPS (upstream's device `Literal` can't take `mps`), CPU fallback
  for ops with no Metal kernel.
- **macOS spawn-safe DataLoader** — upstream's `num_workers: 20` storms
  on macOS *spawn* (each worker reloads Llama/w2v-bert → the predict-stage
  hang). Forced single-process.
- **In-process cached ASR** — replaces the per-run
  `uvx whisperx --device cuda` shell-out (fails on Apple Silicon) with a
  cached in-process whisperx (corr ≈ 0.999), optional `mlx-whisper`.
- **Fewer round-trips** — `HF_HUB_OFFLINE` kills the per-weight etag
  stall; `TORCH_HOME`/`HF_HOME` pinned; the ≈0%-hit on-disk event cache
  disabled (in-track RAM dedupe kept).

`timing.py` prints a per-stage wall-time breakdown (`TRIBE_TIMING`,
default on).

## How the numbers work

- A feature delta = `draft − taste centroid`, **robustly normalized** by
  `max(reference spread, 15% of centroid, an absolute floor)` — a
  perfectly-consistent reference set (spread 0) can't produce infinite
  distances, while tight tastes still count more.
- Overall distance = RMS of the normalized deltas.
- `optimize` perturbs **one actionable feature at a time** toward the
  centroid within a valid step, re-scores, and ranks edits by predicted
  reduction. Confidence is downgraded where the references disagree on
  that feature. Every edit is labeled *"hypothesis to A/B, not a
  guarantee."*

**What it is not:** not a hit predictor, not a stream forecaster, not a
"good vs. bad" grader. Only *near vs. far from the taste you chose.*

## Other entry points

```bash
python -m tastebench                       # the worker, no pip install needed
python -m tastebench compare a.wav b.wav --to demo.wav
make test                                  # model-free smoke suite
pip install -e .                           # manual: core only (numpy/librosa/rich)
python examples/make_examples.py           # lawful synthetic clips (no media shipped)
```

`ffmpeg` must be on PATH for non-WAV audio. Web QA needs
`pip install 'tastebench[web]' && playwright install chromium` (heavy,
optional — the package works fine without it).

## Attribution & development

tastebench is MIT. TRIBE (`facebookresearch/tribev2`), Llama-3.2 and
Whisper are **declared dependencies you install**, not redistributed here
([ATTRIBUTION.md](ATTRIBUTION.md), [NOTICE](NOTICE)). The 12-network
readout follows the Cole-Anticevic Brain-wide Network Partition (Ji et
al., *NeuroImage* 2019), implemented independently.

```bash
pip install -e ".[dev]" && pytest -q       # model-free smoke suite (synthesizes its own audio)
```

No third-party audio, secrets, or model weights are ever committed (see
`.gitignore`) — code only, with one deliberate exception:
`demo/level-up-v4.mp3`, the author's own track, included solely as a test
fixture so [`demo/`](demo/) is reproducible.
