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

# Your own focus group for your drafts

You have a draft. You have a few things you wish it felt like.
**tastebench tells you exactly where yours is off — and what to change first.**

Working alone, the hardest question is *"is this there yet?"* — not "is
this good in the abstract," but *does it hit the way the stuff I love
hits?* Normally you find out by sitting on it, sending it around, and
waiting. tastebench is that gut-check in **seconds, on your machine**,
before anyone else hears it.

Point it at a handful of **references** — the work you wish yours felt
like (tracks, videos, images). It learns the *taste* they share. Drop
your **draft**. It tells you, in plain language, how close you are, which
reference you're nearest, and a **ranked list of fixes** — biggest lever
first, each with a confidence label and how to act:

```
── synthwave · demo.wav ──────────────────

  DIFFERENT RECORD — 11% taste match

  TASTE MATCH  [███·····················]  11%

  ▸ the one thing: your hook shows up at 11.3s — the refs land theirs by
    ~4s; most listeners are gone before your best moment.
  · closest to ref_b.wav of your set
```

No dashboards, no score out of ten, no engagement-bait. It measures
*distance to a taste **you** chose* and hands you the levers — and it
**does not predict hits** (those are irreducibly noisy; Salganik et al.,
*Science* 2006). That honesty is the point.

### Try it in 30 seconds

```bash
git clone https://github.com/publu/tastebench && cd tastebench
make     # tiny model-free venv, then it starts watching a folder
```

Drop references into `references/<name>/refs/` and your draft into
`…/draft/`. It grades on drop and writes a report next to it. No commands
to learn, nothing to configure, nothing gated to download.

### What you can use it for

- **Finishing a track.** "I want this to hit like these three records —
  where am I off?" Get back: hook lands 11s in (theirs land at 4s), it
  doesn't loop clean, dynamics are flat — *fix the hook timing first.*
- **A/Bing a decision.** Two arrangement or mix versions, same references
  — which one is actually closer to the target, by how much, on which
  signals.
- **A standing taste check.** Keep a folder of your favourite references;
  drop every WIP in; get an instant read before you send it to the artist
  or post it.
- **Matching a visual reference.** Hit the palette, contrast, and
  composition of a moodboard; match the cut pacing and motion energy of
  an edit you admire.
- **Briefing an LLM with real numbers.** `--llm` emits a self-contained
  bundle (raw analysis + full glossary) you paste into any model for a
  grounded, no-hallucination breakdown.

> **Licensing (read before any non-personal use).** This wrapper is MIT,
> but it runs on Meta's **TRIBE v2** (CC-BY-NC-4.0, *NonCommercial*) and
> **Llama 3.2** (Llama 3.2 Community License). The tool *as a whole* is
> **non-commercial — research / personal creative use only**. See
> [LICENSE](LICENSE), [NOTICE](NOTICE), [ATTRIBUTION.md](ATTRIBUTION.md).

## Use it: drop files in a folder

Run one thing. No commands to learn:

```bash
tastebench            # creates ./tastebench/ and watches it
```

It creates and watches a folder tree. One folder per taste — whatever you
put in `refs/` defines that taste, whatever you put in `draft/` gets
graded against it:

```
tastebench/references/
  my-sound/
    refs/    ← a few tracks/videos/images you ADMIRE
    draft/   ← your draft → graded the moment it lands
```

Drop files in and watch. The worker learns the taste, grades each draft
against it live in the terminal, and writes a full `<draft>.report.md`
next to it. Make as many `references/<name>/` folders as you want —
they're independent. It's settle-aware (a half-copied or multi-file drag
never triggers a partial run) and re-grades automatically when anything
changes. The neural layer switches on by itself once its weights are
present (offered as a background download on first run); until then you
get the instant craft read.

## Scripting it: the CLI

Every step is also a plain command, for pipelines and one-shots:

```
tastebench profile  ref1.wav ref2.wav ref3.wav            # what you like
tastebench compare  ref1.wav ref2.wav --to demo.wav       # how you diverge
tastebench optimize demo.wav --toward ref1.wav ref2.wav   # ranked edits
tastebench glossary [TERM]                                # the dictionary
tastebench tui      ref1.wav ref2.wav --demo demo.wav     # the visual view
tastebench drop                                           # legacy drop prompt
```

Add `--llm` to any analysis command to get a self-contained bundle (raw
numbers + the **full explainer glossary** + a framing question) you can
paste into any LLM for a deeper, grounded explanation. Add `--format json`
for machine output, `-o FILE` to write to disk.

## Install — clone, then `make`

```bash
git clone https://github.com/publu/tastebench && cd tastebench
make            # builds .venv (core deps only), launches the worker on ./workspace
```

That's the whole thing. `make` needs only `python3`; the core install is
model-free (numpy/librosa/rich — no torch, sub-second) so there's nothing
to download and nothing gated. The worker then prints a folder to drop
files into. Drop references in `workspace/references/<name>/refs/` and a
draft in `…/draft/` → it grades automatically and writes a report.

Other entry points (all equivalent — no install needed for `-m`):

```bash
python -m tastebench                      # the worker, no pip install
python -m tastebench compare a.wav b.wav --to demo.wav
make test                                 # the model-free smoke suite
make brain                                # add the optional ~20 GB TRIBE stack
```

`ffmpeg` must be on PATH to read non-WAV audio.

### Manual setup (no make)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                   # core only: numpy, librosa, soundfile, rich
tastebench                         # the worker (from any dir but the repo root)
# or a one-shot, model-free:
python examples/make_examples.py   # lawful synthetic clips (no media shipped)
tastebench compare examples/ref_a.wav examples/ref_b.wav \
    --to examples/demo.wav --no-brain
```

### The brain layer (optional, heavier)

```bash
make brain                         # core + torch + Meta's tribev2 stack
huggingface-cli login              # accept Meta's Llama 3.2 license (gated)
.venv/bin/python scripts/download_models.py   # ~20 GB → ~/.cache/tastebench
```

The brain stack (torch / tribev2 / whisperx) wants **Python 3.11–3.12**;
`make` auto-picks a compatible interpreter if one is on PATH (override with
`make PY=python3.12`). The model-free core has no such limit and installs on
3.13+ too. The worker flips to the brain layer automatically once the
weights exist; until then everything runs craft-only. None of this is
required to use the tool.

### Hardware reality (read this)

- **~20 GB** model cache (fMRI encoder + Llama-3.2-3B + Whisper + wav2vec2).
- **Apple Silicon** (MPS, with CPU fallback) **or CUDA**. The device
  adaptation auto-detects; it runs on a Mac, just slower than a GPU box.
- **Order of minutes per clip** for the brain layer on Apple Silicon
  (transcription + feature extraction + the brain forward pass dominate).
  The craft layer is sub-second.
- **HF token note:** Llama-3.2 is gated. You must accept Meta's Llama 3.2
  license on Hugging Face and authenticate, or model download will 401.
- If the cache is absent the package still **imports cleanly**; the brain
  layer reports a clear *"models not downloaded — run
  scripts/download_models.py"* and analysis falls back to craft-only.

## The explainer dictionary

A first-class deliverable: `tastebench/explainers/explainers.json` — one
entry per craft feature, brain network, brain-ROI group, and edit type, each
with a plain sentence, a detail (what it measures + why it matters), how it's
computed, units/range, how to act (for actionable features), and references.
Every `compare`/`optimize` line attaches its matching entry; `report --llm`
embeds the whole dictionary. Browse it with `tastebench glossary`.

## How it works

Two layers run on each file.

**Brain.** Your references and your draft go through TRIBE — Meta's
*fMRI-encoding* model, which predicts the brain response a piece of audio,
video, or text evokes. tastebench reads that prediction out as a
12-network *Cole-Anticevic* signature: how strongly the work drives the
auditory, reward, default-mode (DMN), frontoparietal and other networks.
Your references define a target signature; your draft is scored against how
far it sits from it, network by network.

**Craft.** A model-free layer (librosa-derived) measures the concrete,
fixable stuff — hook timing, loopability, chorus lift, tempo and key
stability, dynamics; colour, contrast and motion for video and images. No
download, sub-second; it's also what the tool falls back to when the model
isn't installed.

| Layer | What it measures | Needs the model? | Non-audio? |
|---|---|---|---|
| Brain | TRIBE → 12-network Cole-Anticevic signature | Yes (~20 GB) | yes — carries video/image |
| Craft | librosa musician-actionable features | No | no-ops gracefully |

The brain layer is a *hypothesis view* — it reports a *predicted* neural
response, not a validated outcome — and the tool says so wherever it
appears. Brains differ; this profiles a response *pattern*, not a verdict.

## Why it's faster than vanilla TRIBE v2

Upstream `tribev2` assumes a single CUDA box; run as-is on a Mac it is
slow or fails outright. `tastebench/native.py` and
`tastebench/fast_text.py` are an in-process adaptation layer that runs the
same pipeline with the same numerics. Every change is opt-out via env
vars, so the CUDA path is unchanged.

- **Llama word embeddings** (`fast_text.py`) — ~15–40× on the audio path,
  from three stacked changes: load Llama-3.2 in bf16 instead of the fp32
  default (2–3×); one forward pass per *unique sentence* instead of per
  *word*, since consecutive word events share a sentence (5–10×);
  `sdpa`/`flash_attention_2` instead of the eager kernel (1.3–2×); plus
  optional cross-sentence batching (weights streamed from memory once per
  batch — the workload is memory-bandwidth bound on MPS). The per-token
  slice math is byte-identical to upstream (Llama is a causal decoder, so
  a word's hidden states don't depend on later tokens);
  `TRIBE_FAST_TEXT_BATCH=0` is an exact revert.
- **Apple-Silicon execution** (`native.py`) — upstream resolves
  `device="auto"` to cuda/cpu only and its pydantic device `Literal`
  cannot take `mps`. This retargets the lazily-loaded extractors and the
  brain model onto MPS, with CPU fallback for the ops lacking a Metal
  kernel.
- **macOS spawn-safe DataLoader** — upstream ships `num_workers: 20`
  (tuned for a Linux GPU box). macOS uses *spawn*, so each worker does a
  fresh interpreter import and reloads Llama/w2v-bert, pegging a core for
  minutes and parking the parent in an OpenMP fork barrier (the
  predict-stage hang). Forced to single-process.
- **In-process, cached ASR** — upstream shells out to `uvx whisperx
  --device cuda --compute_type float16` every run: it re-resolves the
  package from PyPI and fails on Apple Silicon (no CUDA; CTranslate2-CPU
  has no float16). Replaced with an in-process call to a cached whisperx
  (corr ≈ 0.999 vs. the upstream transcript), with an optional Apple-GPU
  `mlx-whisper` path.
- **Fewer network/disk round-trips** — `HF_HUB_OFFLINE` stops
  huggingface_hub making a per-model online etag call on every cached
  weight (an observed multi-minute SSL stall); `TORCH_HOME`/`HF_HOME` are
  pinned so the 1.2 GB wav2vec2 align checkpoint resolves from cache
  instead of re-downloading; the on-disk per-event cache (≈0% hit rate
  across tracks) is disabled, keeping only the in-track RAM dedupe.

Above the engine, the worker caches a taste profile per reference-set
signature and only re-grades a draft when it or the refs changed, so
TRIBE does not re-run on unchanged inputs; the sub-second craft layer
answers while the brain layer is downloading or between drops.
`tastebench/timing.py` prints a per-stage wall-time breakdown
(`TRIBE_TIMING`, default on).

## How the numbers work (transparent by design)

- A feature delta = `demo − taste centroid`, **robustly normalized** by
  `max(reference spread, 15% of the centroid, an absolute floor)`. This
  keeps a perfectly-consistent reference set (spread = 0) from producing
  infinite distances, while still weighting tight tastes more.
- Overall distance = RMS of those normalized deltas.
- `optimize` perturbs **one musician-actionable feature at a time** toward
  the centroid within a musically-valid step, re-scores the craft distance,
  and ranks edits by predicted reduction. Confidence is downgraded where the
  reference set is itself inconsistent on that feature. Every edit is
  labeled *"hypothesis to A/B, not a guarantee."*

## What this is not

- Not a hit predictor. Not a stream forecaster. Not a grader of "good vs
  bad" in the abstract — only *near vs far from the taste you gave it*.
- The brain layer is unvalidated against outcomes and is shown as such.

## Upstream & attribution

tastebench is MIT-licensed. The TRIBE brain model
(`facebookresearch/tribev2`), Llama-3.2, and Whisper are **declared
dependencies the user installs** — not redistributed here. See
[`ATTRIBUTION.md`](ATTRIBUTION.md) and [`NOTICE`](NOTICE). The 12-network
readout follows the published Cole-Anticevic Brain-wide Network Partition
(Ji et al., *NeuroImage* 2019), implemented independently.

## Development

```bash
pip install -e ".[dev]"
pytest -q              # model-free smoke suite (synthesizes its own audio)
```

No audio, secrets, model weights, or large caches are ever committed (see
`.gitignore`). The repo ships code only.
