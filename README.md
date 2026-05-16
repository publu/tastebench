# tastebench

**Learn the taste signature of media you admire — then see exactly how your
own demo diverges, and what to change.**

> ⚠️ **Non-commercial — research / personal creative use only.**
> tastebench's own code is MIT, but it runs on Meta's **TRIBE v2 model,
> which is licensed CC-BY-NC-4.0 (NonCommercial)**. The MIT license on this
> wrapper does **not** grant commercial rights to the model it depends on, so
> the tool *as a whole* may not be used commercially. See
> [LICENSE](LICENSE), [NOTICE](NOTICE), and [ATTRIBUTION.md](ATTRIBUTION.md).
>
> **Built with Llama.** The brain layer uses Llama 3.2 — *"Llama 3.2 is
> licensed under the Llama 3.2 Community License, Copyright © Meta Platforms,
> Inc. All Rights Reserved."*

You give tastebench a few **reference** tracks (or videos/images) you think
are great — music, video, or images. It computes the *taste signature* of
what they share — a model-free **craft fingerprint** (for music: hook
timing, loopability, tempo, key stability, dynamics; for video & images:
colour scheme, contrast, saturation, palette, composition — plus cut pacing
and motion for video) and, optionally, a 12-network *brain-response*
signature from Meta's TRIBE model.
Then you give it your own **demo**. It tells you, in plain musical language,
where the demo is off your taste, how far, which reference it's nearest to,
and a **ranked, confidence-labeled edit list** to move it toward the taste —
each edit attached to a glossary entry explaining what it is and how to act.

It does **not** predict hits. Hit outcomes are irreducibly noisy (Salganik
et al., *Science* 2006). tastebench measures *distance to a taste you
defined*, says it in words, and hands you levers. Honesty is a feature.

## The reference → demo flow

```
tastebench profile  ref1.wav ref2.wav ref3.wav            # what you like
tastebench compare  ref1.wav ref2.wav --to demo.wav       # how you diverge
tastebench optimize demo.wav --toward ref1.wav ref2.wav   # ranked edits
tastebench glossary [TERM]                                # the dictionary
tastebench tui      ref1.wav ref2.wav --demo demo.wav     # the visual view
```

Add `--llm` to any analysis command to get a self-contained bundle (raw
numbers + the **full explainer glossary** + a framing question) you can
paste into any LLM for a deeper, grounded explanation. Add `--format json`
for machine output, `-o FILE` to write to disk.

## What it does

**It's a private focus group for your drafts.**

Hand it a few things you wish your work felt like — tracks, videos, or
images you admire — plus your own rough draft. tastebench simulates how a
listener's brain reacts to each, learns the specific *vibe* your references
share, and shows you where your draft misses it and what to change to close
the gap.

It's the second pair of ears you don't always have: a quick way to check
you're heading the right direction and tighten the work *before* you show
friends or launch. Not "does this grab people" in general — but "does this
hit the exact response the stuff I love hits."

It won't be perfect — every brain is different — but once you've profiled
the response your references share, you can push your song, video, or image
toward that vibe on purpose instead of guessing.

## Quickstart

```bash
git clone <this repo> && cd tastebench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[brain]"          # core + torch + Meta's tribev2 stack
huggingface-cli login              # accept Meta's Llama 3.2 license (gated)
python scripts/download_models.py  # ~20 GB → ~/.cache/tastebench

python examples/make_examples.py   # lawful synthetic clips (no media shipped)
tastebench compare  examples/ref_a.wav examples/ref_b.wav --to examples/demo.wav
tastebench optimize examples/demo.wav --toward examples/ref_a.wav examples/ref_b.wav
```

`ffmpeg` must be on PATH to read non-WAV audio.

### Craft-only (no model)

```bash
pip install -e .                   # core only: numpy, librosa, soundfile, rich
tastebench compare examples/ref_a.wav examples/ref_b.wav \
    --to examples/demo.wav --no-brain
```

Sub-second, no download — the same flow without the brain layer.

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
