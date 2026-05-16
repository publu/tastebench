# tribe-taste

**Learn the taste signature of media you admire — then see exactly how your
own demo diverges, and what to change.**

> ⚠️ **Non-commercial — research / personal creative use only.**
> tribe-taste's own code is MIT, but it runs on Meta's **TRIBE v2 model,
> which is licensed CC-BY-NC-4.0 (NonCommercial)**. The MIT license on this
> wrapper does **not** grant commercial rights to the model it depends on, so
> the tool *as a whole* may not be used commercially. See
> [LICENSE](LICENSE), [NOTICE](NOTICE), and [ATTRIBUTION.md](ATTRIBUTION.md).
>
> **Built with Llama.** The brain layer uses Llama 3.2 — *"Llama 3.2 is
> licensed under the Llama 3.2 Community License, Copyright © Meta Platforms,
> Inc. All Rights Reserved."*

You give tribe-taste a few **reference** tracks (or videos/images) you think
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
et al., *Science* 2006). tribe-taste measures *distance to a taste you
defined*, says it in words, and hands you levers. Honesty is a feature.

## The reference → demo flow

```
tribe-taste profile  ref1.wav ref2.wav ref3.wav            # what you like
tribe-taste compare  ref1.wav ref2.wav --to demo.wav       # how you diverge
tribe-taste optimize demo.wav --toward ref1.wav ref2.wav   # ranked edits
tribe-taste glossary [TERM]                                # the dictionary
tribe-taste tui      ref1.wav ref2.wav --demo demo.wav     # the visual view
```

Add `--llm` to any analysis command to get a self-contained bundle (raw
numbers + the **full explainer glossary** + a framing question) you can
paste into any LLM for a deeper, grounded explanation. Add `--format json`
for machine output, `-o FILE` to write to disk.

## Two layers

| Layer | What | Needs the model? | Non-audio? |
|---|---|---|---|
| **Craft** | librosa-derived musician-actionable features (time-to-hook, loopability, chorus lift, intro length, tempo/key stability, brightness, dynamics…) | **No** | no-ops gracefully |
| **Brain** | TRIBE predicted brain response → 12-network Cole-Anticevic signature (auditory/reward/DMN/frontoparietal…) | Yes (~20 GB) | yes — carries video/image |

The craft layer is fully model-free and is the default that always works.
The brain layer is a **hypothesis view** (the spec is explicit that the
neural layer is not yet outcome-validated) and is opt-in.

## Quickstart (craft layer, no model)

```bash
git clone <this repo> && cd tribe-taste
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # core: numpy, librosa, soundfile, rich
python examples/make_examples.py # synthesize lawful demo clips (no media shipped)

tribe-taste compare examples/ref_a.wav examples/ref_b.wav \
    --to examples/demo.wav --no-brain
tribe-taste optimize examples/demo.wav \
    --toward examples/ref_a.wav examples/ref_b.wav
```

`ffmpeg` must be on PATH to read non-WAV audio.

## Adding the brain layer (optional, heavy)

```bash
pip install -e ".[brain]"          # adds torch + Meta's tribev2 stack
huggingface-cli login              # Llama-3.2 is gated — needs HF access
python scripts/download_models.py  # ~20 GB into ~/.cache/tribe-taste
# then drop --no-brain
tribe-taste compare ref_a.wav ref_b.wav --to demo.wav
```

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

A first-class deliverable: `tribe_taste/explainers/explainers.json` — one
entry per craft feature, brain network, brain-ROI group, and edit type, each
with a plain sentence, a detail (what it measures + why it matters), how it's
computed, units/range, how to act (for actionable features), and references.
Every `compare`/`optimize` line attaches its matching entry; `report --llm`
embeds the whole dictionary. Browse it with `tribe-taste glossary`.

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

tribe-taste is MIT-licensed. The TRIBE brain model
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
