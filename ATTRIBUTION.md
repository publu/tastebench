# Attribution & Upstream Licenses

tribe-taste is MIT-licensed (see `LICENSE`). It builds on third-party
models and libraries that it **does not redistribute** — they are declared
dependencies the user installs/downloads. Their licenses govern their use.

## TRIBE (the brain model)

- **Project:** `tribev2` — Meta AI / FAIR
- **Source:** https://github.com/facebookresearch/tribev2
- **Role here:** a *declared dependency* (`pip install -e ".[brain]"`),
  invoked through the scrubbed `tribe_taste/engine.py`. tribe-taste does
  **not** copy or republish Meta's model source or weights. The model
  cache (~20 GB) is fetched by `scripts/download_models.py` directly from
  the upstream/HF sources into the user's local cache.
- **License:** governed by the upstream `facebookresearch/tribev2`
  repository's LICENSE and any model-card terms. Review them before use.

The Apple-Silicon / device-adaptation modules `tribe_taste/native.py`,
`tribe_taste/fast_text.py`, and `tribe_taste/timing.py` are in-process
patches **around** the upstream package (device selection, MPS fallback,
native Whisper transcription, batched embedding, timing). They were written
for an internal TRIBE inference deployment, contain no upstream model
source, no server/credential/storage code, and are carried here as-is. They
no-op or behave unchanged on a CUDA box.

## Llama 3.2 (text-feature backbone, pulled by tribev2)

- **Model:** `meta-llama/Llama-3.2` family (the pipeline uses a 3B variant;
  by default the open `unsloth/Llama-3.2-3B` mirror).
- **License:** the **Llama 3.2 Community License Agreement** (Meta). The
  gated `meta-llama` weights require accepting that license on Hugging Face
  and a token. Not redistributed here.

## Whisper / wav2vec2 (word transcription & alignment, pulled by the stack)

- **Whisper:** OpenAI Whisper / faster-whisper (CTranslate2) — MIT.
- **Alignment:** `WAV2VEC2_ASR_LARGE_LV60K_960H` (torchaudio / Meta) — its
  own license.
- Optional `mlx-whisper` (Apple) for the M-series ASR path — its own license.
- Not redistributed here; fetched on demand into the model cache.

## Core libraries (the model-free craft layer)

- `librosa` (ISC), `numpy` (BSD), `soundfile`/libsndfile (LGPL), `rich`
  (MIT), and their dependencies — under their respective licenses.

## Cole-Anticevic Brain-wide Network Partition

The 12-network mapping in `tribe_taste/brain.py` follows the published
**Cole-Anticevic Brain-wide Network Partition** (Ji, J.L. et al., 2019,
*NeuroImage* 185:35-57, "Mapping the human brain's cortical-subcortical
functional network organization"). Implemented independently from the
public partition; no third-party code copied.

## Scientific framing

The product's honest-claims framing draws on Salganik, Dodds & Watts 2006
(*Science*, "Experimental Study of Inequality and Unpredictability in an
Artificial Cultural Market"), and the neuroforecasting literature (Berns &
Moore 2012; Boksem & Smidts) — cited, not redistributed.

---

If you redistribute tribe-taste, keep this file and `NOTICE`, and do not
imply any endorsement by Meta, OpenAI, or the cited researchers.
