"""Apple-Silicon native adaptation for the TRIBE pipeline.

Provenance: this in-process adaptation layer was written for an internal
TRIBE inference deployment to run Meta's upstream ``tribev2`` pipeline
natively on Apple Silicon (and unchanged on CUDA). It contains no server,
credential, or storage code -- only device/runtime patches around the
upstream wheels. Carried into tribe-taste as-is. Upstream model:
facebookresearch/tribev2 (a declared dependency; see ATTRIBUTION.md).


The upstream ``tribev2`` package assumes a single CUDA box:

* ``TribeModel.from_pretrained`` resolves ``device="auto"`` to
  ``cuda``/``cpu`` only — never ``mps``.
* The neuralset audio/text extractors take a pydantic ``Literal["auto",
  "cpu","cuda","accelerate"]`` device, so ``mps`` cannot be passed through
  config and ``auto`` collapses to ``cpu`` on a Mac.
* Word transcription shells out to ``uvx whisperx ... --device cuda
  --compute_type float16``. On Apple Silicon there is no CUDA, and
  CTranslate2 (faster-whisper) on CPU does not support ``float16`` — the
  original call hard-fails. ``uvx`` also needs to resolve the package from
  PyPI on every run.

This module installs in-process patches so the exact same pipeline runs
natively on an M-series Mac (MPS where it helps, CPU otherwise) without
touching the upstream wheels. Everything is opt-out via env vars so the
CUDA path is unchanged when this code runs on a GPU box.

Env:
  TRIBE_DEVICE       auto|mps|cpu|cuda   (default auto -> mps if available)
  TRIBE_ASR_ENGINE   whisperx|mlx        (default whisperx)
  TRIBE_ASR_MODEL    whisper model id    (default large-v3)
  TRIBE_ASR_COMPUTE  int8|float32|...    (CTranslate2 compute type, CPU)
  TRIBE_ASR_BATCH    whisperx batch size (default 16)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

LOGGER = logging.getLogger(__name__)

_INSTALLED = False


def resolve_device(prefer: str | None = None) -> str:
    """Resolve a concrete torch device string for this box.

    ``TRIBE_DEVICE`` wins. Otherwise ``prefer`` (a config-supplied device),
    otherwise auto-detect: cuda > mps > cpu.
    """
    env = os.environ.get("TRIBE_DEVICE", "auto").strip().lower()
    if env and env != "auto":
        return env
    if prefer and prefer not in ("auto", "accelerate"):
        if prefer == "cuda" and not torch.cuda.is_available():
            pass  # fall through to auto-detect
        else:
            return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _model_cache() -> Path:
    if os.environ.get("MODEL_CACHE"):
        return Path(os.environ["MODEL_CACHE"]).expanduser()
    import platform

    if platform.system() == "Darwin":
        return Path.home() / ".cache" / "tribe-vast" / "model-cache"
    return Path("/model-cache")


def apply_runtime_env() -> None:
    """Set process env that makes MPS/CPU inference reliable and fast."""
    # Any op without an MPS kernel transparently runs on CPU instead of
    # raising. TRIBE's brain model + SeamlessM4T have a few such ops.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Let CTranslate2 / torch CPU paths use every performance core.
    ncpu = os.cpu_count() or 8
    os.environ.setdefault("OMP_NUM_THREADS", str(ncpu))
    # Pin TORCH_HOME / HF_HOME to the model cache so the 1.2 GB wav2vec2
    # align checkpoint (torch.hub) and the HF weights resolve from cache
    # regardless of $HOME. The whisperx subprocess inherits this env, so
    # it finds the same cached align model instead of re-downloading.
    mc = _model_cache()
    os.environ.setdefault("TORCH_HOME", str(mc / "torch"))
    os.environ.setdefault("HF_HOME", str(mc / "hf"))
    os.environ.setdefault("MNE_DATA", str(mc / "mne"))
    # The entire model set ships in the R2 tarball. Without offline mode,
    # huggingface_hub still makes online etag/metadata calls on every
    # cached model and STALLS for minutes on a flaky link (observed:
    # SSL/HTTP-2 hang in CLOSE_WAIT to the HF CDN). Force offline so a
    # cache miss fails fast instead of hanging the whole pipeline. The
    # whisperx subprocess inherits this env too. Opt out with
    # TRIBE_ALLOW_NET=1 (e.g. first-time warm of an uncached asset).
    if os.environ.get("TRIBE_ALLOW_NET") != "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Even if something does reach out, never block the pipeline for long.
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "3")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "10")


def install_singleproc_dataloader() -> None:
    """Force the tribev2 DataLoader to num_workers=0 on macOS.

    The 5090 `config.yaml` ships `num_workers: 20`. On Linux that forks
    cheap workers; on macOS Python multiprocessing uses *spawn*, so each
    of the 20 workers does a full fresh interpreter import AND reloads
    Llama/w2v-bert — pegging one core for minutes and parking the parent
    in an OpenMP fork barrier (this was the predict-stage "hang"). Single
    process is both correct and far faster here. Opt out: TRIBE_DL_WORKERS.
    """
    import sys

    if sys.platform != "darwin" and os.environ.get("TRIBE_FORCE_SINGLEPROC") != "1":
        return
    want = int(os.environ.get("TRIBE_DL_WORKERS", "0"))
    try:
        from neuralset.dataloader import SegmentDataset
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("[native] dataloader patch skipped: %r", e)
        return
    if getattr(SegmentDataset, "_tribe_singleproc", False):
        return
    _orig = SegmentDataset.build_dataloader

    def _build(self, **kw):  # noqa: ANN001
        kw["num_workers"] = want
        kw["persistent_workers"] = False
        kw.pop("prefetch_factor", None)
        return _orig(self, **kw)

    SegmentDataset.build_dataloader = _build
    SegmentDataset._tribe_singleproc = True
    LOGGER.info("[native] DataLoader forced to num_workers=%d (macOS spawn-safe)", want)


def patch_extractor_devices(model) -> str:
    """Point the lazy-loaded HF extractors at the resolved device.

    ``neuralset`` extractors store ``device`` as a plain attribute after
    pydantic post-init (the model is not frozen and has no
    ``validate_assignment``), and the underlying HF model is loaded lazily
    on first use — so reassigning ``.device`` here, before ``predict()``,
    is honoured when the weights load. ``fast_text`` already handles the
    text extractor via its own device picker; this covers audio (and any
    other HuggingFace extractor) which reads ``self.device`` directly.
    """
    device = resolve_device()
    data = getattr(model, "data", None)
    if data is None:
        return device
    for attr in ("audio_feature", "text_feature", "video_feature", "image_feature"):
        ext = getattr(data, attr, None)
        if ext is None or not hasattr(ext, "device"):
            continue
        # text_feature is driven by fast_text._pick_device(TRIBE_DEVICE);
        # leave its attribute alone so the two cannot disagree.
        if attr == "text_feature":
            continue
        try:
            object.__setattr__(ext, "device", device)
            LOGGER.info("[native] %s.device -> %s", attr, device)
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("[native] could not set %s.device: %r", attr, e)
    return device


# --------------------------------------------------------------------------
# Word transcription (replaces the `uvx whisperx --device cuda` subprocess)
# --------------------------------------------------------------------------

_LANG_CODES = {"english": "en", "french": "fr", "spanish": "es", "dutch": "nl", "chinese": "zh"}


def _asr_settings() -> dict:
    engine = os.environ.get("TRIBE_ASR_ENGINE", "whisperx").strip().lower()
    return {
        "engine": engine,
        "model": os.environ.get("TRIBE_ASR_MODEL", "large-v3"),
        "compute": os.environ.get("TRIBE_ASR_COMPUTE", "int8"),
        "batch": int(os.environ.get("TRIBE_ASR_BATCH", "16")),
    }


def _whisperx_bin() -> list[str]:
    """Resolve the isolated whisperx 3.8.5 CLI (mirrors upstream `uvx`).

    Order: explicit override -> the `uv tool install`ed binary under the
    cache HOME -> `uvx whisperx==3.8.5` (pinned to the 5090's version).
    Kept out of the tribev2 venv on purpose: whisperx 3.8.5 pulls
    transformers 4.57 / torch 2.8 / pyannote 4 which would fight
    tribev2's transformers 5 / torch 2.6.
    """
    override = os.environ.get("TRIBE_WHISPERX_BIN")
    if override:
        return override.split()
    home = os.environ.get("HOME", "")
    cand = Path(home) / ".local" / "bin" / "whisperx"
    if cand.is_file():
        return [str(cand)]
    return ["uvx", "whisperx==3.8.5"]


def _whisperx_transcript(wav_filename: Path, language: str):
    """faster-whisper (CTranslate2, CPU on Mac) + wav2vec2 forced alignment.

    Byte-for-byte the same CLI the 5090 baseline used (whisperx 3.8.5,
    large-v3, WAV2VEC2_ASR_LARGE_LV60K_960H, batch 16, json out) — only
    `--device`/`--compute_type` differ, because CTranslate2 has no Metal
    backend (CPU only) and float16 is CUDA-only there. Same parse as the
    upstream `_get_transcript_from_audio`, so the events are identical
    given an identical transcript.
    """
    import json
    import subprocess
    import tempfile

    import pandas as pd

    cfg = _asr_settings()
    if language not in _LANG_CODES:
        raise ValueError(f"Language {language} not supported")
    lang = _LANG_CODES[language]
    compute_type = cfg["compute"]  # int8 (fast) | float32 (faithful)

    with tempfile.TemporaryDirectory() as out_dir:
        cmd = [
            *_whisperx_bin(),
            str(wav_filename),
            "--model", cfg["model"],
            "--language", lang,
            "--device", "cpu",
            "--compute_type", compute_type,
            "--batch_size", str(cfg["batch"]),
            "--align_model",
            "WAV2VEC2_ASR_LARGE_LV60K_960H" if language == "english" else "",
            "--output_dir", out_dir,
            "--output_format", "json",
            "--threads", str(os.cpu_count() or 8),
        ]
        cmd = [c for c in cmd if c]
        LOGGER.info("[native] whisperx (cpu/%s) %s", compute_type, " ".join(cmd[:3]))
        env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
        import time as _t

        _t0 = _t.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        print(f"[timing] whisperx subprocess={_t.monotonic() - _t0:.1f}s", flush=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"whisperx failed (rc={proc.returncode}):\n{proc.stderr[-4000:]}"
            )
        transcript = json.loads(
            (Path(out_dir) / f"{wav_filename.stem}.json").read_text()
        )

    words = []
    for i, segment in enumerate(transcript["segments"]):
        sentence = str(segment.get("text", "")).replace('"', "")
        for w in segment.get("words", []):
            if "start" not in w or "end" not in w:
                continue
            words.append({
                "text": str(w["word"]).replace('"', ""),
                "start": float(w["start"]),
                "duration": float(w["end"]) - float(w["start"]),
                "sequence_id": i,
                "sentence": sentence,
            })
    return pd.DataFrame(words)


def _mlx_transcript(wav_filename: Path, language: str):
    """mlx-whisper (Apple GPU) transcription, optional wav2vec2 realignment.

    Much faster than CTranslate2-CPU large-v3 on M-series. Word timestamps
    come from whisper DTW unless TRIBE_ASR_ALIGN=1, which re-aligns with
    the same wav2vec2 model the baseline used for closer fidelity.
    """
    import mlx_whisper
    import pandas as pd

    cfg = _asr_settings()
    if language not in _LANG_CODES:
        raise ValueError(f"Language {language} not supported")
    lang = _LANG_CODES[language]
    repo = os.environ.get(
        "TRIBE_MLX_MODEL",
        f"mlx-community/whisper-{cfg['model']}-mlx",
    )
    LOGGER.info("[native] mlx-whisper model=%s", repo)
    result = mlx_whisper.transcribe(
        str(wav_filename),
        path_or_hf_repo=repo,
        language=lang,
        word_timestamps=True,
        condition_on_previous_text=False,
    )

    segments = result["segments"]
    if os.environ.get("TRIBE_ASR_ALIGN", "1") == "1":
        try:
            import whisperx

            audio = whisperx.load_audio(str(wav_filename))
            adev = resolve_device()
            adev = "cpu" if adev == "mps" else adev
            am, meta = whisperx.load_align_model(
                language_code=lang, device=adev,
                model_name=("WAV2VEC2_ASR_LARGE_LV60K_960H" if language == "english" else None),
            )
            segments = whisperx.align(
                [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segments],
                am, meta, audio, adev, return_char_alignments=False,
            )["segments"]
            del am
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("[native] wav2vec2 realign failed (%r); using whisper timestamps", e)

    words = []
    for i, segment in enumerate(segments):
        sentence = str(segment.get("text", "")).replace('"', "")
        for w in segment.get("words", []):
            if w.get("start") is None or w.get("end") is None:
                continue
            words.append({
                "text": str(w["word"]).replace('"', ""),
                "start": float(w["start"]),
                "duration": float(w["end"]) - float(w["start"]),
                "sequence_id": i,
                "sentence": sentence,
            })
    return pd.DataFrame(words)


_MLX_WARNED = False


def _native_get_transcript(wav_filename, language: str):
    global _MLX_WARNED
    wav_filename = Path(wav_filename)
    engine = _asr_settings()["engine"]
    if engine == "mlx":
        if not _MLX_WARNED:
            LOGGER.warning(
                "=" * 72 + "\n"
                "[native] TRIBE_ASR_ENGINE=mlx is EXPERIMENTAL and NOT faithful.\n"
                "mlx-whisper decodes ~15% of words differently from the 5090's\n"
                "faster-whisper/CTranslate2 stack -> different predictions, NOT\n"
                "validated against any baseline. Do not use for scored/leaderboard\n"
                "work. Default is whisperx (faithful, corr ~0.999).\n" + "=" * 72
            )
            _MLX_WARNED = True
        return _mlx_transcript(wav_filename, language)
    return _whisperx_transcript(wav_filename, language)


def install_whisper_patch() -> None:
    """Replace ExtractWordsFromAudio._get_transcript_from_audio in place."""
    from tribev2 import eventstransforms

    if getattr(eventstransforms.ExtractWordsFromAudio, "_native_patched", False):
        return
    eventstransforms.ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(
        _native_get_transcript
    )
    eventstransforms.ExtractWordsFromAudio._native_patched = True
    LOGGER.info(
        "[native] patched ExtractWordsFromAudio (engine=%s)",
        _asr_settings()["engine"],
    )


def install() -> None:
    """Idempotent: apply all native patches. Call before model load."""
    global _INSTALLED
    if _INSTALLED:
        return
    apply_runtime_env()
    # Pin the resolved device into TRIBE_DEVICE so every component agrees:
    # fast_text._pick_device, native.resolve_device, and the brain model
    # all read the same value instead of each re-deriving "auto".
    os.environ.setdefault("TRIBE_DEVICE", resolve_device())
    install_whisper_patch()
    install_singleproc_dataloader()
    try:
        from . import timing
        timing.install()
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("[native] timing install failed: %r", e)
    _INSTALLED = True
    LOGGER.info(
        "[native] Apple-Silicon adaptation installed (device=%s, asr=%s)",
        os.environ.get("TRIBE_DEVICE"), _asr_settings()["engine"],
    )
