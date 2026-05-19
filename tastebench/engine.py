"""tastebench.engine — the pure brain-prediction path.

This is a scrubbed re-implementation of the in-memory `compute_predictions`
path from an internal TRIBE inference server. Everything server-side has been
removed: NO FastAPI, NO boto3 / S3 / R2, NO upload branch, NO job queue, NO
credentials. The only thing kept is the value: turn a media file into the
TRIBE brain-response prediction array.

Upstream model: Meta AI's `tribev2`
(https://github.com/facebookresearch/tribev2) — a *declared dependency*, not
vendored here. It pulls Llama-3.2-3B + Whisper + wav2vec2 + the fMRI encoder
(~20 GB). See `scripts/download_models.py` and ATTRIBUTION.md.

The Apple-Silicon / device adaptations live in `tastebench.native` and
`tastebench.fast_text` (carried from the same internal work, attributed in
those files). They no-op on a CUDA box.

Public API
----------
    predict(path) -> (preds: np.ndarray[T, V], info: dict)

`predict` imports cleanly even with no model cache present; the clear
"models not downloaded" error is only raised when you actually call it.
The craft path (`tastebench.features.librosa_report`) needs no model at
all and is the graceful-degradation path for model-free use.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

# Same extension sets as upstream's runner. The brain layer is multimodal.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus"}
SUPPORTED_EXTS = IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS

_MODEL = None


class ModelsNotDownloaded(RuntimeError):
    """Raised when the brain model cache / `tribev2` package is missing."""


def _model_cache_dir() -> Path:
    import os
    import platform

    if os.environ.get("MODEL_CACHE"):
        return Path(os.environ["MODEL_CACHE"]).expanduser()
    if platform.system() == "Darwin":
        return Path.home() / ".cache" / "tastebench" / "model-cache"
    return Path.home() / ".cache" / "tastebench" / "model-cache"


_DOWNLOAD_HINT = (
    "TRIBE brain model not available.\n"
    "  - Install the upstream package:  pip install 'tribev2 @ "
    "git+https://github.com/facebookresearch/tribev2.git'\n"
    "  - Download the ~20 GB model cache:  python scripts/download_models.py\n"
    "  - A Hugging Face token with Llama-3.2 access is required "
    "(huggingface-cli login).\n"
    "The craft/librosa path needs NO model and works without any of this:\n"
    "  tastebench profile / compare / optimize still run on the craft layer."
)


def models_available() -> bool:
    """True if the `tribev2` package imports and a non-empty cache exists.

    Cheap, import-safe, no model load. Used by the CLI/TUI to decide whether
    to offer the brain layer or fall back to the craft-only path.
    """
    try:
        import importlib.util

        if importlib.util.find_spec("tribev2") is None:
            return False
    except Exception:
        return False
    cache = _model_cache_dir()
    return cache.is_dir() and any(cache.iterdir())


def _resolve_device() -> str:
    try:
        from . import native

        return native.resolve_device()
    except Exception:
        return "cpu"


def _total_ram_gb() -> float:
    """Total physical RAM in GiB. Conservative 16.0 if it can't be read."""
    import os

    try:
        import platform
        import subprocess

        if platform.system() == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, check=True,
            )
            return int(out.stdout.strip()) / (1024 ** 3)
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (
            1024 ** 3
        )
    except Exception:
        return 16.0


# (min RAM GiB, num_frames, max_imsize). Upstream runs video through
# `vjepa2-vitg-fpc64-256` at 64 frames/clip full-res — an unbounded working
# set that OOM-panics a 32 GB Mac on the first clip. `num_frames` is the
# dominant RAM+speed lever; `max_imsize` barely moves speed (the model is
# natively 256 px) and only trades RAM headroom, so it stays modest.
# Calibrated on a 32 GB M1 Pro: 8 frames completes with ~1.7 GB headroom,
# 16 frames OOMs — every tier below stays inside that envelope. Descending.
_VIDEO_RAM_TIERS = (
    (96, 48, 288),
    (64, 24, 288),
    (48, 16, 256),
    (36, 10, 256),
    (32,  8, 256),
    (24,  6, 256),
    (16,  4, 224),
    (0,   4, 192),
)


def _auto_video_config(device: str, quiet: bool = False) -> dict:
    """RAM-aware caps for the video extractor so the brain video path fits
    on any Apple-Silicon Mac instead of OOM-panicking it.

    The video counterpart of the text ``config_update`` in ``get_model``:
    upstream never caps ``data.video_feature``, so video ran
    ``vjepa2-vitg-fpc64-256`` at 64 frames/clip full-res and exhausted a
    32 GB Mac. We pick ``num_frames``/``max_imsize`` from total RAM.

    Opt-out / override (mirrors the other speed-layer env knobs):
      ``TRIBE_VIDEO_AUTO=0``    use upstream defaults (full fidelity)
      ``TRIBE_VIDEO_FRAMES=N``  force num_frames
      ``TRIBE_VIDEO_IMSIZE=N``  force max_imsize (0 = no cap)

    CUDA, a ≥128 GiB box, or ``TRIBE_VIDEO_AUTO=0`` are left untouched — a
    GPU box / Modal run keeps full upstream fidelity (64 frames, full res,
    transcription) by design. Fewer frames than that baseline is the
    documented Apple-Silicon trade (see README "Hardware reality").
    """
    import os

    if device == "cuda" or os.environ.get("TRIBE_VIDEO_AUTO", "1") == "0":
        return {}

    ram = _total_ram_gb()
    if ram >= 128:
        return {}  # workstation: treat like a GPU box, no caps

    frames, imsize = next(
        (f, i) for lo, f, i in _VIDEO_RAM_TIERS if ram >= lo
    )

    fenv = os.environ.get("TRIBE_VIDEO_FRAMES")
    ienv = os.environ.get("TRIBE_VIDEO_IMSIZE")
    if fenv:
        frames = int(fenv)
    if ienv is not None:
        imsize = int(ienv)

    cfg: dict = {"data.video_feature.num_frames": frames}
    if imsize:  # 0 / unset -> leave upstream default (no cap)
        cfg["data.video_feature.max_imsize"] = imsize

    if not quiet:
        try:
            import logging

            logging.getLogger(__name__).info(
                "[engine] RAM-aware video config: %.0f GiB -> %s "
                "(TRIBE_VIDEO_AUTO=0 to disable; "
                "TRIBE_VIDEO_FRAMES/IMSIZE to override)",
                ram, cfg,
            )
        except Exception:
            pass
    return cfg


def describe_video_autoconfig() -> "str | None":
    """One-line, model-free summary of the RAM-aware video caps for the
    startup banner. ``None`` when nothing is capped (CUDA / ≥128 GiB /
    ``TRIBE_VIDEO_AUTO=0``) — there is nothing to tell the user then.
    """
    try:
        device = _resolve_device()
        cfg = _auto_video_config(device, quiet=True)
        if not cfg:
            return None
        nf = cfg.get("data.video_feature.num_frames")
        im = cfg.get("data.video_feature.max_imsize")
        ram = _total_ram_gb()
        tail = f", {im}px" if im else ""
        return (
            f"video auto-tuned for {ram:.0f} GB RAM → {nf} frames/clip{tail} "
            f"(set TRIBE_VIDEO_FRAMES / TRIBE_VIDEO_AUTO=0 to change)"
        )
    except Exception:
        return None


def get_model():
    """Load the TRIBE model once per process. Raises ModelsNotDownloaded
    with an actionable message if the package or cache is missing.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    import os

    try:
        import importlib.util

        if importlib.util.find_spec("tribev2") is None:
            raise ModelsNotDownloaded(_DOWNLOAD_HINT)
    except ModelsNotDownloaded:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise ModelsNotDownloaded(f"{_DOWNLOAD_HINT}\n(import probe: {exc!r})")

    cache = _model_cache_dir()
    if not (cache.is_dir() and any(cache.iterdir())):
        raise ModelsNotDownloaded(_DOWNLOAD_HINT)

    # Apple-Silicon adaptation: MPS-fallback env + native whisper
    # transcription + spawn-safe DataLoader. No-op / unchanged on a GPU box.
    from . import native

    native.install()
    if os.environ.get("TRIBE_FAST_TEXT", "1") != "0":
        from . import fast_text

        fast_text.install()

    try:
        from tribev2.demo_utils import TribeModel
    except Exception as exc:
        raise ModelsNotDownloaded(
            f"{_DOWNLOAD_HINT}\n(tribev2 import failed: {exc!r})"
        )

    config_update = {"data.text_feature.model_name": "unsloth/Llama-3.2-3B"}
    device = native.resolve_device()
    # The neuralset extractor `device` is a pydantic Literal that does NOT
    # include "mps", so it can only be injected into config when it is a
    # value the Literal accepts (cuda/cpu/auto). For mps we leave config at
    # its default and retarget the loaded extractors below.
    if device in ("cuda", "cpu"):
        config_update.update(
            {
                "data.text_feature.device": device,
                "data.audio_feature.device": device,
            }
        )

    # Video counterpart of the text config_update above: upstream never
    # caps data.video_feature, so video runs vjepa2-vitg-fpc64-256 at 64
    # frames/clip full-res and OOM-panics Apple Silicon. Cap by RAM.
    # No-op on CUDA / ≥128 GiB / TRIBE_VIDEO_AUTO=0 (full fidelity there).
    config_update.update(_auto_video_config(device))

    # Model resolution can transiently fail ("model … does not exist",
    # network/HTTP, or concurrent HF-cache access from another process):
    # the cache is fine, the call just needs a retry. Retry with backoff;
    # only on a real (non-transient) error do we surface it immediately.
    model = None
    for attempt in range(4):
        try:
            model = TribeModel.from_pretrained(
                "facebook/tribev2",
                cache_folder=str(cache),
                config_update=config_update,
                device=device,
            )
            break
        except ModelsNotDownloaded:
            raise
        except Exception as exc:  # noqa: BLE001
            transient = any(
                s in repr(exc).lower()
                for s in (
                    "does not exist", "could not", "couldn't", "connection",
                    "timed out", "timeout", "rate limit", "429", "503",
                    "max retries", "temporarily", "readtimeout", "proxyerror",
                )
            )
            if not transient:
                raise
            if attempt == 3:
                raise ModelsNotDownloaded(
                    f"{_DOWNLOAD_HINT}\n(transient model-resolution failure "
                    f"after 4 attempts: {exc!r}. The cache is fine — this is "
                    f"usually flaky network or concurrent HF-cache access; "
                    f"just re-run.)"
                ) from exc
            time.sleep((3, 8, 20)[attempt])
    # Retarget lazily-loaded HF extractors at the resolved device (covers
    # mps, which cannot travel through the config Literal).
    native.patch_extractor_devices(model)

    # Disable the on-disk per-event extractor cache: distinct tracks have
    # ~unique words so the hit rate is ~0% across jobs and it just consumes
    # disk. exca keeps an in-process RAM cache regardless, so within a
    # single track we still dedupe.
    for attr in ("text_feature", "audio_feature"):
        extractor = getattr(model.data, attr, None)
        if extractor is None:
            continue
        infra = getattr(extractor, "infra", None)
        if infra is not None and hasattr(infra, "folder"):
            infra.folder = None
            try:
                from exca import base as _exca_base

                state = _exca_base._fast_state(infra)
                state.cache_dict = None
            except Exception:
                pass

    _MODEL = model
    return _MODEL


def _to_wav(audio_path: Path, tmp_dir: str) -> Path:
    import subprocess

    out = Path(tmp_dir) / (audio_path.stem + ".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", str(out)],
        check=True,
        capture_output=True,
    )
    return out


def _image_to_video(
    image_path: Path, tmp_dir: str, duration: float = 4.0, fps: int = 24
) -> Path:
    from moviepy import ImageClip

    video_path = Path(tmp_dir) / (image_path.stem + "_video.mp4")
    clip = ImageClip(str(image_path), duration=duration)
    clip.write_videofile(str(video_path), fps=fps, logger=None)
    return video_path


def predict(path: str | Path) -> "tuple":
    """Run the TRIBE pipeline on a media file, in memory.

    Pure compute: no network writes, no credentials, no server. Returns
    ``(preds, info)`` where ``preds`` is a ``(T, V)`` float array of
    predicted brain response (T ~= seconds, V vertices) and ``info`` is a
    metadata dict (sha256, shape, timings).

    Raises ``ModelsNotDownloaded`` (with an actionable message) if the
    upstream model is not installed/cached. Raises ``ValueError`` for an
    unsupported file type.
    """
    import tempfile

    import numpy as np

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported file extension: {ext or '<none>'} "
            f"(supported: {sorted(SUPPORTED_EXTS)})"
        )

    file_bytes = path.read_bytes()
    sha256 = hashlib.sha256(file_bytes).hexdigest()

    t_load = time.monotonic()
    model = get_model()
    load_seconds = time.monotonic() - t_load

    started = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / path.name
        input_path.write_bytes(file_bytes)

        t_ev = time.monotonic()
        if ext in IMAGE_EXTS:
            video_path = _image_to_video(input_path, tmp)
            events = model.get_events_dataframe(video_path=str(video_path))
        elif ext in VIDEO_EXTS:
            events = model.get_events_dataframe(video_path=str(input_path))
        else:  # audio
            audio_path = _to_wav(input_path, tmp) if ext != ".wav" else input_path
            events = model.get_events_dataframe(audio_path=str(audio_path))
        events_seconds = time.monotonic() - t_ev

        t_pr = time.monotonic()
        preds, _ = model.predict(events)
        predict_seconds = time.monotonic() - t_pr

    preds = np.asarray(preds)
    info = {
        "sha256": sha256,
        "filename": path.name,
        "modality": (
            "image"
            if ext in IMAGE_EXTS
            else "video"
            if ext in VIDEO_EXTS
            else "audio"
        ),
        "n_segments": int(preds.shape[0]),
        "n_vertices": int(preds.shape[1]) if preds.ndim > 1 else 0,
        "duration_seconds": int(preds.shape[0]),
        "model_load_seconds": round(load_seconds, 1),
        "events_seconds": round(events_seconds, 1),
        "predict_seconds": round(predict_seconds, 1),
        "processing_seconds": round(time.monotonic() - started, 1),
    }
    return preds, info
