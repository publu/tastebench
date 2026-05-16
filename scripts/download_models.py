#!/usr/bin/env python3
"""Download the TRIBE brain-model cache (~20 GB) used by the brain layer.

This pre-instantiates Meta's upstream ``tribev2`` model so its own
downloader fetches everything (the fMRI encoder + Llama-3.2-3B +
Whisper/wav2vec2 alignment weights) into the tastebench model cache.

Requirements:
  * `pip install -e ".[brain]"`  (installs the upstream `tribev2` package)
  * A Hugging Face account with **access to meta-llama / Llama-3.2** models,
    then `huggingface-cli login` (or set HF_TOKEN). Llama-3.2 is gated.

The craft layer needs NONE of this; only the brain layer does.

Cache location (override with MODEL_CACHE=/path):
  macOS / default : ~/.cache/tastebench/model-cache
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def cache_dir() -> Path:
    if os.environ.get("MODEL_CACHE"):
        return Path(os.environ["MODEL_CACHE"]).expanduser()
    return Path.home() / ".cache" / "tastebench" / "model-cache"


def main() -> int:
    cache = cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    print(f"[download_models] cache: {cache}")
    print("[download_models] this fetches ~20 GB and can take a while.\n")

    try:
        import importlib.util

        if importlib.util.find_spec("tribev2") is None:
            print(
                "ERROR: the upstream `tribev2` package is not installed.\n"
                "Install it:  pip install -e \".[brain]\"\n"
                "(or: pip install 'tribev2 @ "
                "git+https://github.com/facebookresearch/tribev2.git')",
                file=sys.stderr,
            )
            return 1
    except Exception as exc:  # pragma: no cover
        print(f"ERROR probing tribev2: {exc!r}", file=sys.stderr)
        return 1

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")):
        print(
            "NOTE: Llama-3.2 is a gated model. If the download 401s, run "
            "`huggingface-cli login` with an account that has Llama-3.2 "
            "access (https://huggingface.co/meta-llama), or set HF_TOKEN.\n"
        )

    # Reuse the project's device adaptation so this works on Apple Silicon
    # and CUDA alike (and never tries CUDA-only flags off a GPU box).
    try:
        from tastebench import native

        native.apply_runtime_env()
        # allow network for the one-time warm fetch
        os.environ["TRIBE_ALLOW_NET"] = "1"
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        device = native.resolve_device()
    except Exception:
        device = "cpu"

    print(f"[download_models] resolved device: {device}")
    from tribev2.demo_utils import TribeModel

    config_update = {"data.text_feature.model_name": "unsloth/Llama-3.2-3B"}
    print("[download_models] instantiating facebook/tribev2 "
          "(triggers the full weight download)...")
    TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=str(cache),
        config_update=config_update,
        device=device,
    )
    print(f"\n[download_models] done. Cache populated at: {cache}")
    print("The brain layer is now available "
          "(tastebench compare/optimize without --no-brain).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
