"""Deep per-stage timing for the native pipeline.

Provenance: written for an internal TRIBE inference deployment; pure
instrumentation (monkeypatched timers around upstream methods). No server /
credential / storage code. Carried into tastebench as-is.


Gives a full breakdown so we always know where the wall time goes:

  model_load | whisperx_subprocess | transforms | text_prepare(Llama) |
  audio_prepare(w2v-bert) | brain_forward | events | predict

Implemented as low-risk monkeypatches around upstream methods, gated by
``TRIBE_TIMING`` (default on). Every measured stage is printed live
(flush) so a Monitor sees it as it happens, and a consolidated summary
is emitted at the end of each song.
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict

_ACC: dict[str, float] = defaultdict(float)
_CNT: dict[str, int] = defaultdict(int)
_INSTALLED = False


def reset() -> None:
    _ACC.clear()
    _CNT.clear()


def add(stage: str, seconds: float, *, quiet: bool = False) -> None:
    _ACC[stage] += seconds
    _CNT[stage] += 1
    if not quiet:
        print(f"[timing] {stage}={seconds:.1f}s", flush=True)


def report(extra: dict | None = None) -> dict:
    rows = dict(sorted(_ACC.items(), key=lambda kv: -kv[1]))
    line = "  ".join(
        f"{k}={v:.1f}s" + (f"x{_CNT[k]}" if _CNT[k] > 1 else "") for k, v in rows.items()
    )
    print(f"[timing] SUMMARY  {line}", flush=True)
    out = {f"t_{k}": round(v, 1) for k, v in rows.items()}
    if extra:
        out.update(extra)
    return out


def install() -> None:
    """Idempotent. Timestamped logging + stage monkeypatches."""
    global _INSTALLED
    if _INSTALLED or os.environ.get("TRIBE_TIMING", "1") == "0":
        return

    # Timestamped logs: turns neuralset/tribev2 INFO lines into a timeline.
    root = logging.getLogger()
    if not any(getattr(h, "_tribe_ts", False) for h in root.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
        h._tribe_ts = True  # type: ignore[attr-defined]
        root.addHandler(h)
        root.setLevel(logging.INFO)

    # Per-extractor prepare (text=Llama, audio=w2v-bert) — this is where
    # feature-model load + inference happens inside model.predict().
    try:
        from neuralset.extractors.base import BaseExtractor

        if not getattr(BaseExtractor, "_tribe_timed", False):
            _orig_prepare = BaseExtractor.prepare

            def _timed_prepare(self, obj):  # noqa: ANN001
                t0 = time.monotonic()
                try:
                    return _orig_prepare(self, obj)
                finally:
                    add(f"prepare[{type(self).__name__}]", time.monotonic() - t0)

            BaseExtractor.prepare = _timed_prepare
            BaseExtractor._tribe_timed = True
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).warning("[timing] prepare hook failed: %r", e)

    # Brain model forward — accumulate total + call count.
    try:
        from tribev2.model import FmriEncoderModel

        if not getattr(FmriEncoderModel, "_tribe_timed", False):
            _orig_fwd = FmriEncoderModel.forward

            def _timed_fwd(self, *a, **k):  # noqa: ANN001
                t0 = time.monotonic()
                try:
                    return _orig_fwd(self, *a, **k)
                finally:
                    _ACC["brain_forward"] += time.monotonic() - t0
                    _CNT["brain_forward"] += 1

            FmriEncoderModel.forward = _timed_fwd
            FmriEncoderModel._tribe_timed = True
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).warning("[timing] forward hook failed: %r", e)

    _INSTALLED = True
    print("[timing] deep stage timing installed", flush=True)
