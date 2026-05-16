"""tribe_taste.signature — one media file -> a combined taste signature.

A signature has two layers:

  * craft : the audio-only librosa-derived musician-actionable features
            (no model needed). None on non-audio.
  * brain : the 12-network neural signature from TRIBE predictions
            (needs the model). None / skipped if the model is absent.

`signature_for(path)` runs whatever is available and never hard-fails on a
missing model: the craft layer alone is a valid signature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def craft_layer(path: str | Path) -> dict:
    """Audio-only craft features for one file (model-free)."""
    from .features import librosa_report
    from .features.structural import craft_vector

    report = librosa_report.extract(str(path))
    vec = craft_vector(report)
    return {
        "available": not bool(report.get("_error")),
        "error": report.get("_error"),
        "features": vec,
        "scalars": report.get("scalars", {}) if not report.get("_error") else {},
    }


def brain_layer(path: str | Path) -> dict:
    """12-network neural signature for one file (needs the TRIBE model).

    Never raises for a missing model: returns ``available: False`` with the
    actionable hint so the caller can degrade to craft-only.
    """
    from .brain import network_signature
    from .engine import ModelsNotDownloaded, predict

    try:
        preds, info = predict(path)
    except ModelsNotDownloaded as exc:
        return {"available": False, "reason": "models_not_downloaded", "hint": str(exc)}
    except (ValueError, FileNotFoundError) as exc:
        return {"available": False, "reason": "unsupported", "hint": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive
        return {"available": False, "reason": "error", "hint": repr(exc)}

    sig = network_signature(preds, info)
    return {"available": True, "info": info, "signature": sig}


def signature_for(
    path: str | Path, use_brain: bool = True
) -> dict:
    """Combined signature for one file. Always returns; degrades gracefully."""
    path = Path(path)
    out = {
        "path": str(path),
        "name": path.name,
        "craft": craft_layer(path),
        "brain": (
            brain_layer(path)
            if use_brain
            else {"available": False, "reason": "disabled"}
        ),
    }
    return out


def flatten(sig: dict) -> dict:
    """Combined signature -> flat {key: float|None} vector for centroid /
    distance math. Craft keys are bare names; brain keys are ``net.*``.
    """
    from .brain import signature_vector

    out: dict[str, Optional[float]] = {}
    craft = sig.get("craft", {})
    if craft.get("available"):
        for k, v in (craft.get("features") or {}).items():
            out[k] = v
    brain = sig.get("brain", {})
    if brain.get("available"):
        out.update(signature_vector(brain.get("signature", {})))
    return out
