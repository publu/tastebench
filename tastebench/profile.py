"""tastebench.profile — a set of references -> an aggregate taste signature.

The taste profile is the centroid of the reference signatures plus the
spread (consistency) of the set. It captures *what the user likes*: the
craft centroid (musician-actionable features) and the brain network/ROI
centroid, each with a per-key spread so we know how tight the taste is.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Iterable

from .signature import flatten, signature_for


def _agg(values):
    """[float...] -> (centroid, spread, n). Spread = stdev (0 if n<2)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return float(vals[0]), 0.0, 1
    return (
        float(statistics.fmean(vals)),
        float(statistics.pstdev(vals)),
        len(vals),
    )


def build_profile(
    refs: Iterable[str | Path], use_brain: bool = True
) -> dict:
    """Build a taste profile from reference media.

    Returns:
        {
          "n_refs": int,
          "refs": [{name, path, craft_ok, brain_ok}],
          "centroid": {key: float},          # craft + net.* keys
          "spread":   {key: float},          # per-key stdev (consistency)
          "n":        {key: int},            # how many refs had this key
          "consistency": float,              # 0..1, lower spread = tighter
          "signatures": [full per-ref signatures],
          "layers": {"craft": bool, "brain": bool},
        }
    """
    refs = [Path(r) for r in refs]
    sigs = []
    ref_meta = []
    for r in refs:
        s = signature_for(r, use_brain=use_brain)
        sigs.append(s)
        ref_meta.append(
            {
                "name": r.name,
                "path": str(r),
                "craft_ok": s["craft"].get("available", False),
                "brain_ok": s["brain"].get("available", False),
            }
        )

    flats = [flatten(s) for s in sigs]
    keys = sorted({k for f in flats for k in f})

    centroid, spread, ncount = {}, {}, {}
    for k in keys:
        c, sp, n = _agg([f.get(k) for f in flats])
        if c is None:
            continue
        centroid[k] = round(c, 6)
        spread[k] = round(sp, 6)
        ncount[k] = n

    # Consistency: 1 / (1 + mean normalized spread over craft features).
    # (Craft features are heterogeneously scaled; normalize by |centroid|.)
    craft_keys = [k for k in centroid if not k.startswith("net.")]
    norm_spreads = []
    for k in craft_keys:
        c = abs(centroid[k])
        if c > 1e-9:
            norm_spreads.append(spread[k] / c)
    consistency = (
        round(1.0 / (1.0 + statistics.fmean(norm_spreads)), 4)
        if norm_spreads
        else None
    )

    any_brain = any(s["brain"].get("available") for s in sigs)
    any_craft = any(s["craft"].get("available") for s in sigs)

    return {
        "n_refs": len(refs),
        "refs": ref_meta,
        "centroid": centroid,
        "spread": spread,
        "n": ncount,
        "consistency": consistency,
        "signatures": sigs,
        "layers": {"craft": any_craft, "brain": any_brain},
    }


def profile_summary(profile: dict) -> dict:
    """A compact, JSON-safe view of a profile (drops raw per-ref signatures)."""
    return {
        "n_refs": profile["n_refs"],
        "refs": profile["refs"],
        "centroid": profile["centroid"],
        "spread": profile["spread"],
        "n": profile["n"],
        "consistency": profile["consistency"],
        "layers": profile["layers"],
    }
