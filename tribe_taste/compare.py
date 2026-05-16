"""tribe_taste.compare — a demo vs a taste profile.

Reports, in plain language:
  * per-craft-feature deltas (demo vs taste centroid), each with its
    explainer and a song-bones / production-state tag;
  * per-brain-network and per-ROI-group deltas;
  * an overall normalized distance to the taste;
  * the nearest single reference.

All distances are spread-normalized: a delta is measured in units of the
reference set's own consistency on that feature, so a feature the taste is
tight on counts more than one it is loose on.
"""

from __future__ import annotations

from pathlib import Path

from .explainers import get_explainer
from .features.structural import ACTIONABLE
from .metric import zdelta
from .profile import build_profile
from .signature import flatten, signature_for

# craft features that are production-state rather than song-bones
_PRODUCTION_STATE = {"brightness", "dynamic_range_db", "flatness"}


def _tag(feature: str) -> str:
    if feature.startswith("net."):
        return "neural"
    return "production" if feature in _PRODUCTION_STATE else "song-bones"


def _direction(feature: str, delta: float) -> str:
    if abs(delta) < 1e-9:
        return "on target"
    return "above" if delta > 0 else "below"


def compare(
    demo: str | Path,
    profile: dict | list | tuple,
    use_brain: bool = True,
) -> dict:
    """Compare one demo against a taste profile.

    `profile` may be a prebuilt profile dict or an iterable of reference
    paths (a profile is then built on the fly).
    """
    if not isinstance(profile, dict):
        profile = build_profile(profile, use_brain=use_brain)

    demo_sig = signature_for(demo, use_brain=use_brain)
    demo_flat = flatten(demo_sig)
    centroid = profile["centroid"]
    spread = profile["spread"]

    lines = []
    sq = 0.0
    nterms = 0
    for key, c in centroid.items():
        dv = demo_flat.get(key)
        if dv is None:
            continue
        delta = dv - c
        z = zdelta(key, delta, spread.get(key, 0.0), c)
        sq += z * z
        nterms += 1
        ex = get_explainer(key)
        lines.append(
            {
                "term": key,
                "kind": _tag(key),
                "demo": round(float(dv), 4),
                "taste": round(float(c), 4),
                "delta": round(float(delta), 4),
                "spread_norm": round(float(z), 3),
                "direction": _direction(key, delta),
                "actionable": key in ACTIONABLE,
                "plain": (ex or {}).get("plain", ""),
                "explainer": ex,
            }
        )

    # rank by how far off (spread-normalized), biggest divergence first
    lines.sort(key=lambda r: -abs(r["spread_norm"]))
    overall = round((sq / nterms) ** 0.5, 4) if nterms else None

    nearest = _nearest_reference(demo_flat, profile)

    craft_lines = [r for r in lines if not r["term"].startswith("net.")]
    brain_lines = [r for r in lines if r["term"].startswith("net.")]

    return {
        "demo": {
            "name": Path(demo).name,
            "path": str(demo),
            "craft_ok": demo_sig["craft"].get("available", False),
            "brain_ok": demo_sig["brain"].get("available", False),
            "brain_note": demo_sig["brain"].get("hint"),
        },
        "profile": {
            "n_refs": profile["n_refs"],
            "consistency": profile["consistency"],
            "layers": profile["layers"],
        },
        "overall_distance": overall,
        "verdict": _verdict(overall),
        "nearest_reference": nearest,
        "craft_deltas": craft_lines,
        "brain_deltas": brain_lines,
        "n_terms_compared": nterms,
    }


def _verdict(distance):
    if distance is None:
        return "no comparable features"
    if distance < 0.75:
        return "very close to this taste"
    if distance < 1.5:
        return "broadly in this taste, with clear levers"
    if distance < 3.0:
        return "noticeably off this taste"
    return "far from this taste"


def _nearest_reference(demo_flat: dict, profile: dict) -> dict | None:
    """Robustly-normalized nearest single reference signature."""
    spread = profile["spread"]
    centroid = profile["centroid"]
    best = None
    for s, meta in zip(profile["signatures"], profile["refs"]):
        rf = flatten(s)
        common = [k for k in demo_flat if k in rf and demo_flat[k] is not None
                  and rf[k] is not None]
        if not common:
            continue
        d = 0.0
        for k in common:
            d += zdelta(
                k, demo_flat[k] - rf[k], spread.get(k, 0.0),
                centroid.get(k, rf[k]),
            ) ** 2
        dist = (d / len(common)) ** 0.5
        if best is None or dist < best[0]:
            best = (dist, meta["name"], len(common))
    if best is None:
        return None
    return {
        "name": best[1],
        "distance": round(best[0], 4),
        "n_features": best[2],
    }
