"""tribe_taste.optimize — counterfactual edit search toward a taste.

The taste model here is a transparent distance: the demo's craft signature
vs the reference centroid, each feature spread-normalized by the reference
set's own consistency (same metric as `compare`). To produce a producer's
edit list we perturb ONE musician-actionable craft feature at a time toward
the centroid (within a musically valid step), recompute the overall craft
distance, and rank the edits by predicted distance reduction.

Every edit carries:
  * predicted delta (how much the overall craft distance drops),
  * a confidence label (song-bones edits high; production-state medium;
    edits where the reference set is itself inconsistent are downgraded),
  * the proposed concrete change (from -> to),
  * its explainer (the matching kind="edit" dictionary entry).

This is intentionally a craft-layer search: it is model-free, fast, and
the features are the ones a musician can actually act on (the spec is
explicit that the prescriptive layer must be musician-actionable and
honestly confidence-labeled).
"""

from __future__ import annotations

from pathlib import Path

from .explainers import get_explainer
from .features.structural import ACTIONABLE
from .metric import zdelta
from .profile import build_profile
from .signature import flatten, signature_for

# actionable craft feature -> (edit term, base confidence)
_EDIT_FOR = {
    "intro_length": ("shorten_intro", "high"),
    "time_to_hook": ("shorten_intro", "high"),
    "chorus_lift_db": ("raise_chorus_lift", "high"),
    "loopability": ("commit_tonal_center", "high"),
    "key_stability": ("commit_tonal_center", "high"),
    "tempo": ("shift_tempo_toward_lane", "high"),
    "tempo_stability": ("tighten_tempo", "medium"),
    "brightness": ("match_brightness", "medium"),
    "dynamic_range_db": ("open_dynamics", "medium"),
    "hook_density_per_min": ("adjust_hook_density", "medium"),
}

# musically-valid single-edit step caps (absolute units per feature).
# An edit moves the demo toward the centroid by at most this much, so we
# never propose an implausible one-shot change.
_MAX_STEP = {
    "intro_length": 12.0,       # seconds
    "time_to_hook": 15.0,       # seconds
    "chorus_lift_db": 6.0,      # dB
    "loopability": 0.20,        # ratio
    "key_stability": 0.25,      # ratio
    "tempo": 12.0,              # BPM
    "tempo_stability": 0.15,    # ratio
    "brightness": 600.0,        # Hz
    "dynamic_range_db": 6.0,    # dB
    "hook_density_per_min": 2.0,
}


def _craft_distance(flat: dict, centroid: dict, spread: dict) -> float:
    sq = 0.0
    n = 0
    for k, c in centroid.items():
        if k.startswith("net."):
            continue
        v = flat.get(k)
        if v is None:
            continue
        z = zdelta(k, v - c, spread.get(k, 0.0), c)
        sq += z * z
        n += 1
    return (sq / n) ** 0.5 if n else 0.0


def _confidence(feature: str, base: str, profile: dict) -> str:
    """Downgrade confidence when the reference set is itself inconsistent
    on this feature (a loose taste = a less trustworthy prescription)."""
    c = abs(profile["centroid"].get(feature, 0.0))
    sp = profile["spread"].get(feature, 0.0)
    if c > 1e-9 and sp / c > 0.6:  # reference set very inconsistent here
        order = {"high": "medium", "medium": "low", "low": "low"}
        return order.get(base, "low")
    return base


def optimize(
    demo: str | Path,
    toward,
    use_brain: bool = False,
    top: int = 8,
) -> dict:
    """Rank musician-actionable edits that move the demo toward a taste.

    `toward` is a prebuilt profile dict or an iterable of reference paths.
    `use_brain` defaults False: the edit search is a craft-layer search
    (model-free, fast, musician-actionable).
    """
    if not isinstance(toward, dict):
        toward = build_profile(toward, use_brain=use_brain)

    demo_sig = signature_for(demo, use_brain=use_brain)
    flat = flatten(demo_sig)
    centroid = toward["centroid"]
    spread = toward["spread"]

    if not demo_sig["craft"].get("available"):
        return {
            "demo": {"name": Path(demo).name, "path": str(demo)},
            "edits": [],
            "note": (
                "Craft layer unavailable for this input "
                f"({demo_sig['craft'].get('error')}). The edit search is "
                "craft-based and applies to audio only."
            ),
        }

    base_dist = _craft_distance(flat, centroid, spread)
    edits = []
    for feat in ACTIONABLE:
        if feat not in centroid or flat.get(feat) is None:
            continue
        cur = float(flat[feat])
        target = float(centroid[feat])
        gap = target - cur
        if abs(gap) < 1e-9:
            continue
        step = max(-_MAX_STEP[feat], min(_MAX_STEP[feat], gap))
        proposed = dict(flat)
        proposed[feat] = cur + step
        new_dist = _craft_distance(proposed, centroid, spread)
        gain = base_dist - new_dist
        if gain <= 1e-6:
            continue  # only propose edits that actually help

        edit_term, base_conf = _EDIT_FOR.get(feat, (feat, "medium"))
        conf = _confidence(feat, base_conf, toward)
        ex = get_explainer(f"edit.{edit_term}") or get_explainer(feat)
        edits.append(
            {
                "edit": edit_term,
                "feature": feat,
                "from": round(cur, 4),
                "to": round(cur + step, 4),
                "toward_taste": round(target, 4),
                "predicted_distance_delta": -round(gain, 4),
                "predicted_gain": round(gain, 4),
                "confidence": conf,
                "plain": (ex or {}).get("plain", ""),
                "how_to_act": (ex or {}).get("how_to_act", ""),
                "explainer": ex,
                "caveat": (
                    "Hypothesis to A/B, not a guarantee. Predicted change is "
                    "in the transparent craft-distance metric, not measured "
                    "outcomes."
                ),
            }
        )

    edits.sort(key=lambda e: -e["predicted_gain"])
    edits = edits[: max(0, top)]

    return {
        "demo": {
            "name": Path(demo).name,
            "path": str(demo),
        },
        "profile": {
            "n_refs": toward["n_refs"],
            "consistency": toward["consistency"],
        },
        "base_craft_distance": round(base_dist, 4),
        "edits": edits,
        "note": (
            "Edits perturb one musician-actionable craft feature toward the "
            "taste centroid within a musically-valid step, ranked by modeled "
            "distance reduction. Confidence is lower where the reference set "
            "is itself inconsistent on that feature."
        ),
    }
