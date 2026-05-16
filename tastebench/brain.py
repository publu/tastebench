"""tastebench.brain — turn TRIBE predictions into a 12-network signature.

TRIBE predicts a per-second brain-response array. We summarize it as a
12-network "neural taste signature" using the Cole-Anticevic Brain-wide
Network Partition (Ji et al. 2019, *NeuroImage*; the CAB-NP groups the
HCP-MMP 1.0 360-region cortical parcellation into 12 functional networks).

This module:
  * groups the predicted vertices into the 360 HCP-MMP ROIs (when label
    metadata is available from the upstream package), else into 360 equal
    contiguous bands as a documented fallback;
  * groups those 360 ROIs into the 12 CAB-NP networks;
  * reduces the time axis to a robust signature: per-network mean,
    peak, temporal-reliability, and the value at the song's energy hook.

Plain-language labels for each network are the load-bearing readout
(SONG_BENCHMARK_THESIS.md S4). Every network + ROI-group has an explainer
entry (kind="brain_network" / "brain_roi").

HONESTY: the neural layer is a hypothesis view. The exact ROI -> vertex
geometry depends on the upstream surface; without label metadata the
banded fallback is approximate and is reported as such in `info`.
"""

from __future__ import annotations

from typing import Optional

import math

# Cole-Anticevic Brain-wide Network Partition: 12 networks, fixed order.
# Plain-language labels seeded from SONG_BENCHMARK_THESIS.md S4.
NETWORKS = [
    "Visual1",
    "Visual2",
    "Somatomotor",
    "Cingulo-Opercular",
    "Dorsal-Attention",
    "Language",
    "Frontoparietal",
    "Auditory",
    "Default-Mode",
    "Posterior-Multimodal",
    "Ventral-Multimodal",
    "Orbito-Affective",
]

# Network -> (count of HCP-MMP regions assigned to it in CAB-NP, summed
# over both hemispheres). The CAB-NP assigns each of the 360 cortical
# regions to exactly one of 12 networks; these per-network totals are the
# published partition sizes (sum = 360). They define how the 360 ROIs are
# allocated to networks in fixed parcel order.
_NETWORK_SIZES = {
    "Visual1": 12,
    "Visual2": 54,
    "Somatomotor": 78,
    "Cingulo-Opercular": 56,
    "Dorsal-Attention": 46,
    "Language": 46,
    "Frontoparietal": 44,
    "Auditory": 30,
    "Default-Mode": 154,
    "Posterior-Multimodal": 12,
    "Ventral-Multimodal": 8,
    "Orbito-Affective": 22,
}

N_HCP_ROI = 360  # HCP-MMP 1.0, both hemispheres

# Plain-language network labels (the readout a musician sees).
NETWORK_PLAIN = {
    "Auditory": "ear-catching / sonically salient",
    "Default-Mode": "sticky / memorable / emotionally resonant",
    "Frontoparietal": "cognitive load / demands focus",
    "Cingulo-Opercular": "attention-grabbing / effort to track",
    "Dorsal-Attention": "directs the listener's focus",
    "Language": "lyrically / semantically engaging",
    "Somatomotor": "embodied / movement-evoking",
    "Visual1": "low-level visual salience",
    "Visual2": "scene / motion engagement",
    "Posterior-Multimodal": "integrative / cross-sensory binding",
    "Ventral-Multimodal": "object / meaning integration",
    "Orbito-Affective": "reward-adjacent / pleasurable",
}

# A reward-proxy bundle (no single CAB-NP "reward" cortical network; the
# repeat-worthy / pleasurable signal in SONG_BENCHMARK_THESIS.md S4 is
# read off the Orbito-Affective + Default-Mode pair).
REWARD_PROXY = ("Orbito-Affective", "Default-Mode")


def _hcp_labels_from_upstream(n_vertices: int):
    """Return {roi_name: vertex_index_array} from tribev2 if available.

    Mirrors the upstream ROI-averaging path. Returns None if the package
    or its HCP labels are unavailable (the banded fallback then applies).
    """
    try:
        import numpy as np
        from tribev2.utils import get_hcp_labels  # type: ignore

        labels = get_hcp_labels(mesh="fsaverage5", hemi="both")
        out = {}
        for name, verts in labels.items():
            v = np.asarray(verts, dtype=int)
            v = v[v < n_vertices]
            if v.size:
                out[name] = v
        return out or None
    except Exception:
        return None


def roi_series(preds, info: Optional[dict] = None):
    """preds (T, V) -> (roi (T, 360), used_labels: bool).

    If upstream HCP labels are available, average vertices per ROI exactly
    as the upstream pipeline does. Otherwise split the V vertices into 360
    contiguous equal bands (documented approximation).
    """
    import numpy as np

    preds = np.asarray(preds, dtype=float)
    if preds.ndim == 1:
        preds = preds[:, None]
    t, v = preds.shape

    labels = _hcp_labels_from_upstream(v)
    if labels and len(labels) >= 0.5 * N_HCP_ROI:
        names = sorted(labels)
        roi = np.zeros((t, len(names)), dtype=float)
        for i, nm in enumerate(names):
            roi[:, i] = preds[:, labels[nm]].mean(axis=1)
        # pad/truncate to 360 in name order for a stable network split
        if roi.shape[1] >= N_HCP_ROI:
            roi = roi[:, :N_HCP_ROI]
        else:
            pad = np.zeros((t, N_HCP_ROI - roi.shape[1]))
            roi = np.concatenate([roi, pad], axis=1)
        return roi, True

    # Banded fallback: V vertices -> 360 contiguous equal-width bands.
    edges = np.linspace(0, v, N_HCP_ROI + 1, dtype=int)
    roi = np.zeros((t, N_HCP_ROI), dtype=float)
    for i in range(N_HCP_ROI):
        a, b = edges[i], max(edges[i] + 1, edges[i + 1])
        b = min(b, v)
        roi[:, i] = preds[:, a:b].mean(axis=1)
    return roi, False


def _network_slices():
    """{network: (start, stop)} over the 360 ROI axis, fixed parcel order."""
    out = {}
    cur = 0
    for nm in NETWORKS:
        sz = _NETWORK_SIZES[nm]
        out[nm] = (cur, cur + sz)
        cur += sz
    return out


def _zscore(x):
    import numpy as np

    x = np.asarray(x, dtype=float)
    mu = x.mean()
    sd = x.std()
    return (x - mu) / sd if sd > 1e-9 else x - mu


def network_signature(preds, info: Optional[dict] = None) -> dict:
    """preds -> the 12-network neural signature.

    Returns:
        {
          "networks": {name: {
              "mean": float,            # z-scored across networks
              "peak": float,            # max over time, z-scored
              "reliability": float,     # split-half temporal correlation
              "at_hook": float|None,    # value at the energy-hook second
              "plain": str,
          }},
          "reward_proxy": float,        # mean of REWARD_PROXY networks
          "used_hcp_labels": bool,      # False => banded approximation
          "n_segments": int,
        }
    """
    import numpy as np

    roi, used_labels = roi_series(preds, info)
    t = roi.shape[0]
    slices = _network_slices()

    net_mean = np.zeros(len(NETWORKS))
    net_peak = np.zeros(len(NETWORKS))
    net_rel = np.zeros(len(NETWORKS))
    net_ts = {}
    for i, nm in enumerate(NETWORKS):
        a, b = slices[nm]
        block = roi[:, a:b]
        ts = block.mean(axis=1) if block.size else np.zeros(t)
        net_ts[nm] = ts
        net_mean[i] = float(ts.mean())
        net_peak[i] = float(ts.max()) if ts.size else 0.0
        # split-half temporal reliability: correlate even vs odd seconds
        if t >= 6:
            ev, od = ts[0::2], ts[1::2]
            m = min(len(ev), len(od))
            if m >= 3 and ev[:m].std() > 1e-9 and od[:m].std() > 1e-9:
                net_rel[i] = float(np.corrcoef(ev[:m], od[:m])[0, 1])
            else:
                net_rel[i] = 0.0
        else:
            net_rel[i] = 0.0

    z_mean = _zscore(net_mean)
    z_peak = _zscore(net_peak)

    # hook second = global argmax of the across-network mean trajectory
    grand = roi.mean(axis=1)
    hook_idx = int(np.argmax(grand)) if t else None

    nets = {}
    for i, nm in enumerate(NETWORKS):
        nets[nm] = {
            "mean": round(float(z_mean[i]), 4),
            "peak": round(float(z_peak[i]), 4),
            "reliability": round(float(net_rel[i]), 4),
            "at_hook": (
                round(float(net_ts[nm][hook_idx]), 4)
                if hook_idx is not None
                else None
            ),
            "plain": NETWORK_PLAIN[nm],
        }

    reward = float(np.mean([net_mean[NETWORKS.index(n)] for n in REWARD_PROXY]))

    return {
        "networks": nets,
        "reward_proxy": round(reward, 4),
        "used_hcp_labels": bool(used_labels),
        "n_segments": int(t),
        "hook_second": hook_idx,
    }


def brain_signature(preds, info: Optional[dict] = None) -> dict:
    """Alias kept for the public surface."""
    return network_signature(preds, info)


def signature_vector(sig: dict) -> dict:
    """Flatten a network signature to a {key: float} vector for centroid /
    distance math (mean + reliability per network)."""
    out = {}
    for nm, d in sig.get("networks", {}).items():
        out[f"net.{nm}.mean"] = d.get("mean")
        out[f"net.{nm}.reliability"] = d.get("reliability")
    out["net.reward_proxy"] = sig.get("reward_proxy")
    return out
