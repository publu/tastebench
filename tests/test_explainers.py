"""Explainer-coverage guard (issue #6).

The readout leans on plain-language explainers: every `compare` /
`optimize` line attaches one, `tastebench glossary` browses them, and
`report --llm` embeds the whole dictionary. If a brain network,
reward-proxy member, or emitted craft metric key is added *without* an
explainer, the user silently gets a blank where the meaning should be.

These tests fail loudly the moment that happens. Model-free: they only
read `explainers.json` and the static key lists — no audio, no model.
"""

from __future__ import annotations

from tastebench import explainers
from tastebench.brain import NETWORKS, REWARD_PROXY, signature_vector
from tastebench.features.structural import ACTIONABLE


def test_every_network_has_an_explainer():
    missing = [n for n in NETWORKS if explainers.get_explainer(n) is None]
    assert not missing, f"brain networks with no explainer entry: {missing}"


def test_reward_proxy_has_an_explainer():
    assert (
        explainers.get_explainer("reward_proxy")
        or explainers.get_explainer("net.reward_proxy")
    ), "no explainer for the reward-proxy readout"
    for member in REWARD_PROXY:
        assert explainers.get_explainer(member) is not None, (
            f"reward-proxy member {member!r} has no explainer"
        )


def test_every_actionable_craft_feature_has_an_explainer():
    missing = [k for k in ACTIONABLE if explainers.get_explainer(k) is None]
    assert not missing, f"actionable craft features with no explainer: {missing}"


def test_every_emitted_signature_key_resolves():
    """Every flat key the brain layer actually emits must resolve to an
    explainer (this is what the report rows look up)."""
    fake_sig = {
        "networks": {n: {"mean": 0.0, "reliability": 0.0} for n in NETWORKS},
        "reward_proxy": 0.0,
    }
    emitted = signature_vector(fake_sig)
    assert emitted, "signature_vector emitted nothing — wiring changed"
    missing = [k for k in emitted if explainers.get_explainer(k) is None]
    assert not missing, f"emitted signature keys with no explainer: {missing}"
