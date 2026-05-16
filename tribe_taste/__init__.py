"""tribe-taste — learn the taste signature of media you admire, then see how
your own demo diverges and what to change.

Public surface:

    from tribe_taste.engine import predict          # brain inference
    from tribe_taste.features import librosa_report  # craft features
    from tribe_taste.brain import brain_signature    # preds -> 12-network
    from tribe_taste.profile import build_profile    # references -> taste
    from tribe_taste.compare import compare          # demo vs taste
    from tribe_taste.optimize import optimize        # ranked edits
    from tribe_taste.explainers import get_explainer # the glossary

The brain layer is multimodal (audio / video / image). The craft layer is
audio-only and no-ops gracefully on non-audio input.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
