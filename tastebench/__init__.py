"""tastebench — learn the taste signature of media you admire, then see how
your own demo diverges and what to change.

Public surface:

    from tastebench.engine import predict          # brain inference
    from tastebench.features import librosa_report  # craft features
    from tastebench.brain import brain_signature    # preds -> 12-network
    from tastebench.profile import build_profile    # references -> taste
    from tastebench.compare import compare          # demo vs taste
    from tastebench.optimize import optimize        # ranked edits
    from tastebench.explainers import get_explainer # the glossary

The brain layer is multimodal (audio / video / image). The craft layer is
audio-only and no-ops gracefully on non-audio input.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
