"""Drop-in faster bit-equivalent override for neuralset.extractors.text.HuggingFaceText.

Provenance: written for an internal TRIBE inference deployment to speed up
upstream ``tribev2``'s Llama word-embedding extraction without changing its
numerics. No server / credential / storage code. Carried into tastebench
as-is. Upstream model: facebookresearch/tribev2 (see ATTRIBUTION.md).


Speeds up TRIBE's Llama-3.2-3B word embedding extraction by ~15-40x for the
audio path, where consecutive Word events share a sentence context.

Wins:
  - load model in bf16 (was fp32 by default)              -- 2-3x
  - one forward pass per UNIQUE sentence (was per-word)   -- 5-10x
  - sdpa or flash_attention_2 kernel (was eager default)  -- 1.3-2x

Correctness: Llama is a causal decoder. The hidden state at token position N
depends only on tokens [0..N], so a word's per-token hidden states are
identical whether we run the full sentence or just the prefix-up-to-and-
including-the-word (which is what the original does). We use the same
n_prefix / n_target slice math as the original to pick the word's tokens
out of the full-sentence forward pass.

Install: call install() once at server boot, before TribeModel.from_pretrained.
"""

import itertools
import logging
import os

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def _pick_device(requested: str) -> str:
    override = os.environ.get("TRIBE_DEVICE", "auto").lower()
    if override != "auto":
        return override
    if requested and requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            return "mps" if torch.backends.mps.is_available() else "cpu"
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_dtype(device: str) -> torch.dtype:
    override = os.environ.get("TRIBE_TEXT_DTYPE", "auto").lower()
    if override in {"fp32", "float32"}:
        return torch.float32
    if override in {"fp16", "float16"}:
        return torch.float16
    if override in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def _load_model_fast(self, **kwargs):
    """Replacement for HuggingFaceText._load_model. bf16 + sdpa/flash attn.

    NOTE: We deliberately load in float32 then .to(bfloat16) afterwards.
    Passing torch_dtype=bfloat16 to from_pretrained leaks the global default
    dtype, which breaks downstream extractors (e.g. event.read() returns bf16
    wav, which the SeamlessM4T feature extractor cannot np.asarray).
    """
    from transformers import AutoModel as Model

    if "t5" in self.model_name or "bert" in self.model_name:
        from transformers import AutoModelForTextEncoding as Model
    elif "Phi-3" in self.model_name:
        from transformers import AutoModelForCausalLM as Model
    elif "Llama-3.2-11B-Vision" in self.model_name:
        from transformers import MllamaForConditionalGeneration as Model

    load_kwargs = {}
    device = _pick_device("auto" if self.device == "accelerate" else self.device)

    if self.device == "accelerate":
        load_kwargs["device_map"] = "auto"

    prev_default_dtype = torch.get_default_dtype()
    model = None
    try:
        attn_impls = ("flash_attention_2", "sdpa") if device == "cuda" else ("sdpa",)
        for attn in attn_impls:
            try:
                model = Model.from_pretrained(
                    self.model_name,
                    attn_implementation=attn,
                    **load_kwargs,
                )
                LOGGER.info(
                    "[fast_text] loaded %s attn=%s (will cast to bf16)",
                    self.model_name,
                    attn,
                )
                break
            except Exception as e:
                LOGGER.warning(
                    "[fast_text] attn=%s failed: %s: %s",
                    attn,
                    type(e).__name__,
                    e,
                )
        if model is None:
            model = Model.from_pretrained(self.model_name, **load_kwargs)
            LOGGER.warning("[fast_text] fell back to default attn impl")
        # Cast weights to bfloat16 explicitly. Avoid passing torch_dtype to
        # from_pretrained, which can leak the default dtype globally.
        dtype = _pick_dtype(device)
        if dtype is not torch.float32:
            model = model.to(dtype)
    finally:
        # Belt + suspenders: in case any inner path tampered with default
        # dtype, restore it. SeamlessM4T's np.asarray(wav, dtype=float32)
        # blows up on bf16 tensors otherwise.
        torch.set_default_dtype(prev_default_dtype)

    if not self.pretrained:
        rawmodel = Model.from_config(model.config)
        with torch.no_grad():
            for p1, p2 in itertools.zip_longest(
                model.parameters(), rawmodel.parameters()
            ):
                p1.data = p2.to(p1)
    elif self.pretrained == "part-reversal":
        from neuralset.extractors.text import part_reversal

        with torch.no_grad():
            for p in model.parameters():
                part_reversal(p)

    if self.device != "accelerate":
        model.to(device)
        self.device = device
    model.eval()
    return model


def _forward_sentence_hidden(self, sentence_text, device):
    """One forward for a single sentence -> (n_layers, n_tokens, dim)."""
    inputs = self.tokenizer(
        sentence_text, add_special_tokens=False,
        return_tensors="pt", truncation=True,
    ).to(device)
    outputs = self.model(**inputs, output_hidden_states=True)
    if "hidden_states" in outputs:
        states = outputs.hidden_states
    else:
        states = outputs.encoder_hidden_states + outputs.decoder_hidden_states
    return torch.stack(list(states), dim=0).squeeze(1)


def _batch_sentence_hidden(self, sentences, device, batch_size):
    """Forward UNIQUE sentences in padded batches.

    Llama is a causal decoder, so with RIGHT padding + an attention mask
    the per-token hidden states of the real (non-pad) tokens are identical
    to running each sentence alone (a token only attends to <= its
    position; pads sit after and are masked) — modulo batched-matmul
    reduction-order float noise. The win: the ~6 GB of weights are
    streamed from memory ONCE per batch instead of once per sentence
    (this workload is memory-bandwidth bound on MPS).
    """
    pad_id = (
        self.tokenizer.pad_token_id
        if self.tokenizer.pad_token_id is not None
        else (self.tokenizer.eos_token_id or 0)
    )
    cache = {}
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i:i + batch_size]
        toks = [
            self.tokenizer(
                s, add_special_tokens=False, truncation=True,
                return_tensors="pt",
            )["input_ids"][0]
            for s in chunk
        ]
        lens = [int(t.shape[0]) for t in toks]
        lmax = max(lens)
        ids = torch.full((len(chunk), lmax), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), lmax), dtype=torch.long)
        for j, t in enumerate(toks):
            ids[j, : lens[j]] = t
            attn[j, : lens[j]] = 1
        outputs = self.model(
            input_ids=ids.to(device),
            attention_mask=attn.to(device),
            output_hidden_states=True,
        )
        if "hidden_states" in outputs:
            states = outputs.hidden_states
        else:
            states = outputs.encoder_hidden_states + outputs.decoder_hidden_states
        stacked = torch.stack(list(states), dim=0)  # (n_layers, B, lmax, dim)
        for j, s in enumerate(chunk):
            # slice to the real token count -> exactly the single-seq shape
            cache[s] = stacked[:, j, : lens[j], :].contiguous()
        del outputs, states, stacked
    return cache


def _get_data_fast(self, events):
    """Replacement for HuggingFaceText._get_data.

    Forward pass per UNIQUE sentence (optionally batched across sentences,
    TRIBE_FAST_TEXT_BATCH=1), then per-word slice extraction. Slice math is
    byte-identical to the original; batching only changes how the unique
    sentences are fed through Llama. TRIBE_FAST_TEXT_BATCH=0 restores the
    exact original per-sentence path (instant revert if corr regresses).
    """
    import os

    from exca.utils import environment_variables
    from tqdm import tqdm

    if not events:
        return

    device = _pick_device("auto" if self.device == "accelerate" else self.device)
    batch_on = os.environ.get("TRIBE_FAST_TEXT_BATCH", "1") != "0"
    batch_size = int(os.environ.get("TRIBE_FAST_TEXT_BATCH_SIZE", "16"))

    with torch.no_grad():
        with environment_variables(TOKENIZERS_PARALLELISM="false"):
            sentences = []
            for event in events:
                context = getattr(event, "context", "") or ""
                if not context or not context.strip():
                    raise ValueError(
                        f"Empty context for target_word {event.text!r}"
                    )
                s = getattr(event, "sentence", "") or context
                if s not in sentences:
                    sentences.append(s)

            if batch_on and len(sentences) > 1:
                hidden_by_sentence = _batch_sentence_hidden(
                    self, sentences, device, batch_size
                )
            else:
                hidden_by_sentence = None

            iterable = events
            if len(events) > 1:
                iterable = tqdm(events, desc="word embed (fast)", mininterval=2.0)

            cached_sentence = None
            cached_hidden = None
            for event in iterable:
                target_word = event.text
                context = getattr(event, "context", "") or ""
                sentence_text = getattr(event, "sentence", "") or context

                if hidden_by_sentence is not None:
                    cached_hidden = hidden_by_sentence[sentence_text]
                elif sentence_text != cached_sentence:
                    cached_hidden = _forward_sentence_hidden(
                        self, sentence_text, device
                    )
                    cached_sentence = sentence_text

                # Same slice math as original: n_prefix = tokens of context
                # without the target word; n_target = remaining tokens.
                prefix = (
                    context[: -len(target_word)].rstrip()
                    if len(target_word) > 0
                    else context
                )
                n_prefix = (
                    len(self.tokenizer.encode(prefix, add_special_tokens=False))
                    if prefix
                    else 0
                )
                n_context = len(
                    self.tokenizer.encode(context, add_special_tokens=False)
                )
                n_target = max(1, n_context - n_prefix)

                n_full = cached_hidden.shape[1]
                start = max(0, min(n_prefix, n_full - 1))
                end = min(start + n_target, n_full)
                if end <= start:
                    end = min(start + 1, n_full)

                word_state = cached_hidden[:, start:end]
                word_state = self._aggregate_tokens(word_state)
                out = word_state.detach().cpu().numpy()
                if not self.cache_all_layers and self.cache_n_layers is None:
                    out = self._aggregate_layers(out)
                if np.isnan(out).any():
                    raise ValueError(
                        f"NaN in output for target_word {target_word!r}"
                    )
                yield out


def install():
    """Monkey-patch neuralset.extractors.text.HuggingFaceText. Idempotent.

    For _get_data we MUST preserve the @infra.apply caching decorator that
    wraps the original method. Without it, DataLoader workers re-run feature
    extraction (cache miss), which tries to load Llama inside a forked
    subprocess and fails with the CUDA fork error. Replacing the inner
    function on the MapInfraMethod object keeps the cache and swaps just the
    cache-miss path to our fast version.
    """
    from neuralset.extractors.text import HuggingFaceText

    if getattr(HuggingFaceText, "_fast_installed", False):
        LOGGER.info("[fast_text] already installed, skipping")
        return

    HuggingFaceText._load_model = _load_model_fast

    prop = HuggingFaceText.__dict__.get("_get_data")
    replaced_inner = False
    if isinstance(prop, property) and prop.fget is not None:
        imethod = prop.fget
        if hasattr(imethod, "method"):
            # exca's _factory uses method.__name__ to look the method up
            # on the class; make our replacement masquerade as _get_data.
            _get_data_fast.__name__ = "_get_data"
            _get_data_fast.__qualname__ = imethod.method.__qualname__
            _get_data_fast.__module__ = imethod.method.__module__
            imethod.method = _get_data_fast
            replaced_inner = True
            LOGGER.info(
                "[fast_text] replaced inner _get_data on MapInfraMethod "
                "(decorator caching preserved)"
            )
    if not replaced_inner:
        HuggingFaceText._get_data = _get_data_fast
        LOGGER.warning(
            "[fast_text] direct replacement of _get_data (no decorator "
            "cache — DataLoader workers will fail on cache miss)"
        )

    HuggingFaceText._fast_installed = True
    LOGGER.info(
        "[fast_text] monkey-patched neuralset.extractors.text.HuggingFaceText"
    )
