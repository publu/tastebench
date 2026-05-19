# demo — tastebench on a real released track

This is not a synthetic toy. `level-up-v4.mp3` is a finished, 320 kbps
track (2:38) — the author's own work, owned by the author, shipped here
solely as a test fixture so this demo is reproducible from a clone.

Below is tastebench's actual read of it. The whole thing — ASR, Llama-3.2
word embeddings, w2v-BERT audio features, the TRIBE fMRI-encoder forward
pass — ran **locally on a 32 GB M1 laptop, no GPU, in ~4m40s**:

```
whisperx ASR (transcript)        172.0s
Llama-3.2-3B word embeddings      45.2s
w2v-BERT audio features           10.3s
TRIBE forward + model load       ~31s
────────────────────────────────────────
total                          ~4m 39s   (158s of audio)
```

---

## The neural side — predicted 12-network response

TRIBE predicts the brain response the track evokes; tastebench reads it
out as a 12-network Cole-Anticevic signature. Bars are the per-network
mean, z-scored across the 12 networks (so 0 = network-average drive).
`rel` is split-half temporal reliability (1.0 = perfectly stable over
time, 0 = no stable signal).

```
 NETWORK                drive vs. its 12-network mean (z)        z      rel
 ────────────────────────────────────────────────────────────────────────
 Visual2              ████████████████████████████████████   +1.88   0.99
 Cingulo-Opercular    ██████████████████████████████         +1.49   0.98
 Visual1              ███████████████████████                +1.13   0.98
 Somatomotor          █████████████████████                  +1.05   0.99
 Auditory             ▏                                       -0.69   0.00
 Default-Mode         ▏                                       -0.69   0.00
 Dorsal-Attention     ▏                                       -0.69   0.00
 Frontoparietal       ▏                                       -0.69   0.00
 Language             ▏                                       -0.69   0.00
 Orbito-Affective     ▏                                       -0.69   0.00
 Posterior-Multimodal ▏                                       -0.69   0.00
 Ventral-Multimodal   ▏                                       -0.69   0.00
```

What the four that fired mean (the tool's own plain labels):

```
 Visual2             scene / motion engagement
 Cingulo-Opercular   attention-grabbing / effort to track
 Visual1             low-level visual salience
 Somatomotor         embodied / movement-evoking
```

**Read it honestly.** The predicted response concentrates in
motion/attention/embodiment networks, all with high temporal
reliability (~0.98) — a focused, driving signature, stable across the
whole track. The other eight sit at the network floor with zero
reliability: no differential, time-stable drive predicted there. This
is a *hypothesis view* — a predicted neural response pattern, **not** a
validated outcome — exactly as tastebench labels it everywhere.

## The craft side — concrete, model-free, fixable

These are measured straight off the audio (no model), and they're the
actionable layer:

```
 tempo                 123.05 BPM        tempo_stability   0.060  (very steady)
 time to hook            3.97 s          intro length      3.97 s
 hook density           16.67 /min       chorus lift       8.20 dB
 loopability            0.997  (≈perfect loop)
 dynamic range         11.05 dB          brightness     2036.5 Hz (centroid)
 spectral flatness      0.017  (tonal, not noisy)
 f0 range               1.88 oct         key stability     0.183
 voiced fraction        0.410
```

A hook that lands at ~4 s, ~17 hook moments/min, a near-perfect 0.997
loop, ~11 dB of dynamic range at a steady 123 BPM. Run `tastebench
glossary <term>` for the exact definition and units of any of these.

> This is a `profile` of a single reference, so every `spread` is
> `0.000` — a one-track taste is, by definition, perfectly consistent
> with itself. Point tastebench at *several* references and the spread
> becomes the consistency signal that weights every prescription.

---

## Reproduce it

The brain layer needs the ~20 GB TRIBE cache (see the main README). With
it present, point `MODEL_CACHE` at it and:

```bash
MODEL_CACHE=~/.cache/tribe-vast/model-cache \
  tastebench profile demo/level-up-v4.mp3 --format markdown
```

Drop the cache / add `--no-brain` for the instant craft-only read. To
see the *verdict* form (a draft graded against a taste), give it a
reference set: `tastebench vibe demo/level-up-v4.mp3 --like ref1 ref2`.
