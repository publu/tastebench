# examples

This repo **never ships audio** (all media is `.gitignore`d). Instead,
`make_examples.py` *synthesizes* lawful, public-domain clips at runtime —
pure generated tones with structure, carrying no third-party rights:

```bash
python examples/make_examples.py
```

It writes (locally, untracked):

| file | role |
|---|---|
| `ref_a.wav` | a reference: fast hook, tight loop, bright |
| `ref_b.wav` | another reference in the same taste |
| `demo.wav`  | a demo that diverges: slow hook, key wander, darker |

Then:

```bash
tastebench compare examples/ref_a.wav examples/ref_b.wav \
    --to examples/demo.wav --no-brain
tastebench optimize examples/demo.wav \
    --toward examples/ref_a.wav examples/ref_b.wav
```

## Using your own media

Put any audio (`.wav .mp3 .flac .m4a .ogg`), video (`.mp4 .mov …`), or
image (`.png .jpg …`) here — it stays local (gitignored). The craft layer
handles audio; the brain layer (with the model installed) additionally
handles video/image. `examples/_work/` is a scratch dir for analysis
outputs and is also ignored.
```
