---
name: Bug report
about: Something behaves wrong or crashes
title: ''
labels: bug
assignees: ''
---

**What happened**
A clear description of the bug.

**Repro**
The exact command(s). Use the model-free path if it reproduces there:
```
tastebench compare a.wav b.wav --to demo.wav --no-brain
```

**Expected vs actual**
What you expected; what you got (paste the output / traceback).

**Environment**
- OS + arch (e.g. macOS 14, Apple Silicon / Linux CUDA):
- Python:
- tastebench commit:
- Brain layer: model present? (`--no-brain` to confirm it's craft-side)
