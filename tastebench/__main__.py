"""Enable `python -m tastebench` (no install / no entry-point needed).

`python -m tastebench`            → the folder worker (default)
`python -m tastebench worker DIR` → the worker on an explicit dir
`python -m tastebench compare …`  → any CLI verb still works
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
