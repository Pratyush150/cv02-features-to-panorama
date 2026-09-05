"""Put ``src/`` on ``sys.path`` so the examples run without installing anything.

``python examples/01_corner_response.py`` should work in a fresh clone, before
``pip install -e .`` and without a ``PYTHONPATH`` incantation -- an example that
needs setup before it runs is an example most readers never see.

Every example starts with ``import _bootstrap``. That works because Python puts
the *script's own directory* on ``sys.path[0]``, so this file is importable from
any working directory. The tests do not use it: ``pyproject.toml`` sets pytest's
``pythonpath`` instead, which is the right mechanism there.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    # Appended, not inserted at position 0: if a real `feat` package is
    # installed in the environment, the installed one should win, so that
    # running the examples against an installed build tests the installed build.
    sys.path.append(str(_SRC))
