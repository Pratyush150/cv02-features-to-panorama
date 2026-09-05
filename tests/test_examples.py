"""The examples must at least be valid Python and stay in step with the README.

Running all eight takes a couple of minutes, which is too slow for the default
suite, so this file does the cheap checks that catch the failures that actually
happen: a syntax error introduced while editing, an example that stops writing
the figure the README references, or a figure filename that drifts out of sync
with the documentation.

To run the examples themselves:  python examples/01_corner_response.py  (and so on)
"""

from __future__ import annotations

import py_compile
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted(p for p in (ROOT / "examples").glob("*.py") if not p.name.startswith("_"))


def test_there_are_eight_numbered_examples():
    assert len(EXAMPLES) == 8
    assert [p.name[:2] for p in EXAMPLES] == [f"{i:02d}" for i in range(1, 9)]


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_compiles(path: Path):
    py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_declares_its_figure_and_the_figure_exists(path: Path):
    """Each example's docstring names the figure it writes, and that figure is
    committed. A figure the README shows but the code no longer produces is the
    documentation equivalent of dead code."""
    text = path.read_text()
    declared = re.search(r"^Figure: docs/figures/(\S+)$", text, re.MULTILINE)
    assert declared, f"{path.name} does not declare its figure in the docstring"
    name = declared.group(1)
    # Either quote style: several of these live inside an f-string, where the
    # filename is written with single quotes.
    assert f'"{name}"' in text or f"'{name}'" in text, (
        f"{path.name} declares {name} but never saves it"
    )
    assert (ROOT / "docs" / "figures" / name).exists(), f"{name} is not committed"


def test_readme_references_every_committed_figure():
    readme = (ROOT / "README.md").read_text()
    for figure in sorted((ROOT / "docs" / "figures").glob("*.png")):
        assert figure.name in readme, f"{figure.name} is committed but not shown in the README"


def test_no_ai_assistant_names_anywhere():
    """A house rule for this repository, enforced rather than remembered."""
    banned = ("claude", "anthropic", "copilot", "chatgpt", "co-authored-by")
    this_file = Path(__file__).resolve()
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        # The file that lists the banned words necessarily contains them.
        if path.resolve() == this_file:
            continue
        if path.suffix not in {".py", ".md", ".toml", ".txt", ".yml", ".yaml", ""}:
            continue
        lowered = path.read_text(errors="ignore").lower()
        for word in banned:
            assert word not in lowered, f"{path.relative_to(ROOT)} contains {word!r}"
