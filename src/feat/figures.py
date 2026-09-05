"""Matplotlib defaults and small helpers for the figures in ``docs/figures``.

These are teaching materials, so the palette is light: white paper, dark ink,
readable when printed and readable in a pull request diff on a phone. That is a
deliberate departure from a dark portfolio theme -- a figure whose message
depends on a dark background is a figure that fails the moment somebody pastes
it into a document.

The Agg backend is forced before ``pyplot`` is imported. Every example here
writes a PNG and none of them opens a window, and on a headless machine (CI, a
container, WSL without an X server) importing pyplot with an interactive
backend selected is an import-time crash that has nothing to do with the code
being demonstrated.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  -- must follow matplotlib.use

__all__ = ["FIGURES", "use_teaching_style", "show_gray", "show_bgr", "save", "annotate_points"]

# docs/figures, resolved from this file rather than from the working directory,
# so `python examples/07_ransac.py` writes to the same place as
# `cd examples && python 07_ransac.py`. A figure path built from os.getcwd() is
# a file that lands somewhere different depending on how you invoked it.
FIGURES = Path(__file__).resolve().parents[2] / "docs" / "figures"


def use_teaching_style() -> None:
    """Set the light, high-contrast defaults every figure in this repo uses."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#111111",
            "text.color": "#111111",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "figure.dpi": 110,
            "savefig.dpi": 110,
            "savefig.bbox": "tight",
        }
    )


def show_gray(ax, img: np.ndarray, title: str = "", **kwargs) -> None:
    """Draw a single-channel image with no axes and no interpolation blur.

    ``interpolation="nearest"`` because these figures are often looked at
    zoomed in to count pixels; matplotlib's default antialiasing invents
    intermediate values and would make a plateau look like a smooth peak, which
    is precisely the thing example 06 is trying to show.
    """
    ax.imshow(img, cmap=kwargs.pop("cmap", "gray"), interpolation="nearest", **kwargs)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def show_bgr(ax, img: np.ndarray, title: str = "") -> None:
    """Draw an OpenCV BGR image. The channel swap is the point of the function.

    OpenCV is BGR, matplotlib is RGB. Skip the conversion and every figure
    looks plausible and has its reds and blues exchanged -- a bug that survives
    review because nothing about the image looks broken.
    """
    rgb = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(rgb, cmap="gray" if rgb.ndim == 2 else None, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def annotate_points(ax, points: np.ndarray, colour: str = "#d62728", size: float = 12, **kwargs) -> None:
    """Scatter ``(N, 2)`` ``(x, y)`` points over an image axis."""
    if len(points) == 0:
        return
    ax.scatter(points[:, 0], points[:, 1], s=size, facecolors="none", edgecolors=colour,
               linewidths=0.9, **kwargs)


def save(fig, name: str) -> Path:
    """Write ``fig`` to ``docs/figures/<name>`` and return the path.

    The figure is closed afterwards. An example that builds several figures
    without closing them leaks them into matplotlib's global registry, and the
    warning about too many open figures arrives twenty figures later, in an
    unrelated file.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    return path
