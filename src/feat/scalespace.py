"""A small difference-of-Gaussian pyramid, built by hand.

Harris has a fixed window, so it has a fixed idea of how big a corner is.
Photograph the same corner from twice as far away and the structure no longer
fits the window, the response collapses, and matching fails -- example 02
measures exactly that. The fix is not a better response function. It is to stop
looking at the image once.

A **scale space** is the image blurred with a ladder of increasing sigma.
Blurring simulates stepping back: fine print disappears first, big shapes
survive. Subtracting neighbouring blur levels gives the **difference of
Gaussians**, a cheap stand-in for the Laplacian of Gaussian, and what survives
that subtraction is the detail that lived *between* those two blur levels. A
keypoint is a pixel that beats all 26 of its neighbours -- 8 in its own DoG
image, 9 above, 9 below -- and the scale at which it wins is its characteristic
size.

What is implemented here and what is not, stated plainly because the split is
the engineering point of this module: this builds the pyramid and finds the
extrema, which is SIFT stage 1. It does **not** do sub-pixel quadratic
refinement, edge rejection, orientation assignment or the 128-D descriptor
(stages 2 to 4). Those are in ``cv2.SIFT``, they are hard to get right, and
example 03 uses the library for them. Building stage 1 is what makes the phrase
"scale-invariant" mean something concrete; reimplementing stages 2 to 4 would
teach nothing further and ship a slower, buggier SIFT.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["Octave", "gaussian_octave", "dog_octave", "build_pyramid", "dog_extrema"]


@dataclass(frozen=True)
class Octave:
    """One octave: the blurred images, their sigmas, and the DoG between them."""

    blurred: list[np.ndarray]
    sigmas: list[float]
    dog: list[np.ndarray]  # len(blurred) - 1 images
    dog_sigmas: list[float]
    downsample: int  # 1, 2, 4, ... -- pixels here are this many pixels of the input


def gaussian_octave(
    gray: np.ndarray, n_intervals: int = 3, sigma0: float = 1.6
) -> tuple[list[np.ndarray], list[float]]:
    """Blur ``gray`` by a geometric ladder of sigmas covering one doubling.

    The sigmas are ``sigma0 * 2**(i/n_intervals)``. Geometric, not arithmetic,
    because scale is multiplicative: the visual difference between sigma 1 and
    2 is the same as between 4 and 8, and an arithmetic ladder would sample the
    coarse end far too finely and the fine end not at all.

    ``n_intervals + 3`` images are produced for ``n_intervals`` usable DoG
    layers. The bookkeeping: N images give N-1 DoG layers, and the extremum
    test needs a DoG layer above and below the one being searched, so N-3
    layers are searchable. Ask for 3 and you must build 6.
    """
    g = gray.astype(np.float32)
    sigmas = [sigma0 * (2.0 ** (i / n_intervals)) for i in range(n_intervals + 3)]
    # Each level is blurred from the *original*, not incrementally from the
    # previous level. Incremental blurring is faster and is what SIFT does, but
    # it accumulates the error of every earlier step, and getting the
    # incremental sigma right (sqrt(s_next**2 - s_prev**2), never s_next - s_prev)
    # is its own bug. Clarity wins here; the pyramid is not the hot loop.
    blurred = [cv2.GaussianBlur(g, (0, 0), s) for s in sigmas]
    return blurred, sigmas


def dog_octave(blurred: list[np.ndarray], sigmas: list[float]) -> tuple[list[np.ndarray], list[float]]:
    """Subtract each blurred image from the next, more-blurred one.

    Order matters and the convention here is ``next - current``, so a bright
    blob on a dark background gives a *negative* DoG response. Flip the
    subtraction and every extremum flips sign, which is harmless as long as you
    search for both maxima and minima -- which :func:`dog_extrema` does, for
    exactly this reason.
    """
    dog = [blurred[i + 1] - blurred[i] for i in range(len(blurred) - 1)]
    # The scale a DoG layer "is at" is conventionally the lower of the two
    # sigmas that made it. It is a convention, not a derivation, and it only
    # has to be consistent to make keypoint scales comparable across octaves.
    return dog, sigmas[:-1]


def build_pyramid(
    gray: np.ndarray, n_octaves: int = 4, n_intervals: int = 3, sigma0: float = 1.6
) -> list[Octave]:
    """Repeat :func:`gaussian_octave` on the image, halved each time.

    Halving the image is what makes the pyramid affordable. Doubling sigma in
    place costs quadratically more work per pixel; halving the image and
    reusing the same small sigmas costs a quarter as many pixels instead. The
    two give the same scale coverage, and only one of them is cheap.

    The loop stops early if an octave would fall below 32 pixels on a side --
    below that the Gaussian kernel is wider than the image and the border
    extrapolation, not the scene, decides the answer.
    """
    octaves: list[Octave] = []
    current = gray.astype(np.float32)
    for o in range(n_octaves):
        if min(current.shape[:2]) < 32:
            break
        blurred, sigmas = gaussian_octave(current, n_intervals, sigma0)
        dog, dog_sigmas = dog_octave(blurred, sigmas)
        octaves.append(
            Octave(blurred=blurred, sigmas=sigmas, dog=dog, dog_sigmas=dog_sigmas, downsample=2 ** o)
        )
        h, w = current.shape[:2]
        current = cv2.resize(current, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
    return octaves


def dog_extrema(octave: Octave, contrast: float = 3.0) -> np.ndarray:
    """Pixels that beat all 26 neighbours across three consecutive DoG layers.

    Returns an ``(N, 4)`` array of ``(x, y, sigma, dog_value)`` in the
    coordinates of the *original* image -- the ``downsample`` factor is undone
    here so keypoints from different octaves can be plotted on one picture.

    The 26 neighbours are 8 in the pixel's own DoG layer plus 9 in each of the
    layers above and below. It is done with three dilations and a max rather
    than a Python loop over 26 offsets, which is the difference between
    milliseconds and minutes.

    ``contrast`` is an absolute floor on ``|DoG|``, and it is the one parameter
    here that does not travel between images: DoG values scale with image
    contrast, so a value tuned on a bright scene finds nothing on a dim one.
    That is the same failure mode as an absolute Harris threshold, and the same
    fix applies -- scale it to the image if you take this beyond the examples.

    The ties caveat, stated because this module cannot claim to have fixed it:
    the test below is ``value == max_of_27``, so a plateau of exactly equal DoG
    values passes as a whole, exactly like the naive Harris suppression in
    :mod:`feat.harris`. On the blurred, float-valued DoG of a textured scene
    exact ties essentially never happen; on a synthetic image of flat blocks
    they can.
    """
    kernel = np.ones((3, 3), np.uint8)
    dilated = [cv2.dilate(d, kernel) for d in octave.dog]
    eroded = [cv2.erode(d, kernel) for d in octave.dog]

    found = []
    for i in range(1, len(octave.dog) - 1):  # skip first and last: no layer below/above
        cur = octave.dog[i]
        max27 = np.maximum(np.maximum(dilated[i - 1], dilated[i]), dilated[i + 1])
        min27 = np.minimum(np.minimum(eroded[i - 1], eroded[i]), eroded[i + 1])
        strong = np.abs(cur) > contrast
        mask = ((cur == max27) | (cur == min27)) & strong

        # Drop a one-pixel border: those pixels' 3x3 neighbourhoods were filled
        # in by cv2's border extrapolation, so their "26 neighbours" include
        # values the scene never contained.
        mask[:1, :] = mask[-1:, :] = mask[:, :1] = mask[:, -1:] = False

        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        scale = octave.downsample
        found.append(
            np.stack(
                [
                    xs * scale,
                    ys * scale,
                    np.full(len(xs), octave.dog_sigmas[i] * scale),
                    cur[ys, xs],
                ],
                axis=1,
            )
        )
    if not found:
        return np.zeros((0, 4), np.float64)
    return np.concatenate(found, axis=0)
