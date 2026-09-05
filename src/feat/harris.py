"""The structure tensor, the Harris response, and turning a map into a list.

This module implements corner detection from the gradients up, and then proves
the implementation against ``cv2.cornerHarris``. The proof is the lesson: after
it you know the library call is your arithmetic with a scale factor on it, and
you are free to call the library forever after.

The one-paragraph version of the theory, because the code below assumes it:

Take a window around a pixel and nudge it. The sum of squared differences
between the nudged and un-nudged patch, to first order, is a quadratic form
``[u v] M [u v]^T`` where M is the 2x2 **structure tensor** built from three
windowed sums of gradient products. M's two eigenvalues say how steeply the
patch changes along its two principal directions. Both small is flat, one large
is an edge, both large is a corner. Harris avoids the per-pixel eigendecomposition
by scoring ``R = det(M) - k*trace(M)**2``, using ``det = l1*l2`` and
``trace = l1 + l2``.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "BLOCK",
    "KSIZE",
    "K",
    "structure_tensor",
    "eigenvalues",
    "harris_response",
    "opencv_response_scale",
    "classify_field",
    "peaks_naive_nms",
    "peaks_component_nms",
]

# The three knobs, named once so every example and test uses the same ones and
# the numbers in the README are comparable across files.
BLOCK = 5   # side of the summation window: how far around a pixel M accumulates
KSIZE = 3   # Sobel aperture: how wide a stencil estimates each gradient
K = 0.04    # the k in R = det - k*trace^2


def structure_tensor(
    gray: np.ndarray, block: int = BLOCK, ksize: int = KSIZE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the three windowed sums ``(Sxx, Syy, Sxy)`` that make up M.

    ``block`` and ``ksize`` are two *different* windows and conflating them is
    the most common misreading of this function's signature. ``ksize`` is the
    Sobel aperture -- how many pixels are consulted to estimate one gradient.
    ``block`` is the summation window -- how far around the pixel those
    gradients are accumulated into M. Raising ``ksize`` smooths the gradients;
    raising ``block`` changes the size of structure the detector responds to.
    Someone hoping to find larger corners usually raises the wrong one.
    """
    # float32 in, float32 out. On uint8 input OpenCV folds an extra 1/255 into
    # its Sobel scaling, which moves R by 255**4 (about 4.2e9) with no error and
    # no warning -- so a threshold tuned on float32 input finds nothing at all
    # on the same image as uint8.
    g = gray.astype(np.float32)

    ix = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=ksize)
    iy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=ksize)

    # normalize=False is load-bearing. A normalised box filter divides by the
    # window area, which turns the *sums* in M into *means*. The classification
    # (the sign of R) survives that, but every absolute number changes by
    # block**2 per sum, so R changes by block**4 -- and the agreement with
    # cv2.cornerHarris that this module exists to demonstrate would fail.
    def win(a: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(a, -1, (block, block), normalize=False)

    return win(ix * ix), win(iy * iy), win(ix * iy)


def eigenvalues(sxx: np.ndarray, syy: np.ndarray, sxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both eigenvalues of the 2x2 symmetric M, in closed form.

    For ``[[a, b], [b, c]]`` the eigenvalues are
    ``(a + c)/2 +- sqrt(((a - c)/2)**2 + b**2)``. No ``np.linalg.eigvalsh`` per
    pixel: that would be two million small decompositions on a 1080p frame,
    which is exactly the cost Harris's determinant-and-trace trick avoids.

    The discriminant is clamped at zero before the square root. It is
    non-negative in exact arithmetic for any symmetric matrix, but float32
    subtraction of two nearly equal large numbers can land it at -1e-9, and
    ``sqrt`` of that is a NaN that then propagates silently into every plot.
    """
    half_sum = 0.5 * (sxx + syy)
    half_diff = 0.5 * (sxx - syy)
    disc = np.sqrt(np.maximum(half_diff * half_diff + sxy * sxy, 0.0))
    return half_sum + disc, half_sum - disc  # lambda1 >= lambda2 by construction


def harris_response(
    gray: np.ndarray, block: int = BLOCK, ksize: int = KSIZE, k: float = K
) -> np.ndarray:
    """``R = det(M) - k * trace(M)**2``, computed from our own structure tensor.

    Signs, and what each one means:

    * ``R`` large and positive -- both eigenvalues large -- a corner.
    * ``R`` large and negative -- one eigenvalue dominates -- an edge.
    * ``R`` near zero -- both eigenvalues small -- a flat region.

    The negative case is worth dwelling on, because it is the one people get
    wrong. A 45-degree edge has a large ``Sxx`` *and* a large ``Syy``: read only
    the diagonal of M and it looks exactly like a corner. The cross term
    ``Sxy`` is what kills it -- the gradient points the same way at every pixel
    in the window, so ``det(M)`` collapses to zero and R is pure penalty.
    Corner-ness cannot be read off the diagonal.
    """
    sxx, syy, sxy = structure_tensor(gray, block, ksize)
    det = sxx * syy - sxy * sxy
    trace = sxx + syy
    return det - k * trace * trace


def opencv_response_scale(block: int = BLOCK, ksize: int = KSIZE) -> float:
    """The exact factor between :func:`harris_response` and ``cv2.cornerHarris``.

    OpenCV pre-scales its Sobel output by ``s = 1 / (2**(ksize-1) * block)``
    before squaring anything. Every entry of M is a product of two such scaled
    gradients, so each entry carries ``s**2``; ``det`` and ``trace**2`` are each
    a product of two entries, so both carry ``s**4``, and the whole response
    scales by ``s**4`` and nothing else.

    This is why an absolute Harris threshold is meaningless on its own. The
    number depends on the bit depth, the Sobel scaling and the window size, so
    ``R > 1e6`` tuned on one setup finds every pixel or no pixel on the next.
    Threshold as a fraction of that image's own ``R.max()``, which is exactly
    what ``goodFeaturesToTrack``'s ``qualityLevel`` does.
    """
    s = 1.0 / ((1 << (ksize - 1)) * block)
    return s ** 4


def classify_field(
    lam1: np.ndarray, lam2: np.ndarray, k: float = K, rel: float = 0.01
) -> np.ndarray:
    """Label every pixel 0 = flat, 1 = edge, 2 = corner, from the eigenvalues.

    "Flat" is decided first and by magnitude, not by the sign of R: on a truly
    flat patch both eigenvalues are tiny and R is a tiny number of *either*
    sign, so classifying by sign alone paints sensor noise as a field of edges
    and corners. ``rel`` is a fraction of ``lam1.max()`` for this image, for the
    reason in :func:`opencv_response_scale` -- absolute cut-offs do not travel.
    """
    floor = rel * float(lam1.max())
    out = np.zeros(lam1.shape, np.uint8)
    active = lam1 > floor
    r = lam1 * lam2 - k * (lam1 + lam2) ** 2  # R written in eigenvalue form
    out[active & (r <= 0)] = 1
    out[active & (r > 0)] = 2
    return out


def peaks_naive_nms(response: np.ndarray, rel_thresh: float = 0.01, radius: int = 3) -> np.ndarray:
    """Threshold, then the ``R == dilate(R)`` non-maximum suppression. **Buggy.**

    This is the version in most tutorials, and it is kept here because seeing
    it fail is the point of example 06. Grey dilation replaces each pixel with
    the maximum over its neighbourhood, so ``R == dilate(R)`` is meant to mean
    "this pixel is the largest in its neighbourhood".

    It means something slightly different: "this pixel is *equal to* the largest
    in its neighbourhood". When several adjacent pixels share the exact same
    maximum -- a plateau, which synthetic and rendered images produce constantly
    because their edges are exact -- every pixel on the plateau passes. You get
    a clump where you wanted a point.

    The failure does not reproduce on photographs, because sensor noise breaks
    the ties. That is what makes it dangerous: it passes on your real data and
    fails on the clean test image, so it looks like the *test* is wrong.

    Returns an ``(N, 2)`` array of ``(x, y)`` integer peak coordinates.
    """
    ksz = 2 * radius + 1
    dilated = cv2.dilate(response, np.ones((ksz, ksz), np.uint8))
    mask = (response == dilated) & (response > rel_thresh * response.max())
    ys, xs = np.nonzero(mask)
    return np.stack([xs, ys], axis=1)


def peaks_component_nms(
    response: np.ndarray, rel_thresh: float = 0.01, radius: int = 3
) -> np.ndarray:
    """The same thing, with plateaus collapsed to one point each. **Correct.**

    Label the surviving mask into connected components and keep one point per
    component. The point kept is the component's own argmax rather than its
    centroid: a centroid on an L-shaped plateau can land on a pixel that is not
    in the component at all.

    Cost: one extra pass over a sparse binary mask, which is nothing next to the
    detector itself. Benefit, measured on the default checkerboard in
    ``examples/06_five_shapes_of_bug.py``: the naive version returns hundreds of
    points where the true answer is 49.
    """
    ksz = 2 * radius + 1
    dilated = cv2.dilate(response, np.ones((ksz, ksz), np.uint8))
    mask = ((response == dilated) & (response > rel_thresh * response.max())).astype(np.uint8)

    n_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    peaks = []
    for label in range(1, n_labels):  # 0 is the background
        ys, xs = np.nonzero(labels == label)
        best = int(np.argmax(response[ys, xs]))
        peaks.append((int(xs[best]), int(ys[best])))
    return np.array(peaks, dtype=int).reshape(-1, 2)
