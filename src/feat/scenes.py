"""Synthetic test scenes.

Every image this repository uses is generated here. That is a deliberate
choice, not a convenience: a stitcher that fails on two photographs leaves you
unable to say whether the code is wrong or the photographs were. When the
scene is synthesised we know the ground-truth homography exactly, so "did it
work?" becomes a number instead of a squint at a seam.

Three properties every generator here holds to:

* **Deterministic.** Every function takes a ``seed`` and uses its own
  ``np.random.default_rng``. The global ``np.random`` state is never touched,
  so importing this module cannot change the output of anything else, and two
  runs a week apart produce byte-identical images. The tests depend on that.
* **Offline.** No downloads, no files on disk, no camera.
* **Known answer.** Where a scene has a right answer -- the corner count of a
  checkerboard, the homography relating two views -- the function returns it
  alongside the pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "ViewPair",
    "Checkerboard",
    "textured_wall",
    "brick_wall",
    "checkerboard",
    "gradient_lit_page",
    "two_views",
    "scale_pair",
    "paste_intruder",
]


@dataclass(frozen=True)
class ViewPair:
    """Two synthetic photographs of one planar scene, plus the exact answer.

    ``H_true`` maps a point in ``img_b`` to the corresponding point in
    ``img_a``. The direction is stated here once and never guessed again:
    ``findHomography(src, dst)`` returns the map ``src -> dst``, so throughout
    this repo B is the source and A is the base frame. Getting this backwards
    produces a homography that is *also* a valid homography -- the inverse --
    with an equally healthy inlier count and no error message anywhere, which
    is why it is written down rather than remembered.
    """

    img_a: np.ndarray
    img_b: np.ndarray
    H_true: np.ndarray  # 3x3 float64, normalised so H_true[2, 2] == 1


@dataclass(frozen=True)
class Checkerboard:
    """A checkerboard image together with the answer it is used to check."""

    image: np.ndarray
    n_interior: int  # X-junctions where four squares meet: (squares - 1) ** 2
    interior: tuple[int, int, int, int]  # (x0, y0, x1, y1) box holding exactly those


def _rng(seed: int) -> np.random.Generator:
    # A fresh generator per call rather than a module-level one. A module-level
    # generator would make a scene's content depend on how many other scenes
    # were built first, which turns "the test passed" into "the test passed in
    # this import order".
    return np.random.default_rng(seed)


def textured_wall(h: int = 720, w: int = 1080, seed: int = 7) -> np.ndarray:
    """A flat wall covered in non-repeating clutter: the *easy* scene.

    Rectangles, discs and scratches at random sizes and colours. The point is
    that no two patches look alike, so every descriptor is distinctive and the
    matcher has an easy time. It is the control case for :func:`brick_wall`
    below -- if a matching trick helps here and hurts there, you have learned
    something about the trick that one scene alone could not tell you.

    Everything drawn lives on one plane, which is what makes a homography the
    exactly correct model for two views of it (rather than an approximation
    that happens to work).
    """
    rng = _rng(seed)
    img = np.full((h, w, 3), 38, np.uint8)

    for _ in range(180):  # "posters": large blocks, the coarse structure
        x, y = int(rng.integers(0, w - 100)), int(rng.integers(0, h - 100))
        a, b = int(rng.integers(22, 96)), int(rng.integers(22, 96))
        colour = tuple(int(c) for c in rng.integers(55, 255, 3))
        cv2.rectangle(img, (x, y), (x + a, y + b), colour, -1)

    for _ in range(140):  # discs: curved edges, so corners are not all axis-aligned
        centre = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        colour = tuple(int(c) for c in rng.integers(55, 255, 3))
        cv2.circle(img, centre, int(rng.integers(5, 24)), colour, -1)

    for _ in range(90):  # scratches: thin lines crossing everything, for extra corners
        p = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        q = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        colour = tuple(int(c) for c in rng.integers(55, 255, 3))
        cv2.line(img, p, q, colour, int(rng.integers(1, 4)))

    # A 3x3 blur before returning. Without it every edge is a perfect step and
    # the gradient at a corner is a single-pixel spike, which flatters both the
    # detector and the sub-pixel refinement in a way no lens ever will.
    return cv2.GaussianBlur(img, (3, 3), 0)


def brick_wall(h: int = 720, w: int = 1080, seed: int = 11) -> np.ndarray:
    """A wall of near-identical bricks: the scene that makes matching lie.

    Every brick is the same size, the same colour give or take a few grey
    levels, and offset by half a brick on alternate rows. That is the point.
    A descriptor computed on brick (3, 7) is nearly identical to one computed
    on brick (3, 8), so the nearest neighbour and the runner-up are separated
    by sensor noise and nothing else.

    The per-brick jitter is small on purpose. Make it large and each brick
    becomes distinctive, the ambiguity disappears, and the demo built on this
    scene quietly stops demonstrating anything.
    """
    rng = _rng(seed)
    bw, bh = 96, 40  # brick width and height, in pixels
    mortar = 6

    # The mortar is pale and the bricks are dark, and that is checked in
    # *luminance*, not in the BGR triple. A brick of (58, 74, 148) against a
    # grey mortar of 96 looks like a wall on screen and converts to almost
    # exactly the same grey -- every detector here works on the grey image, so
    # the wall would be invisible to the thing the scene exists to test.
    img = np.full((h, w, 3), 172, np.uint8)  # mortar, luminance ~172
    for row, y in enumerate(range(0, h, bh + mortar)):
        # Alternate rows shift by half a brick. This is what makes the pattern
        # doubly periodic rather than a simple grid, and it is exactly the
        # geometry of a real running-bond wall.
        x_off = -(bw // 2) if row % 2 else 0
        for x in range(x_off, w, bw + mortar):
            shade = int(rng.integers(-9, 10))  # a few grey levels, no more
            colour = (52 + shade, 66 + shade, 132 + shade)  # luminance ~84
            cv2.rectangle(img, (x, y), (x + bw, y + bh), colour, -1)

    # Sensor noise. Without it, bricks that differ by zero grey levels give
    # descriptors that are bit-identical, and the ratio test hits d2 == 0,
    # which is a different (and rarer) failure than the one being shown here.
    noise = rng.normal(0, 2.5, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0)


def checkerboard(
    squares: int = 8, square_px: int = 40, margin: int = 30, blur: bool = False
) -> Checkerboard:
    """A checkerboard, and the exact number of interior corners it contains.

    For an ``n x n`` board the interior corners -- the X-junctions where four
    squares meet -- number ``(n - 1)**2``: 49 for the default 8x8. That is the
    ground truth the non-maximum-suppression demo is scored against, and it is
    why the detector story here starts on a checkerboard rather than a photo.
    ``interior`` is the ``(x0, y0, x1, y1)`` box that contains exactly those 49
    and excludes the corners of the pattern's outer border, which are real
    corners but not the ones being counted.

    The margin matters: run the board to the image edge and the outermost
    corners sit under Sobel's border extrapolation, which invents gradients
    that were never in the scene.

    ``blur`` defaults to **False**, and that is the interesting parameter. An
    unblurred checkerboard has exact plateaus -- runs of pixels sharing the
    identical maximum response -- which is what makes the naive
    ``R == dilate(R)`` suppression in :mod:`feat.harris` fail loudly. A 3x3
    blur, or any sensor noise, breaks those ties and hides the bug completely.
    Pass ``blur=True`` when you want the realistic image; leave it False when
    you want the one that can fail.
    """
    side = squares * square_px
    img = np.full((side + 2 * margin, side + 2 * margin), 30, np.uint8)  # not 0: no sensor floor is
    for r in range(squares):
        for c in range(squares):
            if (r + c) % 2 == 0:
                y0, x0 = margin + r * square_px, margin + c * square_px
                img[y0:y0 + square_px, x0:x0 + square_px] = 225
    if blur:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    pad = 5  # keep the border X-junctions' response blobs out of the interior box
    return Checkerboard(
        image=img,
        n_interior=(squares - 1) ** 2,
        interior=(margin + pad, margin + pad, margin + side - pad, margin + side - pad),
    )


def gradient_lit_page(h: int = 600, w: int = 800, seed: int = 3) -> np.ndarray:
    """A page of text-like blocks under a strong left-to-right light gradient.

    Brightness runs from roughly 60 on one side to roughly 235 on the other,
    which is a factor of four in illumination across one image. This is the
    scene that tests the claim "a descriptor is illumination invariant": the
    same paragraph block is dim on the left and bright on the right, so a
    descriptor that survives is surviving something real.

    It also has a second use. Corner *detection* is not illumination invariant
    -- the Harris response is fourth order in the gradient, so the dim side of
    this page scores about 4**4 = 256 times lower than the bright side for the
    same physical structure. An absolute threshold on R finds corners only on
    the right-hand half.
    """
    rng = _rng(seed)
    img = np.full((h, w), 240, np.uint8)
    y = 40
    while y < h - 40:
        x = 50
        line_h = int(rng.integers(9, 15))
        while x < w - 60:
            word = int(rng.integers(24, 90))
            if x + word > w - 50:
                break
            cv2.rectangle(img, (x, y), (x + word, y + line_h), 35, -1)
            x += word + int(rng.integers(9, 20))
        y += line_h + int(rng.integers(10, 18))

    # The gradient is multiplicative, not additive, because that is what a real
    # light source does: it scales the reflected radiance. An additive ramp
    # would leave every gradient magnitude in the image unchanged, and the
    # detector would notice nothing -- a demo that cannot fail.
    ramp = np.linspace(0.25, 1.0, w, dtype=np.float32)[None, :]
    lit = np.clip(img.astype(np.float32) * ramp, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(lit, (3, 3), 0)


def two_views(
    scene: np.ndarray,
    out_size: tuple[int, int] = (640, 480),
    quad_a: np.ndarray | None = None,
    quad_b: np.ndarray | None = None,
) -> ViewPair:
    """Photograph one planar ``scene`` twice, from two known viewpoints.

    Each "photograph" is the perspective image of a quadrilateral patch of the
    scene, mapped onto an ``out_size`` sensor. Because we choose both patches,
    we know both scene-to-sensor homographies exactly, and therefore we know
    their composition:

        H_true = H_a @ inv(H_b)      # B's pixels -> the scene -> A's pixels

    Read that right to left, in the order a point travels: undo view B to land
    back on the wall, then apply view A's projection. Composing homographies is
    matrix multiplication in travel order, and reversing it is a silent bug --
    the result is still a homography, just the wrong one.
    """
    w, h = out_size
    sensor = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    if quad_a is None:
        quad_a = np.float32([[60, 70], [60 + 620, 70], [60 + 620, 70 + 470], [60, 70 + 470]])
    if quad_b is None:
        # Deliberately not a translation. The corners are pulled unevenly so
        # the bottom row of H_true is non-zero: this pair genuinely needs eight
        # degrees of freedom, and a similarity or affine fit will visibly fail
        # on it. A pair related by a pure shift would let a wrong model look
        # right, which is the "demo with no control" bug in a different suit.
        quad_b = np.float32([[420, 40], [420 + 690, 40 + 74], [420 + 665, 40 + 545], [400, 40 + 500]])

    H_a = cv2.getPerspectiveTransform(quad_a, sensor)  # scene -> photo A
    H_b = cv2.getPerspectiveTransform(quad_b, sensor)  # scene -> photo B
    img_a = cv2.warpPerspective(scene, H_a, out_size)
    img_b = cv2.warpPerspective(scene, H_b, out_size)

    H_true = H_a @ np.linalg.inv(H_b)
    H_true = H_true / H_true[2, 2]  # pin the scale: H and 5H are the same map
    return ViewPair(img_a=img_a, img_b=img_b, H_true=H_true)


def scale_pair(scene: np.ndarray, factor: float = 0.4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The same scene at two scales, and the exact homography between them.

    Returns ``(full, scaled, H)`` where ``H`` maps a point in ``scaled`` to the
    corresponding point in ``full``. A pure scaling is a homography with a
    zero bottom row and equal diagonal entries, so it is the *easiest* possible
    geometric relationship -- which is the point. When Harris fails to match
    across this pair, it is not failing because the geometry was hard.

    ``INTER_AREA`` for the downscale, because it averages the pixels it is
    discarding. ``INTER_LINEAR`` samples them, which aliases fine texture into
    noise and would let us blame the resampler for the detector's problem.
    """
    h, w = scene.shape[:2]
    small = cv2.resize(scene, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)
    inv = 1.0 / factor
    H = np.array([[inv, 0.0, 0.0], [0.0, inv, 0.0], [0.0, 0.0, 1.0]], np.float64)
    return scene, small, H


def paste_intruder(img: np.ndarray, x: int, y: int, size: int = 72) -> np.ndarray:
    """Paste a small, vividly distinctive object into a copy of ``img``.

    Pasted at *different scene positions* in the two views, this is the
    synthetic equivalent of somebody walking through the shot between two
    exposures. It is the outlier source that RANSAC has to reject in example
    07, and it is designed to be *unambiguous*: bright, high-contrast, unique.
    That matters, because it means the ratio test will happily pass every match
    on it. Only geometry can tell you a confident, unambiguous match is wrong,
    and that is the entire argument for running RANSAC after the ratio test.

    Returns a copy -- ``img[y:y+s, x:x+s] = ...`` on the caller's array would
    mutate the scene every later call reuses, and the bug would show up as a
    figure that changes depending on which examples ran first.
    """
    out = img.copy()
    if out.ndim == 2:
        patch = np.full((size, size), 250, np.uint8)
        cv2.circle(patch, (size // 2, size // 2), size // 3, 30, -1)
        cv2.rectangle(patch, (8, 8), (size // 3, size // 4), 140, -1)
        cv2.line(patch, (5, size - 10), (size - 6, size // 2), 10, 4)
    else:
        patch = np.full((size, size, 3), 250, np.uint8)
        cv2.circle(patch, (size // 2, size // 2), size // 3, (30, 30, 205), -1)
        cv2.rectangle(patch, (8, 8), (size // 3, size // 4), (40, 200, 40), -1)
        cv2.line(patch, (5, size - 10), (size - 6, size // 2), (10, 10, 10), 4)
    out[y:y + size, x:x + size] = patch
    return out
