"""The claim this module makes is "my Harris IS cv2.cornerHarris, times a constant".

Asserting that is what makes the claim credible rather than decorative, and it
is the reason the repository is allowed to say "you may now call the library
version for the rest of your life".
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from feat import harris, scenes

# The stated thresholds. They are in the README and in the module docstring, so
# they live here as named constants rather than as magic numbers in an assert.
MIN_CORRELATION = 0.999
MAX_RELATIVE_DIFFERENCE = 1e-6


def _square() -> np.ndarray:
    img = np.zeros((60, 60), np.float32)
    img[20:40, 20:40] = 255.0
    return img


@pytest.mark.parametrize("image", [
    _square(),
    cv2.cvtColor(scenes.textured_wall(200, 260, seed=4), cv2.COLOR_BGR2GRAY).astype(np.float32),
    scenes.checkerboard(squares=5, square_px=30, blur=True).image.astype(np.float32),
])
def test_agrees_with_opencv(image):
    """Correlation above the stated threshold, and the exact s**4 scale factor."""
    mine = harris.harris_response(image)
    theirs = cv2.cornerHarris(image, harris.BLOCK, harris.KSIZE, harris.K)

    correlation = float(np.corrcoef(mine.ravel(), theirs.ravel())[0, 1])
    assert correlation > MIN_CORRELATION, f"correlation only {correlation}"

    # The error is measured against the largest response in the image, not
    # element-wise. R spans about twelve orders of magnitude on one image, so a
    # per-element relative tolerance is dominated by float32 noise on the near-
    # zero flat regions, where "50% off" means 1e-9 against 2e-9 and matters to
    # nobody. The scale-relative figure is the one the README quotes.
    scaled = mine * harris.opencv_response_scale()
    relative = float(np.abs(scaled - theirs).max() / np.abs(theirs).max())
    assert relative < MAX_RELATIVE_DIFFERENCE, f"relative difference {relative:.3e}"


def test_the_scale_factor_is_derived_not_fitted():
    """s**4 must come from the formula, not from dividing the two answers.

    If the scale were fitted, this test would be circular and would keep passing
    if OpenCV changed its internal scaling. Computing it from ksize and
    blockSize means a change would break the test, which is the point.
    """
    assert harris.opencv_response_scale(block=5, ksize=3) == pytest.approx((1 / 20.0) ** 4)
    assert harris.opencv_response_scale(block=2, ksize=3) == pytest.approx((1 / 8.0) ** 4)


def test_the_four_paper_patches_classify_correctly():
    """Flat, edge, corner, and the 45-degree edge that looks like a corner."""
    patches = {
        "flat": (np.full((5, 5), 10.0), "flat"),
        "vertical edge": (np.array([[0, 0, 10, 10, 10]] * 5, float), "edge"),
        "corner": (np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 10, 10, 10],
                             [0, 0, 10, 10, 10], [0, 0, 10, 10, 10]], float), "corner"),
        "45-degree edge": (np.array([[10, 10, 10, 10, 10], [0, 10, 10, 10, 10], [0, 0, 10, 10, 10],
                                     [0, 0, 0, 10, 10], [0, 0, 0, 0, 10]], float), "edge"),
    }
    for name, (img, expected) in patches.items():
        ix = ((img[:, 2:] - img[:, :-2]) / 2.0)[1:-1, :]
        iy = ((img[2:, :] - img[:-2, :]) / 2.0)[:, 1:-1]
        sxx, syy, sxy = (ix * ix).sum(), (iy * iy).sum(), (ix * iy).sum()
        r = (sxx * syy - sxy * sxy) - harris.K * (sxx + syy) ** 2
        got = "corner" if r > 0 else ("edge" if sxx + syy > 0 else "flat")
        assert got == expected, f"{name}: R = {r}, classified {got}"

    # The trap, stated as its own assertion: the 45-degree edge has BOTH
    # diagonal entries large and is still an edge, because the cross term drives
    # the determinant to exactly zero.
    img = patches["45-degree edge"][0]
    ix = ((img[:, 2:] - img[:, :-2]) / 2.0)[1:-1, :]
    iy = ((img[2:, :] - img[:-2, :]) / 2.0)[:, 1:-1]
    sxx, syy, sxy = (ix * ix).sum(), (iy * iy).sum(), (ix * iy).sum()
    assert sxx == 125 and syy == 125 and sxy == -125
    assert sxx * syy - sxy * sxy == 0


def test_eigenvalues_match_numpy_and_reproduce_det_and_trace():
    rng = np.random.default_rng(0)
    a, c = rng.uniform(0, 100, 50), rng.uniform(0, 100, 50)
    b = rng.uniform(-50, 50, 50)
    lam1, lam2 = harris.eigenvalues(a, c, b)
    for i in range(len(a)):
        reference = np.sort(np.linalg.eigvalsh(np.array([[a[i], b[i]], [b[i], c[i]]])))[::-1]
        assert np.allclose([lam1[i], lam2[i]], reference)
    assert np.allclose(lam1 * lam2, a * c - b * b)  # det = l1 * l2
    assert np.allclose(lam1 + lam2, a + c)          # trace = l1 + l2


def test_eigenvalues_never_return_nan():
    """The discriminant is clamped, so float32 cancellation cannot produce NaN.

    A NaN here would propagate silently into every eigenvalue figure and into
    classify_field, where NaN compares False against every threshold and is
    therefore counted as 'flat' by accident rather than on purpose.
    """
    huge = np.float32([1e12, 1e12, 1e-6])
    lam1, lam2 = harris.eigenvalues(huge, huge, np.float32([0, 1e-9, 0]))
    assert np.isfinite(lam1).all() and np.isfinite(lam2).all()


def test_k_at_or_above_a_quarter_switches_the_detector_off():
    """R > 0 requires r/(1+r)**2 > k, and that expression peaks at exactly 0.25."""
    img = _square()
    counts = [int((cv2.cornerHarris(img, 5, 3, k) > 0).sum()) for k in (0.04, 0.24, 0.25, 0.30)]
    assert counts[0] > counts[1] > 0
    assert counts[2] == 0 and counts[3] == 0


def test_naive_suppression_fails_on_plateaus_and_component_suppression_does_not():
    """The bug in example 06, asserted in the direction the text claims.

    Both numbers matter. The naive version returning far too many points is the
    bug; the component version returning exactly the known answer is the fix.
    """
    board = scenes.checkerboard(blur=False)
    response = harris.harris_response(board.image.astype(np.float32))
    naive = harris.peaks_naive_nms(response, 0.01)
    fixed = harris.peaks_component_nms(response, 0.01)

    x0, y0, x1, y1 = board.interior

    def inside(points):
        x, y = points[:, 0], points[:, 1]
        return int(((x > x0) & (x < x1) & (y > y0) & (y < y1)).sum())

    assert inside(fixed) == board.n_interior, "component NMS should find exactly the 49 X-junctions"
    assert inside(naive) > 10 * board.n_interior, "the plateau bug should be an order of magnitude"
    assert len(naive) > len(fixed)


def test_the_plateau_bug_hides_on_a_blurred_image():
    """Which is why it reproduces on the synthetic test image and not on photos."""
    blurred = scenes.checkerboard(blur=True).image.astype(np.float32)
    response = harris.harris_response(blurred)
    naive = harris.peaks_naive_nms(response, 0.01)
    fixed = harris.peaks_component_nms(response, 0.01)
    assert len(naive) == len(fixed)


def test_response_is_fourth_order_in_contrast():
    """Halve the contrast and R falls by 2**4, which is why absolute thresholds lie."""
    img = _square()
    full = float(harris.harris_response(img).max())
    half = float(harris.harris_response(img * 0.5).max())
    assert full / half == pytest.approx(16.0, rel=1e-4)
