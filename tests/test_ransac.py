"""The DLT against OpenCV, and RANSAC against a homography we chose ourselves.

The synthetic fixture is the point: because we generated the correspondences
from a known ``H``, "did it work" is a number rather than a picture, and the
outlier fraction is exactly what we say it is rather than an estimate.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from feat import ransac

SIGMA = 0.5
N_POINTS = 30
N_OUTLIERS = 12                      # 40% of the correspondences are gross errors
OUTLIER_FRACTION = N_OUTLIERS / N_POINTS
PROBE = np.array([[0.0, 0.0], [400.0, 0.0], [400.0, 400.0], [0.0, 400.0]])

H_TRUE = np.array([[1.10, 0.10, 30.0],
                   [-0.05, 1.05, 12.0],
                   [1e-4, 5e-5, 1.0]])


def make_data(seed: int = 1):
    rng = np.random.default_rng(seed)
    src = rng.uniform(0, 400, (N_POINTS, 2))
    dst = ransac.transfer(H_TRUE, src) + rng.normal(0, SIGMA, (N_POINTS, 2))
    dst[:N_OUTLIERS] = rng.uniform(0, 400, (N_OUTLIERS, 2))
    truth = np.zeros(N_POINTS, bool)
    truth[N_OUTLIERS:] = True
    return src, dst, truth


def corner_error(h: np.ndarray) -> float:
    return float(np.linalg.norm(ransac.transfer(h, PROBE) - ransac.transfer(H_TRUE, PROBE), axis=1).max())


# --------------------------------------------------------------------------
# the DLT
# --------------------------------------------------------------------------
def test_dlt_matches_opencvs_plain_least_squares():
    src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    dst = np.array([[12, 9], [108, 3], [120, 112], [5, 98]], float)
    mine = ransac.dlt_homography(src, dst)
    theirs, _ = cv2.findHomography(src, dst, 0)  # method 0 = plain DLT, no RANSAC
    assert np.abs(mine - theirs / theirs[2, 2]).max() < 1e-9


def test_dlt_is_exact_through_four_points():
    src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    dst = np.array([[12, 9], [108, 3], [120, 112], [5, 98]], float)
    h = ransac.dlt_homography(src, dst)
    assert np.abs(ransac.transfer(h, src) - dst).max() < 1e-9
    assert h[2, 2] == pytest.approx(1.0)


def test_dlt_recovers_a_known_homography_from_noiseless_points():
    rng = np.random.default_rng(4)
    src = rng.uniform(0, 400, (12, 2))
    dst = ransac.transfer(H_TRUE, src)
    assert np.abs(ransac.dlt_homography(src, dst) - H_TRUE).max() < 1e-9


def test_hartley_normalisation_is_not_optional():
    """Without it the DLT degrades badly on large pixel coordinates.

    Fitted on points around x = 4000, the un-normalised columns of A span five
    orders of magnitude and the SVD spends its precision on the largest ones.
    The comparison is against the normalised solver on the same data.
    """
    rng = np.random.default_rng(5)
    src = rng.uniform(3800, 4200, (8, 2))
    dst = ransac.transfer(H_TRUE, src)

    normalised = float(np.abs(ransac.transfer(ransac.dlt_homography(src, dst), src) - dst).max())

    rows = []
    for (x, y), (u, v) in zip(src, dst):  # the same solve with the scaling step removed
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.array(rows))
    raw = vt[-1].reshape(3, 3)
    raw = raw / raw[2, 2]
    unnormalised = float(np.abs(ransac.transfer(raw, src) - dst).max())

    assert normalised < 1e-6
    assert unnormalised > 100 * max(normalised, 1e-12), (
        f"normalised residual {normalised:.3e} vs un-normalised {unnormalised:.3e}"
    )


def test_degenerate_samples_return_nan_rather_than_raising():
    """Inside a RANSAC loop a collinear sample is an ordinary event to skip."""
    collinear = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    target = np.array([[0.0, 0.0], [10.0, 1.0], [20.0, 2.0], [30.0, 3.0]])
    h = ransac.dlt_homography(collinear, target)
    assert not np.all(np.isfinite(h))

    with pytest.raises(ValueError):
        ransac.dlt_homography(collinear[:3], target[:3])  # fewer than 4 pairs


# --------------------------------------------------------------------------
# the two derived numbers
# --------------------------------------------------------------------------
def test_inlier_threshold_is_the_chi_squared_answer():
    # 2 DoF at 95%: t = sqrt(5.99) * sigma ~= 2.45 sigma
    assert ransac.inlier_threshold(1.0, dof=2, confidence=0.95) == pytest.approx(2.4477, abs=1e-3)
    # 1 DoF at 95% -- the widely quoted figure, correct for a point-to-LINE residual
    assert ransac.inlier_threshold(1.0, dof=1, confidence=0.95) == pytest.approx(1.9600, abs=1e-3)
    assert ransac.inlier_threshold(2.0, dof=2, confidence=0.95) == pytest.approx(
        2 * ransac.inlier_threshold(1.0, dof=2, confidence=0.95)
    )
    with pytest.raises(ValueError):
        ransac.inlier_threshold(1.0, dof=3)


def test_iteration_formula_including_the_ceiling():
    assert ransac.iterations_needed(0.5, sample=4, success=0.99) == 72   # raw 71.355
    assert ransac.iterations_needed(0.5, sample=2, success=0.99) == 17   # raw 16.008 -- rounds UP
    assert [ransac.iterations_needed(0.5, s) for s in range(1, 9)] == [7, 17, 35, 72, 146, 293, 588, 1177]
    assert ransac.iterations_needed(0.2, 4) == 2876
    # At exactly 16 trials the line case succeeds with probability 0.98998,
    # fractionally under the 0.99 that was requested. That is why it is a ceiling.
    assert (1 - (1 - 0.5 ** 2) ** 16) < 0.99 <= (1 - (1 - 0.5 ** 2) ** 17)


# --------------------------------------------------------------------------
# RANSAC
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(8))
def test_ransac_recovers_the_known_homography_and_the_exact_inlier_set(seed):
    """At the 99% threshold, exactly the 18 clean matches, on every seed tried."""
    src, dst, truth = make_data()
    threshold = ransac.inlier_threshold(SIGMA, dof=2, confidence=0.99)
    result = ransac.ransac_homography(src, dst, threshold, seed=seed)

    assert np.array_equal(result.inliers, truth), (
        f"seed {seed}: recovered {int(result.inliers.sum())} inliers, "
        f"{int((result.inliers & ~truth).sum())} of them outliers"
    )
    assert corner_error(result.H) < 1.5, f"seed {seed}: corner error {corner_error(result.H):.3f} px"
    assert result.inlier_ratio == pytest.approx(1 - OUTLIER_FRACTION)


@pytest.mark.parametrize("seed", range(8))
def test_ransac_never_admits_an_outlier_at_the_95_percent_threshold(seed):
    """The tighter, more standard threshold: no false positives, and by construction
    it may lose about 5% of the true inliers -- which is what 95% confidence means."""
    src, dst, truth = make_data()
    result = ransac.ransac_homography(
        src, dst, ransac.inlier_threshold(SIGMA, dof=2, confidence=0.95), seed=seed
    )
    assert int((result.inliers & ~truth).sum()) == 0
    assert int((result.inliers & truth).sum()) >= 17  # at most one of the 18 lost
    assert corner_error(result.H) < 3.0


def test_ransac_rejects_the_stated_outlier_fraction():
    src, dst, truth = make_data()
    result = ransac.ransac_homography(
        src, dst, ransac.inlier_threshold(SIGMA, dof=2, confidence=0.99), seed=0
    )
    rejected = int((~result.inliers).sum())
    assert rejected == N_OUTLIERS, f"rejected {rejected} of {N_OUTLIERS} planted outliers"
    assert OUTLIER_FRACTION == pytest.approx(0.4)


def test_least_squares_is_destroyed_by_the_same_data():
    """The reason RANSAC exists, as a ratio rather than an adjective."""
    src, dst, _ = make_data()
    robust = ransac.ransac_homography(
        src, dst, ransac.inlier_threshold(SIGMA, dof=2, confidence=0.99), seed=0
    )
    naive = ransac.dlt_homography(src, dst)  # every point, outliers included
    assert corner_error(naive) > 100
    assert corner_error(naive) > 100 * corner_error(robust.H)


def test_ransac_agrees_with_opencvs_estimator():
    src, dst, _ = make_data()
    threshold = ransac.inlier_threshold(SIGMA, dof=2, confidence=0.99)
    mine = ransac.ransac_homography(src, dst, threshold, seed=0)
    theirs, mask = cv2.findHomography(src, dst, cv2.RANSAC, threshold)
    assert int(mask.sum()) == int(mine.inliers.sum())
    # Both are within a couple of pixels of the true corners; OpenCV samples
    # internally so the two matrices are not identical and should not be
    # asserted to be.
    assert corner_error(theirs / theirs[2, 2]) < 3.0


def test_ransac_is_deterministic_for_a_given_seed():
    src, dst, _ = make_data()
    threshold = ransac.inlier_threshold(SIGMA, dof=2, confidence=0.99)
    first = ransac.ransac_homography(src, dst, threshold, seed=3)
    second = ransac.ransac_homography(src, dst, threshold, seed=3)
    assert np.array_equal(first.inliers, second.inliers)
    assert np.array_equal(first.H, second.H)


def test_adaptive_stopping_is_cheaper_and_no_worse_here():
    src, dst, truth = make_data()
    threshold = ransac.inlier_threshold(SIGMA, dof=2, confidence=0.99)
    adaptive = ransac.ransac_homography(src, dst, threshold, seed=0, adaptive=True)
    fixed = ransac.ransac_homography(src, dst, threshold, seed=0, adaptive=False,
                                     max_iterations=2000)
    assert adaptive.iterations < fixed.iterations
    assert np.array_equal(adaptive.inliers, truth)


def test_transfer_applies_the_perspective_divide():
    """The worked-by-hand example: two points on one horizontal line stop being
    on one horizontal line, because w varies from pixel to pixel."""
    h = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, 20.0], [0.001, 0.0, 1.0]])
    out = ransac.transfer(h, np.array([[100.0, 50.0], [500.0, 50.0]]))
    assert out[0] == pytest.approx([190.909091, 109.090909], abs=1e-5)
    assert out[1] == pytest.approx([673.333333, 80.0], abs=1e-5)

    affine = h.copy()
    affine[2, 0] = 0.0  # zero the bottom row and it is a plain affine transform
    flat = ransac.transfer(affine, np.array([[100.0, 50.0], [500.0, 50.0]]))
    assert flat[0][1] == flat[1][1] == 120.0
    assert np.linalg.norm(flat[1] - flat[0]) == pytest.approx(800.0)


def test_transfer_does_not_produce_nan_past_the_horizon():
    """A point on the line w == 0 maps to infinity. It must come back as a huge
    residual (and be rejected) rather than a NaN, which compares False against
    every threshold and would be counted as a rejection by accident."""
    h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.001, 0.0, 1.0]])
    out = ransac.transfer(h, np.array([[-1000.0, 0.0]]))  # exactly on the horizon
    assert np.isfinite(out).all()
    assert np.abs(out).max() > 1e6
