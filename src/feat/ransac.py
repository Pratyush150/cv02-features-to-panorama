"""Robust homography estimation: the DLT, and RANSAC around it.

**Why least squares is not enough.** Least squares minimises the sum of
*squared* errors, which assumes every measurement is a mildly noisy inlier. A
0.5 px error contributes 0.25 to the sum; a 200 px error contributes 40,000.
One wrong match therefore outvotes 160,000 good ones. It is the arithmetic of a
mean salary in a room where somebody walked a billionaire in.

**What RANSAC does instead.** Do not fit all the data -- find the model the
majority agrees on. Sample the minimum number of correspondences that
determines the model (four, for a homography), fit, apply that model to every
match, count how many land within a threshold, keep the biggest consensus, and
only then refit on the whole inlier set. Outliers never enter the final fit.

**Two numbers people get wrong**, both derived rather than guessed below: the
inlier threshold (:func:`inlier_threshold`) and the iteration count
(:func:`iterations_needed`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "RansacResult",
    "dlt_homography",
    "transfer",
    "residuals",
    "inlier_threshold",
    "iterations_needed",
    "ransac_homography",
]


@dataclass(frozen=True)
class RansacResult:
    """The fitted homography and the evidence for it."""

    H: np.ndarray  # 3x3, normalised so H[2, 2] == 1
    inliers: np.ndarray  # bool mask over the input correspondences
    iterations: int  # how many samples were actually drawn
    threshold: float  # the residual cut-off used, in pixels

    @property
    def inlier_ratio(self) -> float:
        return float(self.inliers.mean()) if self.inliers.size else 0.0


def _hartley_normalise(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate the centroid to the origin and scale so mean distance is sqrt(2).

    This is not optional and skipping it is the single most common reason a
    hand-rolled homography "almost works". The DLT matrix has columns like
    ``u*x`` alongside columns that are just ``1``; with pixel coordinates in the
    hundreds, ``u*x`` runs to 1e5 while its neighbour is 1, and the SVD spends
    all of its precision on the large columns. The symptom is an H that fits
    the four points it was given and visibly misfits everything else, with no
    error raised anywhere.

    Returns the normalised points and the 3x3 ``T`` that produced them, because
    the fit has to be undone afterwards: ``H = inv(T_dst) @ H_norm @ T_src``.
    """
    mu = p.mean(axis=0)
    mean_dist = np.sqrt(((p - mu) ** 2).sum(axis=1)).mean()
    # A degenerate sample (all four points identical) gives mean_dist == 0 and
    # a division by zero that becomes an inf and then a NaN in the SVD. Return
    # something finite and let the caller's isfinite guard reject the sample.
    scale = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
    t = np.array(
        [[scale, 0.0, -scale * mu[0]], [0.0, scale, -scale * mu[1]], [0.0, 0.0, 1.0]]
    )
    homog = np.hstack([p, np.ones((len(p), 1))]) @ t.T
    return homog[:, :2], t


def dlt_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The Direct Linear Transform: the 3x3 H mapping ``src -> dst``.

    Start from ``x' = x_tilde / w_tilde``, multiply through by ``w_tilde`` to
    clear the fraction, and what is left is *linear* in the nine entries of H.
    Do the same for ``y'`` and each correspondence contributes two rows:

        [ -x  -y  -1   0   0   0   u*x   u*y   u ]
        [  0   0   0  -x  -y  -1   v*x   v*y   v ]

    Stack them into ``A`` and the system is ``A h = 0``. That has no useful
    solution as stated -- ``h = 0`` always works, and any scaling of a solution
    is also a solution -- so fix the scale and ask instead which *unit-length*
    ``h`` makes ``||A h||`` smallest. The SVD answers exactly that: the
    direction ``A`` shrinks most is the last row of ``Vt``.

    ``Vt[-1]`` rather than "the column matching the smallest singular value" is
    deliberate. With exactly four correspondences ``A`` is 8x9, so NumPy returns
    eight singular values for nine directions in h-space -- the ninth direction
    is the exact null space and never had a singular value to be smallest.
    ``Vt[-1]`` is right either way.

    Needs at least four correspondences, and they must not be collinear: four
    points on a line leave whole directions of the plane-to-plane map
    unconstrained. Collinear input returns a non-finite H rather than raising,
    because inside a RANSAC loop a degenerate sample is an ordinary event to be
    skipped, not an exception to be handled.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError(f"need >= 4 matched pairs, got {len(src)} and {len(dst)}")

    src_n, t_src = _hartley_normalise(src)
    dst_n, t_dst = _hartley_normalise(dst)

    rows = np.empty((2 * len(src_n), 9))
    x, y = src_n[:, 0], src_n[:, 1]
    u, v = dst_n[:, 0], dst_n[:, 1]
    one = np.ones_like(x)
    zero = np.zeros_like(x)
    rows[0::2] = np.stack([-x, -y, -one, zero, zero, zero, u * x, u * y, u], axis=1)
    rows[1::2] = np.stack([zero, zero, zero, -x, -y, -one, v * x, v * y, v], axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        if not np.all(np.isfinite(rows)):
            return np.full((3, 3), np.nan)
        _, _, vt = np.linalg.svd(rows)
        h_norm = vt[-1].reshape(3, 3)
        h = np.linalg.inv(t_dst) @ h_norm @ t_src
        if abs(h[2, 2]) < 1e-12:
            return np.full((3, 3), np.nan)
        return h / h[2, 2]  # pin the scale: H and 5H are the same homography


def transfer(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Push ``(N, 2)`` points through a homography, perspective divide included.

    The divide by ``w_tilde`` is the only nonlinear step in the whole operation
    and it is what makes perspective possible: ``w_tilde`` differs from pixel to
    pixel, so different parts of the image shrink by different amounts. Forget
    it and you have applied an affine transform with an extra row.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    homog = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(h, dtype=np.float64).T
    w = homog[:, 2:]
    # A point on the homography's horizon line has w == 0 and maps to infinity;
    # past it, w flips sign and the point lands mirrored on the far side. That
    # is the torn, smeared, reflected warp you see from a broken H. Guard the
    # divide so the residual comes out huge (and the match is rejected) instead
    # of NaN (which compares False against every threshold and is thus counted
    # as a rejection by accident rather than on purpose).
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return homog[:, :2] / w


def residuals(h: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Reprojection error in pixels: how far each mapped source point misses."""
    return np.linalg.norm(transfer(h, src) - np.asarray(dst, dtype=np.float64).reshape(-1, 2), axis=1)


def inlier_threshold(sigma: float, dof: int = 2, confidence: float = 0.95) -> float:
    """The residual cut-off, derived from the detector's localisation noise.

    A reprojection residual is the distance between two 2-D points, so **both**
    its x and y parts are noisy. The sum of two squared unit Gaussians follows
    chi-squared with **2** degrees of freedom, whose 95% point is 5.99, giving
    ``t = sqrt(5.99) * sigma ~= 2.45 * sigma``.

    The widely quoted 1.96 is the two-sided 95% interval for a *single* scalar
    -- chi-squared with 1 degree of freedom, 95% point 3.84, sqrt = 1.96. It is
    the right answer to a different question: a point-to-**line** residual, such
    as the epipolar distance in ``findFundamentalMat``. Using it for a
    homography sets the threshold about 20% too tight, which discards roughly
    10% of genuine inliers -- and because the measured inlier ratio then sits in
    the *exponent* of :func:`iterations_needed`, it also inflates the iteration
    count by around half. One wrong distribution, two costs.

    Derived rather than tabulated: for ``dof = 2`` the survival function is
    ``exp(-t**2 / 2)``, so ``t**2 = -2 ln(1 - confidence)``. The 1-DoF case uses
    the normal quantile. No SciPy dependency for two numbers.
    """
    if dof == 2:
        return float(np.sqrt(-2.0 * np.log(1.0 - confidence)) * sigma)
    if dof == 1:
        # Two-sided normal quantile at `confidence`, by bisection on erf. Exact
        # to float precision and shorter than shipping a table.
        from math import erf, sqrt

        lo, hi = 0.0, 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if erf(mid / sqrt(2.0)) < confidence:
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi) * sigma)
    raise ValueError("dof must be 1 (point-to-line) or 2 (point-to-point)")


def iterations_needed(inlier_ratio: float, sample: int = 4, success: float = 0.99) -> int:
    """``N = ceil(log(1 - p) / log(1 - w**s))`` -- how many samples to draw.

    One sample is all-inliers with probability ``w**s``, so it is contaminated
    with probability ``1 - w**s``, so N independent samples are all contaminated
    with probability ``(1 - w**s)**N``. Require that below ``1 - p`` and solve.

    The **ceiling** is not cosmetic. The line-fitting case lands on 16.008, and
    at exactly 16 trials the success probability is 0.98998 -- fractionally under
    the 0.99 that was asked for. A fractional trial buys nothing.

    The sample size sits in the *exponent*, which is the fact worth carrying:
    at 50% inliers a 4-point homography needs 72 iterations and a 2-point line
    needs 17. That is why you always fit with the smallest sample the model
    allows, and why dropping from 50% to 20% inliers takes the homography from
    72 iterations to 2876.
    """
    if not 0.0 < inlier_ratio < 1.0:
        raise ValueError(f"inlier_ratio must be in (0, 1), got {inlier_ratio}")
    p_clean = inlier_ratio ** sample
    if p_clean >= 1.0:
        return 1
    return int(np.ceil(np.log(1.0 - success) / np.log(1.0 - p_clean)))


def ransac_homography(
    src: np.ndarray,
    dst: np.ndarray,
    threshold: float,
    max_iterations: int = 2000,
    success: float = 0.99,
    seed: int = 0,
    adaptive: bool = True,
) -> RansacResult:
    """Sample four, fit, count, keep the best, refit on the whole consensus set.

    ``seed`` is required rather than optional. RANSAC is randomised, so two runs
    on identical data return slightly different matrices and slightly different
    inlier masks -- which is a regression test that passes four times out of
    five unless the generator is seeded here.

    ``adaptive=True`` recomputes the needed iteration count from the best inlier
    ratio seen so far and stops early once it is reached. Hard-coding 2000 when
    72 would do is waste; hard-coding 100 when the data is 20% inliers fails
    *silently*, returning a confidently wrong model, because 20% inliers needs
    2876 samples. ``max_iterations`` stays as a ceiling for the case where the
    data is so contaminated that the formula asks for more than you can afford.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    n = len(src)
    if n < 4:
        raise ValueError(f"need >= 4 correspondences, got {n}")

    rng = np.random.default_rng(seed)
    best_mask = np.zeros(n, bool)
    best_h = None
    drawn = 0
    budget = max_iterations

    while drawn < budget:
        drawn += 1
        idx = rng.choice(n, 4, replace=False)
        h = dlt_homography(src[idx], dst[idx])
        # Three of four sampled points can be collinear, or two can coincide;
        # then A loses rank and H comes back inf or nan. Without this guard the
        # NaN propagates into best_h and every later comparison is False, so
        # the run "succeeds" with a garbage model.
        if not np.all(np.isfinite(h)):
            continue
        mask = residuals(h, src, dst) < threshold
        if mask.sum() > best_mask.sum():
            best_mask, best_h = mask, h
            if adaptive and 0 < mask.mean() < 1:
                need = iterations_needed(mask.mean(), sample=4, success=success)
                budget = min(max_iterations, max(drawn, need))

    if best_h is None or best_mask.sum() < 4:
        raise RuntimeError(
            f"no consensus set of >= 4 found in {drawn} samples at threshold {threshold:.3f} px"
        )

    # Step 6, the one everybody skips. The model out of the loop was fitted to
    # exactly four points and carries all four points' noise at full strength.
    # Refitting on the whole consensus set is safe -- the outliers are already
    # gone -- and it is the difference between a homography that is right in
    # kind and one that is right in pixels. Skip it and a stitched panorama
    # gains a soft double edge you will spend an afternoon blaming on blending.
    #
    # The refit is iterated to a fixed point rather than done once. A better
    # model admits inliers the four-point model just missed, and those inliers
    # make the next model better again. It converges in two or three passes and
    # is bounded here so a pathological oscillation cannot loop forever. This is
    # the cheap half of what the literature calls LO-RANSAC (locally optimised
    # RANSAC); on the 30-correspondence fixture in example 07 it is worth about
    # one recovered inlier and a factor of two in corner accuracy.
    for _ in range(5):
        refit = dlt_homography(src[best_mask], dst[best_mask])
        if not np.all(np.isfinite(refit)):
            break
        new_mask = residuals(refit, src, dst) < threshold
        if new_mask.sum() < 4:
            break
        best_h = refit
        if np.array_equal(new_mask, best_mask):
            best_mask = new_mask
            break
        best_mask = new_mask

    return RansacResult(H=best_h, inliers=best_mask, iterations=drawn, threshold=threshold)
