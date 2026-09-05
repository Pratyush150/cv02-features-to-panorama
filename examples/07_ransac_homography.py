"""07 -- RANSAC and the homography: why least squares dies, and how to size a threshold.

Least squares minimises the sum of *squared* errors, which assumes every
measurement is a mildly noisy inlier. A 0.5 px error contributes 0.25 to the
sum; a 200 px error contributes 40,000. One wrong match outvotes 160,000 good
ones, and this example measures exactly how badly.

Then the two numbers people guess and should derive:

* the inlier **threshold** -- chi-squared with 2 degrees of freedom, because a
  reprojection residual is a distance between two 2-D points and both
  coordinates are noisy. ``t = sqrt(5.99) * sigma``, not ``1.96 * sigma``.
* the **iteration count** -- ``N = ceil(log(1-p) / log(1 - w**s))``, with the
  ceiling, and with the sample size ``s`` in the exponent.

Run:  python examples/07_ransac_homography.py
Figure: docs/figures/07_ransac_homography.png
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  -- puts src/ on sys.path; see the file for why

import cv2
import numpy as np
from matplotlib import pyplot as plt

from feat import figures, ransac

SIGMA = 0.5           # keypoint localisation noise, in pixels, per coordinate
N_POINTS = 30
N_OUTLIERS = 12       # 40% of the correspondences are gross errors
PROBE = np.array([[0.0, 0.0], [400.0, 0.0], [400.0, 400.0], [0.0, 400.0]])

H_TRUE = np.array([[1.10, 0.10, 30.0],
                   [-0.05, 1.05, 12.0],
                   [1e-4, 5e-5, 1.0]])


def make_correspondences(seed: int = 1):
    """30 correspondences from a known H, 12 of them then replaced with junk.

    The outliers are drawn from the same box as the true points, so they are
    not obvious in a scatter plot and cannot be removed by any check on the
    coordinates alone. Only the *model* separates them, which is the situation
    RANSAC is for.
    """
    rng = np.random.default_rng(seed)
    src = rng.uniform(0, 400, (N_POINTS, 2))
    dst = ransac.transfer(H_TRUE, src) + rng.normal(0, SIGMA, (N_POINTS, 2))
    dst[:N_OUTLIERS] = rng.uniform(0, 400, (N_OUTLIERS, 2))
    truth = np.zeros(N_POINTS, bool)
    truth[N_OUTLIERS:] = True
    return src, dst, truth


def corner_error(h: np.ndarray) -> float:
    """Worst displacement of the four probe corners against the true homography.

    This is the metric that matters, and it is not the residual on the fitted
    points. A model can fit its own data beautifully and still push the corners
    of the image tens of pixels out of place -- the residual is what the fit
    minimised, so it is the one number the fit cannot be judged by.
    """
    return float(np.linalg.norm(ransac.transfer(h, PROBE) - ransac.transfer(H_TRUE, PROBE), axis=1).max())


def main() -> None:
    figures.use_teaching_style()
    src, dst, truth = make_correspondences()

    # ---- the DLT, checked against OpenCV's own least-squares solve ---------
    exact_src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    exact_dst = np.array([[12, 9], [108, 3], [120, 112], [5, 98]], float)
    mine = ransac.dlt_homography(exact_src, exact_dst)
    theirs, _ = cv2.findHomography(exact_src, exact_dst, 0)  # 0 = plain DLT, no RANSAC
    theirs = theirs / theirs[2, 2]
    print("the DLT, on four exact correspondences")
    print(f"  largest entry-wise difference from cv2.findHomography(..., 0) : "
          f"{np.abs(mine - theirs).max():.3e}")
    print(f"  the four source points pushed through our H land within        : "
          f"{np.abs(ransac.transfer(mine, exact_src) - exact_dst).max():.3e} px of the targets")
    print("  A is 8x9 for four points, so numpy returns 8 singular values for 9 directions in")
    print("  h-space. Vt[-1] is right either way -- the 9th direction is the exact null space\n")

    # ---- least squares against RANSAC, on the contaminated set ------------
    threshold = ransac.inlier_threshold(SIGMA, dof=2, confidence=0.99)
    tight = ransac.inlier_threshold(SIGMA, dof=1, confidence=0.95)  # the widely quoted 1.96 sigma
    result = ransac.ransac_homography(src, dst, threshold, seed=0)
    least_squares = ransac.dlt_homography(src, dst)  # every point, outliers included

    print(f"{N_POINTS} correspondences, {N_OUTLIERS} of them gross outliers "
          f"({100 * N_OUTLIERS / N_POINTS:.0f}%), sigma = {SIGMA} px")
    print(f"  worst residual of a CLEAN match under H_true : "
          f"{ransac.residuals(H_TRUE, src, dst)[N_OUTLIERS:].max():.3f} px")
    print(f"  threshold sqrt(-2 ln 0.01) * sigma (2 DoF)   : {threshold:.3f} px")
    print(f"  the 1.96 * sigma figure (1 DoF, wrong here)  : {tight:.3f} px\n")

    print(f"  least squares on all {N_POINTS}   : worst corner error "
          f"{corner_error(least_squares):10.3f} px")
    print(f"  RANSAC (ours)              : worst corner error {corner_error(result.H):10.3f} px   "
          f"inliers {int(result.inliers.sum())}/{N_POINTS}   "
          f"recovered exactly the {N_POINTS - N_OUTLIERS} clean ones: "
          f"{np.array_equal(result.inliers, truth)}   "
          f"iterations drawn {result.iterations}")

    h_cv, mask_cv = cv2.findHomography(src, dst, cv2.RANSAC, threshold)
    h_mg, mask_mg = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, threshold)
    print(f"  cv2.RANSAC                 : worst corner error "
          f"{corner_error(h_cv / h_cv[2, 2]):10.3f} px   inliers {int(mask_cv.sum())}/{N_POINTS}")
    print(f"  cv2.USAC_MAGSAC            : worst corner error "
          f"{corner_error(h_mg / h_mg[2, 2]):10.3f} px   inliers {int(mask_mg.sum())}/{N_POINTS}")
    print(f"\n  Least squares is {corner_error(least_squares) / corner_error(result.H):.0f}x worse. "
          "Not a little worse -- it is not an estimate of anything.")
    print("  MAGSAC++ marginalises over the threshold instead of making you guess it, which is,")
    print("  per everything below, the parameter that actually breaks RANSAC.\n")

    # ---- the threshold sweep ---------------------------------------------
    print(f"{'t (px)':>8}  {'what it is':<30}{'kept':>6}{'true':>8}{'outliers':>10}{'corner err':>13}")
    sweep = []
    for label, t in (
        ("1.96*sigma  (1 DoF, wrong)", tight),
        ("2.45*sigma  (2 DoF, 95%)", ransac.inlier_threshold(SIGMA, 2, 0.95)),
        ("3.03*sigma  (2 DoF, 99%)", threshold),
        ("3.0 px      (a magic number)", 3.0),
        ("60.0 px     (far too loose)", 60.0),
    ):
        r = ransac.ransac_homography(src, dst, t, seed=0)
        tp = int((r.inliers & truth).sum())
        fp = int((r.inliers & ~truth).sum())
        err = corner_error(r.H)
        sweep.append((t, label, int(r.inliers.sum()), tp, fp, err))
        print(f"{t:>8.2f}  {label:<30}{int(r.inliers.sum()):>6}{tp:>5}/{N_POINTS - N_OUTLIERS}"
              f"{fp:>10}{err:>10.3f} px")
    print("\n  Row 1 is paper-drill arithmetic made real: the 1-DoF threshold is 20% too tight and")
    print("  discards genuine matches. The last row is the other failure -- one admitted outlier")
    print(f"  moves the worst corner by {sweep[-1][5]:.1f} px against {sweep[2][5]:.2f} px.\n")

    # ---- the iteration count ---------------------------------------------
    print("iterations needed, N = ceil(log(1-p) / log(1 - w^s)), p = 0.99")
    print("  w = 0.5, sample size s = 1..8 :",
          [ransac.iterations_needed(0.5, s) for s in range(1, 9)])
    print("  s = 4 (homography), w sweep   :",
          [(w, ransac.iterations_needed(w, 4)) for w in (0.8, 0.6, 0.5, 0.3, 0.2)])
    print("  The sample size sits in the EXPONENT, so each extra point roughly doubles the work.")
    print("  That is why you always fit with the smallest sample the model allows -- and why a")
    print("  hard-coded 100 iterations returns a confidently wrong model at 20% inliers, silently.\n")

    # ---- the figure -------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), constrained_layout=True)

    ax = axes[0, 0]
    mapped = ransac.transfer(result.H, src)
    for i in range(N_POINTS):
        colour = "#1a9641" if result.inliers[i] else "#c1272d"
        ax.plot([src[i, 0], dst[i, 0]], [src[i, 1], dst[i, 1]], "-", color=colour, lw=0.8, alpha=0.75)
    ax.scatter(src[:, 0], src[:, 1], s=16, c="#333333", label="source point", zorder=3)
    ax.scatter(dst[:, 0], dst[:, 1], s=16, c="#3b6ea5", marker="s", label="matched point", zorder=3)
    ax.scatter(mapped[:, 0], mapped[:, 1], s=40, facecolors="none", edgecolors="#e08214",
               label="where our H sends the source", zorder=4)
    ax.set_title(f"{N_POINTS} correspondences, {N_OUTLIERS} of them junk\n"
                 f"green = RANSAC believed it ({int(result.inliers.sum())}), red = rejected")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_aspect("equal")

    ax = axes[0, 1]
    res_true = ransac.residuals(result.H, src, dst)
    ax.hist(res_true[truth], bins=np.linspace(0, 8, 33), color="#1a9641", label="true inliers")
    ax.hist(np.clip(res_true[~truth], 0, 8), bins=np.linspace(0, 8, 33), color="#c1272d",
            alpha=0.85, label="true outliers (clipped at 8)")
    # The thresholds go in the legend rather than as inline text: at 0.98 and
    # 1.52 px on an axis that runs to 8, two rotated labels overlap each other
    # and neither is readable.
    for value, name, colour in ((tight, f"1.96 sigma, 1 DoF = {tight:.2f} px", "#7b3294"),
                                (threshold, f"3.03 sigma, 2 DoF = {threshold:.2f} px", "#111111")):
        ax.axvline(value, color=colour, ls="--", lw=1.4, label=name)
    ax.set_xlabel("reprojection residual under the fitted H (px)")
    ax.set_ylabel("count")
    ax.set_title("the threshold is a decision boundary\nand the two rules disagree in a real band")
    ax.legend(fontsize=7.5)

    ax = axes[0, 2]
    labels = ["least squares\n(all 30 points)", "RANSAC\n(ours)", "cv2.RANSAC", "cv2.USAC_MAGSAC"]
    values = [corner_error(least_squares), corner_error(result.H),
              corner_error(h_cv / h_cv[2, 2]), corner_error(h_mg / h_mg[2, 2])]
    ax.bar(labels, values, color=["#c1272d", "#1a9641", "#3b6ea5", "#7b3294"], width=0.6)
    ax.set_yscale("log")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.2f} px", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("worst corner displacement vs H_true (px, log scale)")
    ax.set_title("one bad match is not a small problem")
    ax.tick_params(axis="x", labelsize=7.5)

    ax = axes[1, 0]
    samples = np.arange(1, 9)
    for w, colour in ((0.8, "#1a9641"), (0.5, "#3b6ea5"), (0.3, "#c1272d")):
        ax.semilogy(samples, [ransac.iterations_needed(w, int(s)) for s in samples], "o-",
                    color=colour, label=f"inlier ratio w = {w}", ms=4)
    ax.axvline(4, color="#333333", ls=":", lw=1)
    ax.text(4.1, 2, "s = 4\n(homography)", fontsize=7.5)
    ax.set_xlabel("minimal sample size s")
    ax.set_ylabel("iterations for p = 0.99")
    ax.set_title("the sample size is in the exponent")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ratios = np.linspace(0.15, 0.95, 60)
    ax.semilogy(ratios, [ransac.iterations_needed(w, 4) for w in ratios], "-", color="#3b6ea5")
    ax.set_xlabel("inlier ratio w")
    ax.set_ylabel("iterations for p = 0.99, s = 4")
    ax.set_title("and a too-tight threshold shrinks w,\nwhich is what makes it expensive twice over")
    for w in (0.2, 0.5, 0.8):
        n = ransac.iterations_needed(w, 4)
        ax.plot([w], [n], "o", color="#c1272d")
        ax.annotate(f"w={w}: {n}", (w, n), textcoords="offset points", xytext=(6, 4), fontsize=7.5)

    ax = axes[1, 2]
    ts = [row[0] for row in sweep]
    ax.plot(ts, [row[3] for row in sweep], "o-", color="#1a9641", label="true inliers kept")
    ax.plot(ts, [row[4] for row in sweep], "s-", color="#c1272d", label="outliers admitted")
    ax.set_xscale("log")
    ax.set_xlabel("inlier threshold t (px, log scale)")
    ax.set_ylabel("count")
    twin = ax.twinx()
    twin.plot(ts, [row[5] for row in sweep], "^--", color="#7b3294", label="corner error")
    twin.set_yscale("log")
    twin.set_ylabel("worst corner error (px)", color="#7b3294")
    twin.grid(False)
    ax.set_title("too tight loses inliers, too loose admits one\noutlier and the model follows it")
    ax.legend(fontsize=7.5, loc="center left")

    fig.suptitle(
        "RANSAC: sample four, fit, count the agreement, keep the biggest consensus, refit on it",
        fontsize=11,
    )
    print(f"wrote {figures.save(fig, '07_ransac_homography.png')}")


if __name__ == "__main__":
    main()
