"""02 -- Why corners are not enough: Harris has a fixed idea of how big a corner is.

Harris is rotation invariant for free -- eigenvalues do not care which way the
coordinate frame points. It is **not** scale invariant, and the reason is in the
signature: ``blockSize`` is a fixed number of pixels, so the detector is always
asking about structure of one particular size.

This example measures the failure rather than asserting it, on two axes:

1. **Repeatability** -- photograph the same scene at a range of scales and count
   how many of the original corners are found again in the same place. Harris
   against SIFT, on identical inputs.
2. **The response of one disc** as the disc grows. Same shape, same contrast,
   nothing but size changing.

Run:  python examples/02_scale_breaks_harris.py
Figure: docs/figures/02_scale_breaks_harris.png
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  -- puts src/ on sys.path; see the file for why

import cv2
import numpy as np
from matplotlib import pyplot as plt

from feat import figures, harris, scenes

SCALES = [1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.25]
TOL_PX = 3.0  # how close a re-detection has to be to count as the same corner


def harris_points(gray: np.ndarray, rel: float = 0.05) -> np.ndarray:
    """Harris corners as an (N, 2) array of (x, y).

    The threshold is a fraction of *this image's own* ``R.max()``. An absolute
    cut-off would confound the experiment completely: R is fourth order in the
    gradient, so downscaling an image changes R by far more than it changes the
    number of real corners, and the "scale failure" measured below would be
    mostly a thresholding artefact. This is the same reasoning behind
    ``goodFeaturesToTrack``'s ``qualityLevel``.
    """
    return harris.peaks_component_nms(harris.harris_response(gray.astype(np.float32)), rel)


def sift_points(gray: np.ndarray, n: int = 600) -> np.ndarray:
    kps = cv2.SIFT_create(nfeatures=n).detect(gray, None)
    return np.float32([kp.pt for kp in kps]).reshape(-1, 2)


def repeatability(base_pts: np.ndarray, small_pts: np.ndarray, factor: float) -> float:
    """Fraction of ``base_pts`` re-detected within TOL_PX in the scaled image.

    The scaled image's points are mapped back to full-resolution coordinates by
    dividing by ``factor`` -- a pure scaling, so the geometry between the two
    images is the simplest relationship there is. Whatever fails here is not
    failing because the transform was hard.
    """
    if len(base_pts) == 0 or len(small_pts) == 0:
        return 0.0
    mapped = small_pts / factor
    d = np.linalg.norm(base_pts[:, None, :] - mapped[None, :, :], axis=2)
    return float((d.min(axis=1) < TOL_PX).mean())


def main() -> None:
    figures.use_teaching_style()

    scene = cv2.cvtColor(scenes.textured_wall(600, 800, seed=5), cv2.COLOR_BGR2GRAY)
    base_harris = harris_points(scene)
    base_sift = sift_points(scene)
    print(f"full resolution: {len(base_harris)} Harris corners, {len(base_sift)} SIFT keypoints\n")

    print(f"{'scale':>7}{'size':>12}{'Harris n':>10}{'repeat':>9}{'SIFT n':>9}{'repeat':>9}")
    harris_rep, sift_rep = [], []
    small_for_figure = None
    for factor in SCALES:
        _, small, _ = scenes.scale_pair(scene, factor)
        h_pts, s_pts = harris_points(small), sift_points(small)
        h_rep = repeatability(base_harris, h_pts, factor)
        s_rep = repeatability(base_sift, s_pts, factor)
        harris_rep.append(h_rep)
        sift_rep.append(s_rep)
        print(f"{factor:>7.2f}{str(small.shape[::-1]):>12}{len(h_pts):>10}{100 * h_rep:>8.1f}%"
              f"{len(s_pts):>9}{100 * s_rep:>8.1f}%")
        if abs(factor - 0.4) < 1e-9:
            small_for_figure = (small, h_pts)

    print(f"\nHarris repeatability falls from {100 * harris_rep[0]:.1f}% at scale 1.00 "
          f"to {100 * harris_rep[-1]:.1f}% at scale {SCALES[-1]:.2f}.")
    print(f"SIFT falls from {100 * sift_rep[0]:.1f}% to {100 * sift_rep[-1]:.1f}% on the same images.")
    print("Same scene, same geometry, one detector that searches over scale and one that does not.\n")

    # The single-disc experiment: one shape, one contrast, only its size
    # changing -- and the response read at the disc's own centre.
    #
    # A small disc sits entirely inside the summation window, so the window
    # sees structure in two directions and R is huge. Grow the disc past the
    # window and the window sees nothing but the disc's flat interior, so R
    # goes to *exactly* zero. There is no gradual decline and no threshold that
    # rescues it: the feature does not weaken, it ceases to exist.
    radii = list(range(1, 26))
    blocks = [5, 15, 31]
    curves: dict[int, list[float]] = {}
    for block in blocks:
        values = []
        for radius in radii:
            canvas = np.zeros((200, 200), np.float32)
            cv2.circle(canvas, (100, 100), radius, 255, -1)
            values.append(float(harris.harris_response(canvas, block=block)[100, 100]))
        curves[block] = values

    print(f"{'blockSize':>10}{'largest radius still detected':>32}")
    for block in blocks:
        last = max((r for r, v in zip(radii, curves[block]) if v > 0), default=0)
        print(f"{block:>10}{last:>32}")
    print("\nThe cut-off tracks the window and nothing else. blockSize is not a tuning knob for")
    print("sensitivity -- it is a declaration of what size of structure this detector can see, and")
    print("a detector can only declare one. Searching over scale is the only way out, which is")
    print("what a difference-of-Gaussian pyramid does (example 03).")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True)

    figures.show_gray(axes[0, 0], scene, f"full resolution: {len(base_harris)} Harris corners")
    figures.annotate_points(axes[0, 0], base_harris, size=9)

    small, small_pts = small_for_figure
    figures.show_gray(axes[0, 1], small, f"same scene at 0.40x: {len(small_pts)} corners")
    figures.annotate_points(axes[0, 1], small_pts, size=9)

    ax = axes[1, 0]
    ax.plot(SCALES, [100 * v for v in harris_rep], "o-", color="#c1272d", label="Harris (fixed window)")
    ax.plot(SCALES, [100 * v for v in sift_rep], "s-", color="#3b6ea5", label="SIFT (searches scale)")
    ax.invert_xaxis()  # the story reads left to right as the camera moves away
    ax.set_xlabel("scale factor applied to the second image")
    ax.set_ylabel("% of original keypoints re-detected within 3 px")
    ax.set_title("repeatability across scale")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[1, 1]
    for block, colour in zip(blocks, ("#c1272d", "#e08214", "#3b6ea5")):
        values = np.array(curves[block])
        # Zeros cannot be drawn on a log axis, and dropping them would hide the
        # cliff that is the whole point. They are plotted as open markers on the
        # floor instead, so "exactly zero" is visible as a distinct state.
        nonzero = values > 0
        ax.semilogy(np.array(radii)[nonzero], values[nonzero], "o-", ms=3.5, color=colour,
                    label=f"blockSize = {block}")
        if (~nonzero).any():
            floor = values[nonzero].min() / 30
            ax.semilogy(np.array(radii)[~nonzero], np.full((~nonzero).sum(), floor), "x",
                        ms=4, color=colour)
    ax.set_xlabel("disc radius (px)")
    ax.set_ylabel("R at the disc's centre")
    ax.set_title("one shape, one contrast -- only the size changes\n(x = R is exactly zero)")
    ax.legend(loc="lower left", fontsize=8)

    fig.suptitle(
        "Harris is rotation invariant for free and scale invariant not at all -- which is why SIFT exists",
        fontsize=11,
    )
    print(f"\nwrote {figures.save(fig, '02_scale_breaks_harris.png')}")


if __name__ == "__main__":
    main()
