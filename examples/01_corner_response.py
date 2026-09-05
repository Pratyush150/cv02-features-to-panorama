"""01 -- What makes a corner a corner: the structure tensor and its eigenvalues.

Builds the Harris response from the gradients up, shows the two eigenvalue
fields that the response is a shortcut for, and proves the whole thing against
``cv2.cornerHarris``.

Run:  python examples/01_corner_response.py
Figure: docs/figures/01_corner_response.png
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  -- puts src/ on sys.path; see the file for why

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from feat import figures, harris, scenes


def hand_worked_patches() -> None:
    """The four 5x5 patches you can do on paper, printed with their answers.

    Central differences, not Sobel, so the arithmetic stays small enough to
    check by hand: ``Ix = (I[r, c+1] - I[r, c-1]) / 2``. That is only defined
    where a pixel has neighbours on both sides, so the window is the inner 3x3
    and every weight is 1.
    """
    patches = {
        "A flat": np.full((5, 5), 10.0),
        "B vertical edge": np.array([[0, 0, 10, 10, 10]] * 5, float),
        "C corner": np.array(
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 10, 10, 10],
             [0, 0, 10, 10, 10], [0, 0, 10, 10, 10]], float
        ),
        "D 45-degree edge": np.array(
            [[10, 10, 10, 10, 10], [0, 10, 10, 10, 10], [0, 0, 10, 10, 10],
             [0, 0, 0, 10, 10], [0, 0, 0, 0, 10]], float
        ),
    }

    print(f"{'patch':<18}{'Sxx':>8}{'Syy':>8}{'Sxy':>8}{'det':>10}{'trace':>8}"
          f"{'R':>10}{'l1':>8}{'l2':>8}  verdict")
    for name, img in patches.items():
        ix = (img[:, 2:] - img[:, :-2]) / 2.0  # central differences, columns 1..3
        iy = (img[2:, :] - img[:-2, :]) / 2.0  # central differences, rows 1..3
        ix, iy = ix[1:-1, :], iy[:, 1:-1]      # the common inner 3x3 window
        sxx, syy, sxy = float((ix * ix).sum()), float((iy * iy).sum()), float((ix * iy).sum())
        det = sxx * syy - sxy * sxy
        trace = sxx + syy
        r = det - harris.K * trace * trace
        l1, l2 = harris.eigenvalues(np.array(sxx), np.array(syy), np.array(sxy))
        verdict = "corner" if r > 0 else ("edge" if trace > 0 else "flat")
        print(f"{name:<18}{sxx:8.0f}{syy:8.0f}{sxy:8.0f}{det:10.0f}{trace:8.0f}"
              f"{r:10.0f}{float(l1):8.0f}{float(l2):8.0f}  {verdict}")

    print("\nPatch D is the one that matters: Sxx = 125 and Syy = 125 are BOTH large,")
    print("and it is still an edge. The cross term Sxy = -125 drives det(M) to exactly")
    print("zero -- the gradient points the same way at every pixel, so there is no second")
    print("independent direction. Corner-ness cannot be read off the diagonal of M.\n")


def main() -> None:
    figures.use_teaching_style()
    hand_worked_patches()

    # A single white square on black: three of the four paper patches appear in
    # it at once, at full Sobel scale, so the printed numbers below are the
    # hand-worked ones scaled up rather than a different experiment.
    square = np.zeros((60, 60), np.float32)
    square[20:40, 20:40] = 255.0

    sxx, syy, sxy = harris.structure_tensor(square)
    response = harris.harris_response(square)
    lam1, lam2 = harris.eigenvalues(sxx, syy, sxy)

    print(f"{'patch':<10}{'Sxx':>12}{'Syy':>12}{'Sxy':>12}{'det(M)':>13}"
          f"{'trace(M)':>12}{'R':>13}")
    for name, (row, col) in [("flat", (30, 30)), ("edge", (30, 20)), ("corner", (20, 20))]:
        det = sxx[row, col] * syy[row, col] - sxy[row, col] ** 2
        print(f"{name:<10}{sxx[row, col]:12.0f}{syy[row, col]:12.0f}{sxy[row, col]:12.0f}"
              f"{det:13.4e}{sxx[row, col] + syy[row, col]:12.0f}{response[row, col]:13.4e}")

    # The referee. Nothing above used cv2.cornerHarris.
    reference = cv2.cornerHarris(square, harris.BLOCK, harris.KSIZE, harris.K)
    scale = harris.opencv_response_scale()
    rel_diff = float(np.abs(response * scale - reference).max() / np.abs(reference).max())
    corr = float(np.corrcoef(response.ravel(), reference.ravel())[0, 1])
    print(f"\nOpenCV pre-scales its Sobel by 1/(2^(ksize-1) * blockSize), so R scales by s^4 = {scale:.6e}")
    print(f"largest relative difference vs cv2.cornerHarris : {rel_diff:.3e}")
    print(f"Pearson correlation over all 3600 pixels        : {corr:.10f}")
    print(f"np.allclose(mine * s^4, cv2, rtol=1e-5)         : "
          f"{bool(np.allclose(response * scale, reference, rtol=1e-5, atol=1e-6))}")

    # A richer image for the classification panels. The white square has four
    # corners and the checkerboard has forty-nine, which is enough to look at
    # and far too few to make an eigenvalue scatter plot say anything: a scene
    # with flat regions, long edges and hundreds of corners is what shows the
    # three populations separating.
    scene = cv2.cvtColor(scenes.textured_wall(300, 400, seed=5), cv2.COLOR_BGR2GRAY)
    b_sxx, b_syy, b_sxy = harris.structure_tensor(scene.astype(np.float32))
    b_lam1, b_lam2 = harris.eigenvalues(b_sxx, b_syy, b_sxy)
    b_r = harris.harris_response(scene.astype(np.float32))
    labels = harris.classify_field(b_lam1, b_lam2)

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), constrained_layout=True)
    figures.show_gray(axes[0, 0], scene, "input: synthetic textured wall")

    for ax, field, title in (
        (axes[0, 1], b_lam1, r"$\lambda_1$  (larger eigenvalue)"),
        (axes[0, 2], b_lam2, r"$\lambda_2$  (smaller eigenvalue)"),
    ):
        # A log-ish scale: the eigenvalue fields span six orders of magnitude,
        # so a linear colour map shows the four brightest corners and a black
        # frame. The +1 keeps log of the exactly-zero flat regions finite.
        figures.show_gray(ax, np.log10(field + 1.0), title, cmap="magma")

    # A diverging map centred on zero, because the *sign* of R is the whole
    # classification: red positive is a corner, blue negative is an edge.
    # The limit is the 99.5th percentile of |R|, not its maximum. R is fourth
    # order in the gradient, so the handful of strongest corners are two orders
    # of magnitude above everything else and a max-scaled colour map renders the
    # entire image as blank paper with four dots on it.
    limit = float(np.percentile(np.abs(b_r), 99.5))
    figures.show_gray(axes[1, 0], b_r, r"$R = \det M - k\,(\mathrm{tr}\,M)^2$",
                      cmap="RdBu_r", vmin=-limit, vmax=limit)

    figures.show_gray(
        axes[1, 1], labels, "classified from the eigenvalues",
        cmap=ListedColormap(["#f2f2f2", "#3b6ea5", "#c1272d"]), vmin=0, vmax=2,
    )
    axes[1, 1].set_xlabel("grey = flat, blue = edge, red = corner")

    # The scatter is the panel that explains the other five. Every pixel is a
    # point in (lambda2, lambda1) space, and the Harris decision boundary is the
    # curve R = 0, which in eigenvalue coordinates is l1*l2 = k*(l1+l2)^2.
    ax = axes[1, 2]
    ax.grid(True)
    step = 1  # the whole field; flat pixels pile up at the origin, which is the point
    x = b_lam2[::step, ::step].ravel()
    y = b_lam1[::step, ::step].ravel()
    lab = labels[::step, ::step].ravel()
    for value, colour, name in ((0, "#bbbbbb", "flat"), (1, "#3b6ea5", "edge"), (2, "#c1272d", "corner")):
        sel = lab == value
        ax.scatter(x[sel] + 1, y[sel] + 1, s=1.5, c=colour, label=name, alpha=0.35, linewidths=0)
    lo, hi = 1.0, float(b_lam1.max()) * 2
    grid = np.logspace(0, np.log10(hi), 300)
    # The R = 0 boundary, in closed form. R = 0 means l1*l2 = k*(l1 + l2)**2;
    # divide through by l2**2 and write r = l1/l2 to get k*r**2 + (2k - 1)*r + k = 0,
    # whose upper root is the largest eigenvalue ratio that still counts as a
    # corner. It is real only while 1 - 4k >= 0, which is the algebraic reason
    # k >= 0.25 switches the detector off on every image, forever.
    ratio = ((1 - 2 * harris.K) + np.sqrt(max(1 - 4 * harris.K, 0.0))) / (2 * harris.K)
    ax.plot(grid, grid * ratio, color="#111111", lw=1.2, ls="--",
            label=f"R = 0  ($\\lambda_1/\\lambda_2$ = {ratio:.1f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\lambda_2 + 1$")
    ax.set_ylabel(r"$\lambda_1 + 1$")
    ax.set_title("every pixel, in eigenvalue space")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

    fig.suptitle(
        "Harris from the structure tensor: two eigenvalues, one response, and the k = 0.04 boundary",
        fontsize=11,
    )
    path = figures.save(fig, "01_corner_response.png")
    print(f"\nwrote {path}")
    print(f"k = {harris.K} accepts eigenvalue ratios up to {ratio:.1f}:1 -- and k >= 0.25 makes")
    print("R <= 0 on every image, forever, because r/(1+r)^2 peaks at exactly 0.25 when r = 1.")
    for k in (0.04, 0.20, 0.24, 0.25, 0.30):
        positives = int((cv2.cornerHarris(square, harris.BLOCK, harris.KSIZE, k) > 0).sum())
        print(f"  k = {k:.2f} -> pixels with R > 0: {positives:>4}")


if __name__ == "__main__":
    main()
