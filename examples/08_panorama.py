"""08 -- The panorama: warp, composite, and the difference between a seam and a blend.

Everything in this repository, connected end to end:

    detect (01) -> describe (04) -> match + ratio test (05) -> RANSAC (07)
      -> warp -> composite

Two things are deliberately wrong with the input, because both are things real
shots do and both fail in ways worth recognising:

* an **intruder** -- one vivid object pasted at two irreconcilable positions,
  the synthetic version of somebody walking through the shot. It is bright and
  unique, so the ratio test waves every match on it through. Only geometry can
  reject it.
* an **exposure difference** -- the second shot is 22% brighter, as auto-exposure
  would make it. RANSAC does not care, and the compositor does: it is what turns
  the seam into a visible step.

Estimating H and painting pixels are two different steps. The corner error says
whether the first one worked; the seam profile says whether the second one did.

Run:  python examples/08_panorama.py
Figure: docs/figures/08_panorama.png
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  -- puts src/ on sys.path; see the file for why

import cv2
import numpy as np
from matplotlib import pyplot as plt

from feat import describe, figures, matching, panorama, ransac, scenes

SIGMA_PX = 1.0  # keypoint localisation noise we assume for these detections
INTRUDER = 72   # side of the pasted object, in pixels


def main() -> None:
    figures.use_teaching_style()

    pair = scenes.two_views(scenes.textured_wall())
    base = scenes.paste_intruder(pair.img_a, 430, 70, INTRUDER)   # the object, here in shot A
    other = scenes.paste_intruder(pair.img_b, 60, 320, INTRUDER)  # ...and somewhere impossible in B
    # Auto-exposure between the two frames. Multiplicative plus a small offset,
    # which is what a gain and black-level change actually do -- an additive-only
    # difference would leave every gradient untouched and the seam would be far
    # less visible than a real one.
    other = np.clip(other.astype(np.float32) * 1.22 + 6, 0, 255).astype(np.uint8)

    feat_q = describe.detect_describe(other, "sift")  # query = the image being warped
    feat_t = describe.detect_describe(base, "sift")   # train = the base frame
    print(f"keypoints   base {len(feat_t.keypoints)}   other {len(feat_q.keypoints)}")

    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(feat_q.descriptors, feat_t.descriptors, k=2)
    good = matching.ratio_test(knn, 0.75)
    print(f"ratio-test survivors: {len(good)}")
    assert len(good) >= 4, "need at least 4 correspondences to fit a homography"

    src, dst = matching.matched_points(good, feat_q.points, feat_t.points)
    threshold = ransac.inlier_threshold(SIGMA_PX, dof=2, confidence=0.95)
    result = ransac.ransac_homography(src.reshape(-1, 2), dst.reshape(-1, 2), threshold, seed=0)
    inliers = result.inliers
    print(f"RANSAC threshold {threshold:.3f} px (2 DoF, 95%, sigma = {SIGMA_PX} px), "
          f"{result.iterations} samples drawn")
    print(f"inliers: {int(inliers.sum())} / {len(good)}      outliers: {int((~inliers).sum())}   "
          f"inlier ratio {100 * result.inlier_ratio:.1f}%")

    corners = np.float32([[0, 0], [other.shape[1], 0],
                          [other.shape[1], other.shape[0]], [0, other.shape[0]]])
    err = float(np.abs(ransac.transfer(result.H, corners)
                       - ransac.transfer(pair.H_true, corners)).max())
    h_cv, mask_cv = cv2.findHomography(src, dst, cv2.RANSAC, threshold)
    err_cv = float(np.abs(ransac.transfer(h_cv / h_cv[2, 2], corners)
                          - ransac.transfer(pair.H_true, corners)).max())
    print(f"max corner error vs the known H:  ours {err:.3f} px   "
          f"cv2.findHomography {err_cv:.3f} px  ({int(mask_cv.sum())} inliers)")

    # Where did the outliers come from? The intruder occupies a known box in
    # the query image, so this is checkable rather than a guess from the picture.
    qx, qy = src[:, 0, 0], src[:, 0, 1]
    on_intruder = (qx >= 60) & (qx < 60 + INTRUDER) & (qy >= 320) & (qy < 320 + INTRUDER)
    print(f"matches on the intruder: {int(on_intruder.sum())}, "
          f"rejected by RANSAC: {int((on_intruder & ~inliers).sum())}")
    print("  Those matches are unambiguous -- the object is bright and unique, so the ratio test")
    print("  passes them. They are also all wrong. The ratio test filters ambiguity; only")
    print("  geometry can tell you that a confident, unambiguous match is impossible.\n")

    # ---- the canvas -------------------------------------------------------
    canvas = panorama.warp_onto_canvas(base, other, result.H)
    tx, ty = canvas.translation[0, 2], canvas.translation[1, 2]
    print(f"canvas sized from the transformed corners: {canvas.size[0]} x {canvas.size[1]}, "
          f"shifted by ({tx:.0f}, {ty:.0f})")

    naive_size = (base.shape[1] + other.shape[1], max(base.shape[0], other.shape[0]))
    naive = cv2.warpPerspective(other, result.H, naive_size)
    naive[0:base.shape[0], 0:base.shape[1]] = base
    kept_naive = int((naive.sum(axis=2) > 0).sum())
    pasted = panorama.paste(canvas)
    kept_correct = int((pasted.sum(axis=2) > 0).sum())
    print(f"naive canvas   {naive_size[0]:>4} x {naive_size[1]:<4} keeps {kept_naive:>7} image pixels")
    print(f"correct canvas {canvas.size[0]:>4} x {canvas.size[1]:<4} keeps {kept_correct:>7} image pixels"
          f"   ({kept_correct - kept_naive:+d} against the naive crop)")
    print("  The warped image sticks out above row zero because the camera pivoted. The naive")
    print("  canvas silently crops those rows, and a panorama with a straight-cut corner looks")
    print("  deliberate -- which is what makes guessing the canvas size a dangerous bug.\n")

    # ---- estimation health, separately from compositing health ------------
    #
    # A ladder, because "the overlap disagrees" has three different causes with
    # three different fixes, and one combined quality score cannot tell them
    # apart. Each rung adds exactly one defect to the same pair of images and
    # the same homography, so whatever moves is caused by the thing just added.
    clean_pair = scenes.two_views(scenes.textured_wall())
    exposed = np.clip(clean_pair.img_b.astype(np.float32) * 1.22 + 6, 0, 255).astype(np.uint8)
    rungs = [
        ("clean, our H", panorama.warp_onto_canvas(clean_pair.img_a, clean_pair.img_b, result.H)),
        ("clean, known H_true",
         panorama.warp_onto_canvas(clean_pair.img_a, clean_pair.img_b, clean_pair.H_true)),
        ("+ exposure difference", panorama.warp_onto_canvas(clean_pair.img_a, exposed, result.H)),
        ("+ the intruder too", canvas),
    ]
    print("overlap agreement -- the GEOMETRY, measured through the pixels")
    print(f"  {'':<24}{'overlap px':>11}{'NCC':>9}{'mean|diff|':>12}{'>40 apart':>11}"
          f"{'largest blob':>14}")
    for name, c in rungs:
        stats = panorama.overlap_agreement(c)
        grey = lambda a: a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)  # noqa: E731
        bad = ((cv2.absdiff(grey(c.base), grey(c.other)) > 40) & c.overlap).astype(np.uint8)
        areas = cv2.connectedComponentsWithStats(bad, 8)[2][1:, 4]
        print(f"  {name:<24}{stats['overlap_px']:>11}{stats['ncc']:>9.4f}"
              f"{stats['mean_abs_diff']:>12.2f}{100 * stats['frac_disagree']:>10.2f}%"
              f"{int(areas.max()) if len(areas) else 0:>14}")
    print("  Row 1 against row 2: our homography agrees with the true one to four decimal places")
    print("  in NCC, so the ESTIMATOR is not the problem in any row below.")
    print("  Row 3: mean |diff| jumps by a factor of 25 while NCC moves in the third decimal.")
    print("  NCC is invariant to a uniform gain and mean |diff| is not, so reading the two")
    print("  together separates 'misaligned' from 'differently exposed' -- and only the second")
    print("  of those is fixed in the compositor.")
    print("  Row 4: NCC finally falls, and the largest disagreeing blob grows to roughly the")
    print("  intruder's own area. A disagreement that is blob-shaped means something MOVED; one")
    print("  that is a thin halo along every edge in the frame means the homography is wrong.\n")

    # ---- the two compositors ----------------------------------------------
    feathered = panorama.feather(canvas)
    seam_x = int(base.shape[1] + tx) - 1  # the right-hand edge of the base image
    profile_paste = panorama.seam_column_profile(pasted, seam_x)
    profile_feather = panorama.seam_column_profile(feathered, seam_x)
    step_paste = float(np.abs(np.diff(profile_paste)).max())
    step_feather = float(np.abs(np.diff(profile_feather)).max())
    print(f"seam at x = {seam_x}, measured as the largest column-to-column brightness jump:")
    print(f"  hard paste  : {step_paste:6.2f} grey levels")
    print(f"  feathered   : {step_feather:6.2f} grey levels   "
          f"({step_paste / max(step_feather, 1e-9):.1f}x smaller)")
    print("  Feathering hides an exposure step. It cannot hide a contradiction: where the two")
    print("  images genuinely disagree it paints half of each, so the intruder appears twice at")
    print("  half strength. The production fixes are a seam routed around the disagreement")
    print("  (cv2.detail_GraphCutSeamFinder) or a per-pixel median over three or more frames.")

    # ---- the figure -------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4), constrained_layout=True)

    draw = lambda ms, colour: cv2.drawMatches(  # noqa: E731
        other, feat_q.keypoints, base, feat_t.keypoints, ms, None, matchColor=colour,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    kept = [m for m, ok in zip(good, inliers) if ok]
    rejected = [m for m, ok in zip(good, inliers) if not ok]
    figures.show_bgr(axes[0, 0], draw(kept[::6], (0, 200, 0)),
                     f"inliers: {len(kept)} of {len(good)} (every 6th drawn)\n"
                     "a calm, near-parallel bundle -- that is consensus")
    figures.show_bgr(axes[0, 1], draw(rejected, (0, 0, 255)),
                     f"outliers RANSAC rejected: {len(rejected)}\n"
                     f"including all {int(on_intruder.sum())} matches on the intruder")

    figures.show_bgr(axes[0, 2], naive,
                     f"the guessed canvas {naive_size[0]}x{naive_size[1]}\n"
                     f"crops the top of the warp and pads black on the right")

    figures.show_bgr(axes[1, 0], pasted,
                     f"hard paste: seam step {step_paste:.1f} grey levels")
    axes[1, 0].axvline(seam_x, color="#ffcc00", lw=0.9, ls="--")
    figures.show_bgr(axes[1, 1], feathered,
                     f"feathered: seam step {step_feather:.1f} grey levels")
    axes[1, 1].axvline(seam_x, color="#ffcc00", lw=0.9, ls="--")

    ax = axes[1, 2]
    xs = np.arange(len(profile_paste)) + max(0, seam_x - 40)
    ax.plot(xs, profile_paste, color="#c1272d", label=f"hard paste (step {step_paste:.1f})")
    ax.plot(xs, profile_feather, color="#1a9641", label=f"feathered (step {step_feather:.1f})")
    ax.axvline(seam_x, color="#333333", ls="--", lw=1)
    ax.set_xlabel("column x (px)")
    ax.set_ylabel("mean brightness of the column")
    ax.set_title("the seam, measured instead of squinted at")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Panorama: {int(inliers.sum())}/{len(good)} inliers, {err:.2f} px corner error against the "
        f"known homography, and a {step_paste / max(step_feather, 1e-9):.0f}x smaller seam after feathering",
        fontsize=11,
    )
    print(f"\nwrote {figures.save(fig, '08_panorama.png')}")


if __name__ == "__main__":
    main()
