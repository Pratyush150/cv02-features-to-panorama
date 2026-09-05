"""05 -- Matching, and why raw matching lies.

Nearest-neighbour search always returns an answer and never returns "I do not
know". On a brick wall that guarantee is the entire problem: forty near-identical
patches produce forty near-identical descriptors, one of them wins by sensor
noise, and the matcher reports it with total confidence.

Three things happen here:

1. The brute-force matcher is written out in NumPy and checked against
   ``cv2.BFMatcher`` -- same neighbours, same distances.
2. The ratio test and cross-check are measured on two scenes, one distinctive
   and one repetitive, by precision **and** recall.
3. The many-to-one collision the ratio test cannot see is printed by name.

Run:  python examples/05_matching_lies.py
Figure: docs/figures/05_matching_lies.png
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  -- puts src/ on sys.path; see the file for why

import collections

import cv2
import numpy as np
from matplotlib import pyplot as plt

from feat import describe, figures, matching, scenes

RATIOS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]


def verify_against_opencv(des_a: np.ndarray, des_b: np.ndarray) -> None:
    """Our nine-line NumPy matcher against OpenCV's, on the same descriptors."""
    idx, dist = matching.knn_l2_numpy(des_a[:300], des_b[:300], k=2)
    reference = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a[:300], des_b[:300], k=2)
    ref_idx = np.array([[m.trainIdx for m in pair] for pair in reference])
    ref_dist = np.array([[m.distance for m in pair] for pair in reference])

    same_first = float((idx[:, 0] == ref_idx[:, 0]).mean())
    max_dist_error = float(np.abs(dist - ref_dist).max())
    print("brute force, written out in numpy, against cv2.BFMatcher (300 x 300 SIFT descriptors)")
    print(f"  identical nearest neighbour on : {100 * same_first:.1f}% of queries")
    print(f"  largest distance disagreement  : {max_dist_error:.3e}")
    print("  (the expansion ||a-b||^2 = a.a - 2a.b + b.b turns the whole matcher into one")
    print("   matrix multiply, so no N x M x 128 array is ever allocated)\n")


def run_scene(label: str, scene: np.ndarray) -> dict:
    """Every filter, scored by precision and recall, on one scene."""
    pair = scenes.two_views(scene)
    feat_q = describe.detect_describe(pair.img_b, "sift", 800)  # query = B, so H_true maps B -> A
    feat_t = describe.detect_describe(pair.img_a, "sift", 800)
    matchable = matching.count_matchable(feat_q.points, feat_t.points, pair.H_true)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(feat_q.descriptors, feat_t.descriptors, k=2)

    def grade(name, ms):
        return matching.score_matches(
            name, ms, feat_q.points, feat_t.points, pair.H_true, matchable=matchable
        )

    raw = [p[0] for p in knn if len(p) == 2]
    results = {
        "raw nearest neighbour": grade("raw nearest neighbour", raw),
        "cross-check alone": grade("cross-check alone",
                                   matching.cross_check(raw, feat_q.descriptors, feat_t.descriptors)),
    }
    sweep = {}
    for ratio in RATIOS:
        kept = matching.ratio_test(knn, ratio)
        sweep[ratio] = grade(f"ratio test @ {ratio:.2f}", kept)
    results["ratio test @ 0.75"] = sweep[0.75]
    results["ratio 0.75 + cross-check"] = grade(
        "ratio 0.75 + cross-check",
        matching.cross_check(matching.ratio_test(knn, 0.75), feat_q.descriptors, feat_t.descriptors),
    )

    print(f"--- {label} ---")
    print(f"keypoints: query {len(feat_q.keypoints)}, train {len(feat_t.keypoints)}, "
          f"of which {matchable} query points have a true partner\n")
    for score in results.values():
        print(f"  {score}")
    print()
    for ratio in RATIOS:
        print(f"  {sweep[ratio]}")

    # The many-to-one collision. The ratio test asks its question once per
    # query and has no way to notice that six different queries all chose the
    # same train keypoint -- at most one of the six can be right.
    good = matching.ratio_test(knn, 0.75)
    claims = collections.Counter(m.trainIdx for m in good)
    worst = claims.most_common(3)
    print("\n  train keypoints claimed by more than one query, after ratio 0.75:")
    for train_idx, count in worst:
        x, y = feat_t.points[train_idx]
        print(f"    train #{train_idx} at ({x:6.1f}, {y:6.1f}) is claimed by {count} query keypoints")
    print(f"    -> {sum(c - 1 for c in claims.values() if c > 1)} matches are provably wrong on "
          "counting alone, and only cross-check can see it\n")

    return {
        "pair": pair, "feat_q": feat_q, "feat_t": feat_t, "knn": knn,
        "results": results, "sweep": sweep, "raw": raw, "good": good,
    }


def draw(pair, feat_q, feat_t, matches, colour) -> np.ndarray:
    return cv2.drawMatches(
        pair.img_b, feat_q.keypoints, pair.img_a, feat_t.keypoints, matches, None,
        matchColor=colour, singlePointColor=(160, 160, 160),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def main() -> None:
    figures.use_teaching_style()

    clutter = run_scene("distinctive clutter (the easy scene)", scenes.textured_wall())
    verify_against_opencv(clutter["feat_q"].descriptors, clutter["feat_t"].descriptors)
    bricks = run_scene("repeated brick wall (the scene that lies)", scenes.brick_wall())

    print("Read the two blocks against each other. On the clutter the ratio test is exactly the")
    print("hero it is advertised as. On the brick wall it raises precision and destroys recall --")
    print("it is refusing to guess between forty identical bricks, which is correct behaviour and")
    print("is also not a matcher you can hand to RANSAC. A working matcher reporting an ambiguous")
    print("scene looks identical, from outside, to a broken matcher. The number that tells them")
    print("apart is the absolute distance d1, and almost nobody prints it:")
    for label, data in (("clutter", clutter), ("bricks", bricks)):
        d1 = np.array([p[0].distance for p in data["knn"] if len(p) == 2])
        d2 = np.array([p[1].distance for p in data["knn"] if len(p) == 2])
        print(f"  {label:<8} median d1 = {np.median(d1):6.1f}   median d1/d2 = "
              f"{np.median(d1 / np.maximum(d2, 1e-9)):.3f}")
    print("  Similar d1 with a ratio near 1.0 means the descriptors are matching beautifully -- to")
    print("  the wrong brick. That is a scene problem, and loosening the ratio cannot fix it.\n")

    # ---- the figure -------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    raw_vis = draw(bricks["pair"], bricks["feat_q"], bricks["feat_t"],
                   bricks["raw"][:120], (0, 0, 255))
    figures.show_bgr(
        axes[0, 0], raw_vis,
        f"brick wall, raw nearest neighbour (120 of {bricks['results']['raw nearest neighbour'].kept} drawn)\n"
        f"precision {100 * bricks['results']['raw nearest neighbour'].precision:.1f}% "
        "-- confident, unanimous, wrong",
    )

    good_vis = draw(bricks["pair"], bricks["feat_q"], bricks["feat_t"],
                    bricks["good"], (0, 200, 0))
    figures.show_bgr(
        axes[0, 1], good_vis,
        f"the same pair after the ratio test: {bricks['results']['ratio test @ 0.75'].kept} matches left\n"
        f"precision {100 * bricks['results']['ratio test @ 0.75'].precision:.1f}%, "
        f"recall {100 * bricks['results']['ratio test @ 0.75'].recall:.1f}% "
        "-- it refuses to guess",
    )

    ax = axes[1, 0]
    for label, data, colour in (("distinctive clutter", clutter, "#3b6ea5"),
                                ("repeated brick wall", bricks, "#c1272d")):
        prec = [100 * data["sweep"][r].precision for r in RATIOS]
        rec = [100 * data["sweep"][r].recall for r in RATIOS]
        ax.plot(RATIOS, prec, "o-", color=colour, label=f"{label}: precision", ms=4)
        ax.plot(RATIOS, rec, "o--", color=colour, alpha=0.55, label=f"{label}: recall", ms=4)
    ax.axvline(0.75, color="#333333", lw=1, ls=":")
    ax.set_xlabel("Lowe ratio threshold (1.0 = no filtering)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.set_title("the ratio is a precision-versus-quantity dial,\nand which end you want depends on the scene")
    ax.legend(fontsize=7.5, loc="center left")

    ax = axes[1, 1]
    names = ["raw nearest neighbour", "ratio test @ 0.75", "cross-check alone", "ratio 0.75 + cross-check"]
    x = np.arange(len(names))
    width = 0.2
    for offset, (label, data, colour) in enumerate((
        ("clutter: precision", clutter, "#3b6ea5"),
        ("clutter: recall", clutter, "#a6c8e8"),
        ("bricks: precision", bricks, "#c1272d"),
        ("bricks: recall", bricks, "#efa9a9"),
    )):
        key = "precision" if "precision" in label else "recall"
        values = [100 * getattr(data["results"][n], key) for n in names]
        ax.bar(x + (offset - 1.5) * width, values, width, label=label, color=colour)
    ax.set_xticks(x)
    ax.set_xticklabels(["raw NN", "ratio 0.75", "cross-check", "ratio + cross"], fontsize=8)
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.set_title("the two filters catch different diseases:\nratio kills ambiguity, cross-check kills many-to-one")
    ax.legend(fontsize=7.5, ncol=2)

    fig.suptitle(
        "Raw nearest-neighbour matching always answers -- on a repeated texture it answers wrongly",
        fontsize=11,
    )
    print(f"wrote {figures.save(fig, '05_matching_lies.png')}")


if __name__ == "__main__":
    main()
