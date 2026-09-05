"""04 -- Descriptors: what they must be invariant to, and what each one costs.

Two descriptors, opposite ends of one trade-off, measured rather than quoted:

* SIFT -- 128 float32 values (512 bytes), compared with L2.
* ORB  -- 256 bits (32 bytes, exactly 16x smaller), compared with Hamming
  distance, which is four XORs and four popcounts on 64-bit words.

The figure shows what each one physically *is*, what its distance distribution
looks like for true matches against random pairs, how long a match costs on this
CPU, and how much rotation each survives.

Run:  python examples/04_descriptors.py
Figure: docs/figures/04_descriptors.png
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  -- puts src/ on sys.path; see the file for why

import cv2
import numpy as np
from matplotlib import pyplot as plt

from feat import describe, figures, matching, scenes

ANGLES = [0, 10, 20, 30, 45, 60, 90]


def rotate(img: np.ndarray, degrees: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate about the image centre, and return the exact 3x3 that did it.

    The canvas is not resized, so corners rotate out of view. That is fine and
    deliberate: keypoints that leave the frame are not "matching failures", and
    the recall denominator below counts only keypoints whose partner is still
    inside the image.
    """
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    rotated = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR)
    homography = np.vstack([m, [0.0, 0.0, 1.0]])  # an affine IS a homography with a zero bottom row
    return rotated, homography


def main() -> None:
    figures.use_teaching_style()

    scene = scenes.textured_wall(600, 800, seed=5)
    pair = scenes.two_views(scene)

    sift_a = describe.detect_describe(pair.img_b, "sift", 800)
    sift_b = describe.detect_describe(pair.img_a, "sift", 800)
    orb_a = describe.detect_describe(pair.img_b, "orb", 800)
    orb_b = describe.detect_describe(pair.img_a, "orb", 800)

    print("what each descriptor physically is")
    for feat in (sift_a, orb_a):
        facts = describe.descriptor_facts(feat)
        for key, value in facts.items():
            print(f"  {feat.name:<5} {key:<34} {value}")
        print()

    # The 128 numbers are a 4x4 grid of cells, each an 8-bin orientation
    # histogram, concatenated in reading order. So entry (row r, col c, bin b)
    # is at index (r*4 + c)*8 + b. Printing both directions is the check: if the
    # flat index and the reshaped index disagree, the layout in your head is
    # wrong, and every hand-rolled descriptor you write after that is wrong too.
    one = sift_a.descriptors[0]
    r, c, b = 2, 3, 5
    index = (r * 4 + c) * 8 + b
    print(f"SIFT layout: cell (row {r}, col {c}), bin {b} is entry (({r}*4 + {c})*8 + {b}) = {index}")
    print(f"  flat des[{index}]              = {one[index]:.4f}")
    print(f"  reshaped des.reshape(4,4,8)[{r},{c},{b}] = {one.reshape(4, 4, 8)[r, c, b]:.4f}")
    print(f"  and backwards: {index} // 8 = {index // 8} (cell), {index} % 8 = {index % 8} (bin); "
          f"cell {index // 8} is row {index // 8 // 4}, col {index // 8 % 4}")
    print(f"  L2 norm of this descriptor  = {np.linalg.norm(one):.2f}  "
          "(OpenCV normalises to 1 and then multiplies by 512)")

    # Hamming by hand on two real descriptors, checked against OpenCV.
    d0, d1 = orb_a.descriptors[0], orb_a.descriptors[1]
    by_hand = int(np.unpackbits(np.bitwise_xor(d0, d1)).sum())
    by_cv2 = int(cv2.norm(d0.reshape(1, -1), d1.reshape(1, -1), cv2.NORM_HAMMING))
    print(f"\nORB Hamming, first two descriptors: XOR then popcount = {by_hand}, "
          f"cv2.norm(NORM_HAMMING) = {by_cv2}, agree = {by_hand == by_cv2}")
    print(f"  first byte: {d0[0]} = {d0[0]:08b}")
    print(f"              {d1[0]} = {d1[0]:08b}")
    print(f"         XOR: {int(d0[0]) ^ int(d1[0])} = {int(d0[0]) ^ int(d1[0]):08b}  "
          f"-> {bin(int(d0[0]) ^ int(d1[0])).count('1')} differing bits in this byte alone")

    # Distances: true matches against random pairs, for both descriptors. The
    # random baseline is the panel that makes the true-match histogram mean
    # something -- "small distance" is only small relative to what unrelated
    # descriptors score.
    rng = np.random.default_rng(0)
    dist_data = {}
    for name, fa, fb in (("SIFT", sift_a, sift_b), ("ORB", orb_a, orb_b)):
        norm = describe.metric_for(fa.descriptors)
        knn = cv2.BFMatcher(norm).knnMatch(fa.descriptors, fb.descriptors, k=2)
        good = matching.ratio_test(knn, 0.75)
        score = matching.score_matches(name, good, fa.points, fb.points, pair.H_true)
        true_d = np.array([m.distance for m in good])
        idx_a = rng.integers(0, len(fa.descriptors), 4000)
        idx_b = rng.integers(0, len(fb.descriptors), 4000)
        if norm == cv2.NORM_HAMMING:
            random_d = np.unpackbits(
                np.bitwise_xor(fa.descriptors[idx_a], fb.descriptors[idx_b]), axis=1
            ).sum(axis=1).astype(float)
        else:
            random_d = np.linalg.norm(
                fa.descriptors[idx_a].astype(np.float64) - fb.descriptors[idx_b].astype(np.float64),
                axis=1,
            )
        dist_data[name] = (true_d, random_d)
        print(f"\n{score}")
        print(f"  true-match distance   median {np.median(true_d):8.1f}")
        print(f"  random-pair distance  median {np.median(random_d):8.1f}   "
              f"separation {np.median(random_d) / max(np.median(true_d), 1e-9):.1f}x")

    # Timing. Per descriptor *pair*, so the two detectors' differing keypoint
    # counts cannot flatter either of them.
    print("\nmatching cost on this CPU (minimum of 5 runs, knnMatch k=2)")
    timing = {}
    for name, fa, fb in (("SIFT", sift_a, sift_b), ("ORB", orb_a, orb_b)):
        seconds, per_pair = describe.time_knn_match(fa.descriptors, fb.descriptors, repeats=5)
        timing[name] = per_pair
        print(f"  {name:<5} {len(fa.descriptors):>5} x {len(fb.descriptors):>5} descriptors "
              f"-> {seconds * 1000:7.2f} ms   {per_pair:7.2f} ns per pair")
    print(f"  ORB is {timing['SIFT'] / timing['ORB']:.1f}x cheaper per descriptor pair here, "
          f"on {32} bytes against {512}.")

    # Rotation invariance, measured. The claim is not "SIFT is rotation
    # invariant"; the claim is a curve, and a raw pixel patch is the control
    # that shows what having no invariance actually looks like.
    print(f"\nrotation invariance: precision of the ratio-tested matches\n"
          f"{'degrees':>8}{'SIFT':>10}{'ORB':>10}{'raw 16x16 patch':>18}")
    grey = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    curves = {"SIFT": [], "ORB": [], "raw patch": []}
    for degrees in ANGLES:
        rotated, homography = rotate(grey, degrees)
        row = {}
        for name, method in (("SIFT", "sift"), ("ORB", "orb")):
            fa = describe.detect_describe(grey, method, 800)
            fb = describe.detect_describe(rotated, method, 800)
            knn = cv2.BFMatcher(describe.metric_for(fa.descriptors)).knnMatch(
                fa.descriptors, fb.descriptors, k=2
            )
            score = matching.score_matches(
                name, matching.ratio_test(knn, 0.75), fa.points, fb.points, homography
            )
            row[name] = score.precision

        # The control: the raw 16x16 patch around each SIFT keypoint, flattened.
        # It has a location and a scale and no orientation handling at all, so
        # it should collapse the moment the image turns -- and if it does not,
        # the experiment is not measuring rotation.
        fa = describe.detect_describe(grey, "sift", 400)
        fb = describe.detect_describe(rotated, "sift", 400)

        def patches(img, feat):
            out, keep = [], []
            for i, kp in enumerate(feat.keypoints):
                x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
                if 8 <= x < img.shape[1] - 8 and 8 <= y < img.shape[0] - 8:
                    out.append(img[y - 8:y + 8, x - 8:x + 8].astype(np.float32).ravel())
                    keep.append(i)
            return np.array(out, np.float32), np.array(keep, int)

        pa, ka = patches(grey, fa)
        pb, kb = patches(rotated, fb)
        knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(pa, pb, k=2)
        kept = matching.ratio_test(knn, 0.75)
        # Remap indices back to the full keypoint arrays before scoring, since
        # border keypoints were dropped above and the indices no longer line up.
        for m in kept:
            m.queryIdx, m.trainIdx = int(ka[m.queryIdx]), int(kb[m.trainIdx])
        row["raw patch"] = matching.score_matches(
            "raw", kept, fa.points, fb.points, homography
        ).precision

        for key in curves:
            curves[key].append(row[key])
        print(f"{degrees:>8}{100 * row['SIFT']:>9.1f}%{100 * row['ORB']:>9.1f}%"
              f"{100 * row['raw patch']:>17.1f}%")

    # ---- the figure -------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7), constrained_layout=True)

    bits = np.unpackbits(orb_a.descriptors[:48], axis=1)
    figures.show_gray(axes[0, 0], bits, "ORB: 48 descriptors, 256 bits each", cmap="binary")
    axes[0, 0].set_xlabel("256 brightness comparisons -> 32 bytes")

    cells = one.reshape(4, 4, 8)
    ax = axes[0, 1]
    ax.grid(False)
    for row_i in range(4):
        for col_i in range(4):
            base_x, base_y = col_i * 9, (3 - row_i) * 1.15
            ax.bar(base_x + np.arange(8), cells[row_i, col_i] / 512.0, bottom=base_y,
                   width=0.85, color="#3b6ea5")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("SIFT: one keypoint, 4x4 cells x 8 bins")
    ax.set_xlabel(f"entry {index} = cell (row {r}, col {c}), bin {b}")

    for ax, name, colour in ((axes[0, 2], "ORB", "#1a9641"), (axes[1, 0], "SIFT", "#3b6ea5")):
        true_d, random_d = dist_data[name]
        ax.hist(random_d, bins=50, color="#bbbbbb", label="random pairs", density=True)
        ax.hist(true_d, bins=50, color=colour, alpha=0.85, label="ratio-test survivors", density=True)
        ax.set_title(f"{name} distances "
                     f"({'Hamming, 0..256' if name == 'ORB' else 'L2 over ~512-norm vectors'})")
        ax.set_xlabel("DMatch.distance")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

    ax = axes[1, 1]
    names = list(timing)
    ax.bar(names, [timing[n] for n in names], color=["#3b6ea5", "#1a9641"], width=0.55)
    for i, n in enumerate(names):
        ax.text(i, timing[n], f"{timing[n]:.1f} ns", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("ns per descriptor pair (min of 5 runs)")
    ax.set_title(f"brute-force match cost on this CPU\nORB is {timing['SIFT'] / timing['ORB']:.1f}x cheaper")

    ax = axes[1, 2]
    for name, colour, marker in (("SIFT", "#3b6ea5", "s"), ("ORB", "#1a9641", "d"),
                                 ("raw patch", "#c1272d", "o")):
        ax.plot(ANGLES, [100 * v for v in curves[name]], marker + "-", color=colour, label=name, ms=4)
    ax.set_xlabel("in-plane rotation (degrees)")
    ax.set_ylabel("precision of ratio-tested matches")
    ax.set_ylim(0, 100)
    ax.set_title("rotation invariance, with a control")
    ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        "Two descriptors, one trade: 512 bytes and L2 against 32 bytes and Hamming",
        fontsize=11,
    )
    print(f"\nwrote {figures.save(fig, '04_descriptors.png')}")


if __name__ == "__main__":
    main()
