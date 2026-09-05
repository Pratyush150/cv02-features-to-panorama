"""03 -- Scale space: a difference-of-Gaussian pyramid, and where to stop building.

Example 02 showed the failure. This one builds the fix, up to a point, and is
explicit about where that point is.

**Implemented here (SIFT stage 1):** the Gaussian ladder, the difference of
Gaussians between neighbouring levels, and the 26-neighbour extremum search that
finds a keypoint *and* the scale at which it is most distinctive.

**Called from OpenCV (SIFT stages 2 to 4):** sub-pixel quadratic refinement,
low-contrast and edge rejection, orientation assignment, and the 128-D
descriptor. Those stages are where SIFT's accuracy actually lives, they are
fiddly, and a hand-rolled version would be slower and worse with nothing gained.

The split is the engineering point. Build the stage that makes the *idea*
concrete -- "scale invariance comes from the way the point was found, not from a
correction applied afterwards" -- and call the library for the stages where the
only thing at stake is quality of implementation.

Run:  python examples/03_scale_space.py
Figure: docs/figures/03_scale_space.png
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  -- puts src/ on sys.path; see the file for why

import cv2
import numpy as np
from matplotlib import pyplot as plt

from feat import figures, harris, scalespace, scenes

SCALES = [1.0, 0.8, 0.65, 0.5, 0.4, 0.3]
TOL_PX = 3.0

# Chosen so the bare DoG extremum test returns about 600 keypoints on the test
# scene -- the same budget the SIFT and ORB calls below are given. Comparing
# 2200 unfiltered candidates against 600 filtered keypoints would make our
# detector look worse for a reason that has nothing to do with scale: a
# repeatability rate has the candidate count in its denominator, so the method
# that emits more junk loses automatically. Match the budget, then compare.
CONTRAST = 10.0


def dog_points(gray: np.ndarray, contrast: float = CONTRAST) -> np.ndarray:
    """Every DoG extremum in the pyramid, as (x, y) in full-image coordinates."""
    pyramid = scalespace.build_pyramid(gray, n_octaves=4)
    found = [scalespace.dog_extrema(octave, contrast) for octave in pyramid]
    found = [f for f in found if len(f)]
    if not found:
        return np.zeros((0, 2), np.float32)
    return np.concatenate(found)[:, :2].astype(np.float32)


def repeatability(base: np.ndarray, other: np.ndarray, factor: float) -> float:
    if len(base) == 0 or len(other) == 0:
        return 0.0
    d = np.linalg.norm(base[:, None, :] - (other / factor)[None, :, :], axis=2)
    return float((d.min(axis=1) < TOL_PX).mean())


def main() -> None:
    figures.use_teaching_style()

    scene = cv2.cvtColor(scenes.textured_wall(600, 800, seed=5), cv2.COLOR_BGR2GRAY)
    pyramid = scalespace.build_pyramid(scene, n_octaves=4)

    print("the pyramid, octave by octave")
    print(f"{'octave':>7}{'downsample':>12}{'image':>14}{'sigmas':>10}{'DoG layers':>12}"
          f"{'searchable':>12}{'extrema':>9}")
    total = 0
    for index, octave in enumerate(pyramid):
        found = scalespace.dog_extrema(octave, contrast=CONTRAST)
        total += len(found)
        print(f"{index:>7}{octave.downsample:>12}{str(octave.blurred[0].shape[::-1]):>14}"
              f"{len(octave.sigmas):>10}{len(octave.dog):>12}{len(octave.dog) - 2:>12}{len(found):>9}")
    print(f"{'':>7}{'':>12}{'':>14}{'':>10}{'':>12}{'total':>12}{total:>9}")
    print("\nN blurred images give N-1 DoG layers, and the extremum test needs a layer above and")
    print("below the one being searched, so N-3 are searchable. Ask for 3 usable layers and you")
    print("must build 6 -- the bookkeeping everybody gets wrong once.\n")

    # Repeatability, on the same images and the same measure as example 02, so
    # the two tables can be read side by side.
    base_dog = dog_points(scene)
    base_sift = np.float32([kp.pt for kp in cv2.SIFT_create(600).detect(scene, None)]).reshape(-1, 2)
    base_orb = np.float32([kp.pt for kp in cv2.ORB_create(600).detect(scene, None)]).reshape(-1, 2)
    base_har = harris.peaks_component_nms(harris.harris_response(scene.astype(np.float32)), 0.05)

    print(f"{'scale':>7}{'DoG (ours)':>12}{'SIFT':>9}{'ORB':>9}{'Harris':>9}")
    series: dict[str, list[float]] = {"DoG (ours)": [], "SIFT": [], "ORB": [], "Harris": []}
    for factor in SCALES:
        _, small, _ = scenes.scale_pair(scene, factor)
        values = {
            "DoG (ours)": repeatability(base_dog, dog_points(small), factor),
            "SIFT": repeatability(
                base_sift,
                np.float32([kp.pt for kp in cv2.SIFT_create(600).detect(small, None)]).reshape(-1, 2),
                factor,
            ),
            "ORB": repeatability(
                base_orb,
                np.float32([kp.pt for kp in cv2.ORB_create(600).detect(small, None)]).reshape(-1, 2),
                factor,
            ),
            "Harris": repeatability(
                base_har,
                harris.peaks_component_nms(harris.harris_response(small.astype(np.float32)), 0.05),
                factor,
            ),
        }
        for key, value in values.items():
            series[key].append(value)
        print(f"{factor:>7.2f}" + "".join(f"{100 * values[k]:>8.1f}%" for k in
                                          ("DoG (ours)", "SIFT", "ORB", "Harris")))

    print("\nRead the two ends of the Harris and DoG columns against each other, because the")
    print("crossover is the honest result. At a mild 0.8x our bare DoG extrema are WORSE than")
    print("Harris -- they have no sub-pixel refinement, so a point that is genuinely re-detected")
    print("often lands more than 3 px from where it should. At 0.3x they are twice as good,")
    print("because by then Harris is not finding the structure at all and a pyramid is.")
    print("The pyramid buys large scale changes; it does not buy accuracy, and stage 1 alone")
    print("is not SIFT. SIFT beats both everywhere, and the gap between it and our column is")
    print("exactly what stages 2 to 4 -- sub-pixel fitting, low-contrast and edge rejection --")
    print("are worth. ORB does best on this particular scene: it is FAST corners ranked by")
    print("Harris over its own pyramid, and this synthetic wall is all high-contrast corners.\n")

    # ---- the figure -------------------------------------------------------
    fig = plt.figure(figsize=(12, 7.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)

    octave = pyramid[0]
    for i in range(4):
        ax = fig.add_subplot(grid[0, i])
        layer = octave.dog[i]
        limit = float(np.percentile(np.abs(layer), 99.5))
        figures.show_gray(
            ax, layer,
            f"DoG {i}:  $\\sigma$ {octave.sigmas[i]:.2f} $\\to$ {octave.sigmas[i + 1]:.2f}",
            cmap="RdBu_r", vmin=-limit, vmax=limit,
        )
    ax = fig.add_subplot(grid[1, 0:2])
    figures.show_gray(ax, scene, "")
    extrema = np.concatenate([scalespace.dog_extrema(o, CONTRAST) for o in pyramid])
    # The circle radius is the keypoint's own sigma, so the picture shows the
    # scale each point was found at -- which is the entire content of "scale
    # invariance comes from how the point was found".
    for x, y, sigma, _ in extrema[:: max(1, len(extrema) // 400)]:
        ax.add_patch(plt.Circle((x, y), sigma * 1.5, fill=False, color="#c1272d", lw=0.6, alpha=0.8))
    ax.set_title(f"our DoG extrema ({len(extrema)} total), circle radius = the sigma each was found at")

    ax = fig.add_subplot(grid[1, 2:4])
    for name, colour, marker in (
        ("DoG (ours)", "#7b3294", "^"),
        ("SIFT", "#3b6ea5", "s"),
        ("ORB", "#1a9641", "d"),
        ("Harris", "#c1272d", "o"),
    ):
        ax.plot(SCALES, [100 * v for v in series[name]], marker + "-", color=colour, label=name, ms=4)
    ax.invert_xaxis()
    ax.set_xlabel("scale factor applied to the second image")
    ax.set_ylabel("% re-detected within 3 px")
    ax.set_title("repeatability across scale, same images for all four")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left", fontsize=8)

    fig.suptitle(
        "Scale space: build stage 1 to understand it, call OpenCV for stages 2 to 4\n"
        "top row -- octave 0, where each DoG layer keeps only the detail that lived between two blur levels",
        fontsize=11,
    )
    print(f"wrote {figures.save(fig, '03_scale_space.png')}")


if __name__ == "__main__":
    main()
