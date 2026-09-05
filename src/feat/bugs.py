"""Five shapes of bug, each reproduced on purpose.

Bugs come in shapes. Naming a shape is what lets you look for it, and the
right-hand column of the table below -- the ten-second detector -- is the part
that transfers to code you have never seen.

Each function here builds a deliberate, reproducible instance of one shape,
inside feature matching, and returns the measurements as a :class:`BugReport`.
Nothing is asserted in prose that is not printed as a number. Example 06 draws
all five; ``tests/test_bugs.py`` checks that each one still misbehaves in the
documented direction, which is the only way a demonstration of a bug stays
honest as the library underneath it moves.

============================ ============================================== =========================================================
Shape                        This chapter's instance                        The ten-second detector
============================ ============================================== =========================================================
1. Partial implementation    ``cv2.cornerHarris`` returns a response map    Count the algorithm's stages, then count the parameters.
                             -- no threshold, no suppression                A stage with no knob is a stage you own.
2. Hidden operator in the    ``BFMatcher(normType, crossCheck)`` -- three    Split the name into morphemes; design an input that
   name                      morphemes, three behaviours                    makes each one fire.
3. Ambiguous unit            ``DMatch.distance`` is a bit count for ORB      Read the constructor and the dtype, not the label.
                             and an L2 distance near 512 for SIFT           A number without its unit is not a number.
4. Demo with no control      "the ratio test improves matching", shown       Before running a demo, name the input that would make
                             only on a distinctive scene                    it print the opposite. If you cannot, it is not evidence.
5. Fact with an expiry date  "SIFT is in contrib" -- true until July 2020    Version-stamp the claim, then ask the interpreter.
============================ ============================================== =========================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import harris, matching, scenes
from .describe import detect_describe

__all__ = [
    "BugReport",
    "shape1_partial_implementation",
    "shape2_hidden_operator",
    "shape3_ambiguous_unit",
    "shape4_demo_without_control",
    "shape5_expired_fact",
    "all_shapes",
]


@dataclass(frozen=True)
class BugReport:
    """One reproduced bug: what shape it is, what fired, and the evidence."""

    number: int
    shape: str
    instance: str
    detector: str
    symptom: str  # how to recognise it from the symptom alone, with no source in front of you
    numbers: dict = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            f"SHAPE {self.number} -- {self.shape}",
            f"  instance : {self.instance}",
            f"  detector : {self.detector}",
            f"  symptom  : {self.symptom}",
        ]
        for key, value in self.numbers.items():
            if isinstance(value, float):
                lines.append(f"  {key:<34} {value:.4g}")
            else:
                lines.append(f"  {key:<34} {value}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Shape 1 -- the partial implementation
# --------------------------------------------------------------------------
def shape1_partial_implementation() -> BugReport:
    """``cv2.cornerHarris`` implements three stages of a five-stage algorithm.

    Harris corner detection, as an algorithm you would describe to somebody, is:
    gradients, structure tensor, response, **threshold**, **non-maximum
    suppression** (and sub-pixel refinement if you need it). ``cv2.cornerHarris``
    does the first three and hands you a picture. Its own documentation says so
    -- "Corners in the image can be found as the local maxima of this response
    map" -- in a sentence that reads as description and is actually an
    instruction.

    The detector fires immediately: ``cornerHarris(src, blockSize, ksize, k)``
    has no threshold parameter and no suppression radius. Compare
    ``goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance, ...)``,
    which has all three knobs -- because it *is* the complete pipeline.

    Measured on a checkerboard whose interior corner count we know exactly.
    """
    board = scenes.checkerboard(blur=False)  # blur=False: exact plateaus, see below
    response = harris.harris_response(board.image.astype(np.float32))

    above = int((response > 0.01 * response.max()).sum())
    naive = harris.peaks_naive_nms(response, rel_thresh=0.01)
    fixed = harris.peaks_component_nms(response, rel_thresh=0.01)

    x0, y0, x1, y1 = board.interior

    def interior(points: np.ndarray) -> int:
        if len(points) == 0:
            return 0
        x, y = points[:, 0], points[:, 1]
        return int(((x > x0) & (x < x1) & (y > y0) & (y < y1)).sum())

    # The same image with a 3x3 blur, which is all it takes to break the exact
    # ties. This is the second half of the lesson: the naive suppression looks
    # fine on any photograph, so the bug reproduces on the clean synthetic test
    # image and not on real data -- which makes it look like the test is wrong.
    blurred = scenes.checkerboard(blur=True).image
    naive_blurred = harris.peaks_naive_nms(harris.harris_response(blurred.astype(np.float32)))

    harris_params = cv2.cornerHarris.__doc__.splitlines()[0]
    gftt_params = cv2.goodFeaturesToTrack.__doc__.splitlines()[0]

    return BugReport(
        number=1,
        shape="partial implementation",
        instance="cv2.cornerHarris returns a response map; threshold and non-max suppression are yours",
        detector="count the algorithm's stages, then count the parameters -- a stage with no knob is a stage you own",
        symptom="roughly ten times too many keypoints, in tight clumps around each real corner, "
                "on synthetic or rendered images -- and it does not reproduce on noisy photos",
        numbers={
            "signature (3 of 5 stages)": harris_params,
            "the complete pipeline": gftt_params,
            "true interior corners": board.n_interior,
            "pixels above 1% of R.max": above,
            "naive R == dilate(R), all": len(naive),
            "naive R == dilate(R), interior": interior(naive),
            "component-wise NMS, all": len(fixed),
            "component-wise NMS, interior": interior(fixed),
            "naive NMS on the BLURRED board": len(naive_blurred),
        },
    )


# --------------------------------------------------------------------------
# Shape 2 -- the operator hidden in the name
# --------------------------------------------------------------------------
def shape2_hidden_operator() -> BugReport:
    """``BFMatcher(normType, crossCheck)``: three morphemes, three behaviours.

    Most people read "matcher". The name has three parts and every one of them
    changes the result:

    * **BF** -- brute force. Every query against every train descriptor, exact
      and ``O(N*M)``. Doubling both inputs quadruples the work, which is fine at
      a thousand descriptors and a coffee break at fifty thousand.
    * **norm** -- the metric, which must match the descriptor type. ``NORM_L2``
      on ORB's ``uint8`` runs **silently**: it treats each of the 32 bytes as a
      coordinate, which is a real distance on a space the descriptor does not
      live in. ``NORM_HAMMING`` on SIFT's float32 throws immediately. Only one
      of the two mistakes tells you about itself.
    * **crossCheck** -- not a setting, a *filter*. It changes what ``match()``
      returns, and it makes ``knnMatch(k=2)`` raise outright, so the flag cannot
      be combined with the ratio test.

    The detector is the one from the shape: split the name into morphemes, then
    design an input that makes each fire.
    """
    pair = scenes.two_views(scenes.textured_wall())
    orb_a = detect_describe(pair.img_b, "orb", 1200)  # B is query: H_true maps B -> A
    orb_b = detect_describe(pair.img_a, "orb", 1200)
    matchable = matching.count_matchable(orb_a.points, orb_b.points, pair.H_true)

    def score(label: str, norm: int) -> tuple[matching.MatchScore, matching.MatchScore]:
        knn = cv2.BFMatcher(norm).knnMatch(orb_a.descriptors, orb_b.descriptors, k=2)
        graded = lambda tag, ms: matching.score_matches(  # noqa: E731
            f"{label} {tag}", ms, orb_a.points, orb_b.points, pair.H_true, matchable=matchable
        )
        # Raw nearest neighbour keeps one match per query no matter what, so
        # the two metrics are compared at an *identical* kept count and only
        # the correct count can move. That isolates the metric from the filter.
        return (
            graded("raw NN", [p[0] for p in knn if len(p) == 2]),
            graded("+ratio", matching.ratio_test(knn, 0.75)),
        )

    right_raw, right_ratio = score("HAMMING (right)", cv2.NORM_HAMMING)
    wrong_raw, wrong_ratio = score("L2      (wrong)", cv2.NORM_L2)

    # morpheme "norm", the loud direction: float32 descriptors, binary metric.
    sift_a = detect_describe(pair.img_b, "sift", 300)
    sift_b = detect_describe(pair.img_a, "sift", 300)
    try:
        cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(sift_a.descriptors, sift_b.descriptors, k=2)
        hamming_on_float = "did not raise (unexpected)"
    except cv2.error as exc:
        hamming_on_float = str(exc).strip().splitlines()[0].split("error: ")[-1]

    # morpheme "crossCheck", the loud direction.
    try:
        cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).knnMatch(
            orb_a.descriptors, orb_b.descriptors, k=2
        )
        cross_knn = "did not raise (unexpected)"
    except cv2.error as exc:
        cross_knn = str(exc).strip().splitlines()[0].split("error: ")[-1]

    # morpheme "crossCheck", the quiet direction: same method name, same call,
    # a different number of results, and no indication that a filter ran.
    plain = cv2.BFMatcher(cv2.NORM_HAMMING).match(orb_a.descriptors, orb_b.descriptors)
    checked = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(
        orb_a.descriptors, orb_b.descriptors
    )

    # morpheme "BF": the cost is quadratic, so measure it at two sizes rather
    # than asserting the complexity. Minimum of three runs -- interference can
    # only ever make a timing larger, so the minimum is closest to the truth.
    def bf_seconds(n: int) -> float:
        a, b = orb_a.descriptors[:n], orb_b.descriptors[:n]
        m = cv2.BFMatcher(cv2.NORM_HAMMING)
        return min(
            (lambda t0: (m.knnMatch(a, b, k=2), time.perf_counter() - t0)[1])(time.perf_counter())
            for _ in range(3)
        )

    n_small = min(300, len(orb_a.descriptors), len(orb_b.descriptors))
    n_big = min(2 * n_small, len(orb_a.descriptors), len(orb_b.descriptors))
    t_small, t_big = bf_seconds(n_small), bf_seconds(n_big)

    return BugReport(
        number=2,
        shape="hidden operator in the name",
        instance="BFMatcher(normType, crossCheck) -- 'BF', 'norm' and 'crossCheck' each change the answer",
        detector="split the name into morphemes, then design an input that makes each one fire",
        symptom="a pipeline that 'sort of works' and gets worse on hard pairs -- the wrong metric "
                "barely moves the precision people print and halves the recall they do not",
        numbers={
            "norm, quiet: right metric": str(right_raw),
            "norm, quiet: wrong metric": str(wrong_raw),
            "norm, quiet: right + ratio": str(right_ratio),
            "norm, quiet: wrong + ratio": str(wrong_ratio),
            "norm, loud: Hamming on float32": hamming_on_float,
            "crossCheck, loud: with knnMatch": cross_knn,
            "crossCheck, quiet: match() plain": len(plain),
            "crossCheck, quiet: match() checked": len(checked),
            f"BF: {n_small}x{n_small} descriptors (s)": t_small,
            f"BF: {n_big}x{n_big} descriptors (s)": t_big,
            "BF: cost ratio for 2x the data": t_big / t_small if t_small > 0 else float("nan"),
        },
    )


# --------------------------------------------------------------------------
# Shape 3 -- the unit with two meanings
# --------------------------------------------------------------------------
def shape3_ambiguous_unit() -> BugReport:
    """``DMatch.distance`` is two different quantities wearing one name.

    For ORB it is a **Hamming bit count**: an integer in 0..256, where 0 is
    identical and about 128 is what two unrelated descriptors score, because
    half the bits of a coin flip differ. For SIFT it is an **L2 distance**
    between vectors OpenCV has scaled to a norm near 512, so it is an unbounded
    float in the hundreds.

    Carry ``if m.distance < 50`` from an ORB pipeline to a SIFT one and it stops
    meaning "a strict cut" and starts meaning "keep essentially nothing" -- or
    the other way round, depending which way you carried it. The label is the
    same in both. This is MB against MiB with a matcher in front of it.

    The punchline is the fix. Lowe's ratio test is a **ratio**, so the unit
    divides out; the identical line of code is meaningful for both descriptors.
    That does not make the two keep the *same* fraction -- they are different
    descriptors on the same scene, and they should not -- but it collapses the
    spread. Measured below: a fixed ``d1 < 50`` keeps 24 times as large a
    fraction of one detector's matches as the other's, while ``d1 < 0.75*d2``
    puts them within a factor of two. A dimensionless threshold is the only kind
    that travels between pipelines at all.

    Two more units in the same neighbourhood, measured here because they cost
    people the same afternoon: ``KeyPoint.size`` is a **diameter** (crop with it
    as a radius and every patch is twice the intended size), and
    ``KeyPoint.angle`` is in **degrees**, not radians, and is -1 when the
    detector assigned no orientation.
    """
    pair = scenes.two_views(scenes.textured_wall())
    out: dict = {}
    survivors_abs, survivors_ratio = {}, {}

    for method in ("orb", "sift"):
        n = 1200 if method == "orb" else 600
        qa = detect_describe(pair.img_b, method, n)
        qb = detect_describe(pair.img_a, method, n)
        norm = cv2.NORM_HAMMING if method == "orb" else cv2.NORM_L2
        knn = cv2.BFMatcher(norm).knnMatch(qa.descriptors, qb.descriptors, k=2)
        d1 = np.array([p[0].distance for p in knn if len(p) == 2])

        out[f"{method}: descriptor dtype"] = str(qa.descriptors.dtype)
        out[f"{method}: distance is"] = "hamming bit count 0..256" if method == "orb" else "L2 over ~512-norm vectors"
        out[f"{method}: d1 min / median / max"] = (
            f"{d1.min():.1f} / {np.median(d1):.1f} / {d1.max():.1f}"
        )
        survivors_abs[method] = float((d1 < 50).mean())
        survivors_ratio[method] = len(matching.ratio_test(knn, 0.75)) / max(len(knn), 1)

        angles = np.array([kp.angle for kp in qa.keypoints])
        sizes = np.array([kp.size for kp in qa.keypoints])
        out[f"{method}: kp.size (a DIAMETER)"] = f"{sizes.min():.2f} .. {sizes.max():.2f} px"
        out[f"{method}: kp.angle (DEGREES)"] = f"{angles.min():.1f} .. {angles.max():.1f}"

    out["fixed cut d1 < 50 keeps (orb)"] = f"{100 * survivors_abs['orb']:.1f}%"
    out["fixed cut d1 < 50 keeps (sift)"] = f"{100 * survivors_abs['sift']:.1f}%"
    out["ratio test 0.75 keeps (orb)"] = f"{100 * survivors_ratio['orb']:.1f}%"
    out["ratio test 0.75 keeps (sift)"] = f"{100 * survivors_ratio['sift']:.1f}%"

    def spread(d: dict) -> float:
        lo, hi = min(d.values()), max(d.values())
        return hi / lo if lo > 0 else float("inf")

    # The one number the whole shape reduces to: how badly each threshold's
    # meaning changes when you carry it from one descriptor to the other.
    out["spread, fixed cut (x)"] = spread(survivors_abs)
    out["spread, ratio test (x)"] = spread(survivors_ratio)

    return BugReport(
        number=3,
        shape="ambiguous unit",
        instance="DMatch.distance -- a 0..256 bit count for ORB, an unbounded L2 float for SIFT",
        detector="read the constructor and the dtype, not the attribute name; a dimensionless "
                 "threshold (the ratio test) is the only kind that travels between the two",
        symptom="a threshold copied from a working pipeline keeps everything or nothing in the new "
                "one, with no error -- and the two match counts differ by more than an order of magnitude",
        numbers=out,
    )


# --------------------------------------------------------------------------
# Shape 4 -- the demo that cannot fail
# --------------------------------------------------------------------------
def shape4_demo_without_control() -> BugReport:
    """"The ratio test improves matching", demonstrated only where it can win.

    Run the ratio test on a scene of non-repeating clutter and precision jumps.
    Every tutorial shows that, and it is true. It is also unfalsifiable as
    presented: the demo has no input that could make it print the opposite, so
    it is evidence for nothing.

    Two controls are added here, and each one flips a different claim.

    **Control A -- a repeated-texture scene.** On a brick wall the ratio test
    still raises precision, and its *recall* collapses: it is refusing to guess
    between forty identical bricks, which is exactly what it was designed to do,
    and the number of correct matches that survive falls with it. If the next
    stage is RANSAC, which needs quantity, "higher precision" can leave you with
    a cleaner set that is too small to fit anything.

    **Control B -- the scoring metric itself.** Take the metric bug from shape 2
    and score it by precision alone, which is what most matching demos report:
    the wrong metric looks nearly fine. Score the same run by *recall*, or
    simply by the number of correct matches, and it has lost half of them. A
    demo is only as honest as the number it chooses to report, and precision
    alone can always be bought by keeping less.
    """
    rows = []
    for scene_name, scene in (
        ("distinctive clutter", scenes.textured_wall()),
        ("repeated brick wall", scenes.brick_wall()),
    ):
        pair = scenes.two_views(scene)
        qa = detect_describe(pair.img_b, "sift", 800)
        qb = detect_describe(pair.img_a, "sift", 800)
        matchable = matching.count_matchable(qa.points, qb.points, pair.H_true)
        knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(qa.descriptors, qb.descriptors, k=2)

        def grade(label, ms):
            return matching.score_matches(
                label, ms, qa.points, qb.points, pair.H_true, matchable=matchable
            )

        raw = grade("raw nearest neighbour", [p[0] for p in knn if len(p) == 2])
        ratio = grade("ratio test 0.75", matching.ratio_test(knn, 0.75))
        rows.append((scene_name, raw, ratio))

    numbers: dict = {}
    for scene_name, raw, ratio in rows:
        numbers[f"[{scene_name}] {raw.label}"] = (
            f"kept {raw.kept}  correct {raw.correct}  P {100*raw.precision:.1f}%  R {100*raw.recall:.1f}%"
        )
        numbers[f"[{scene_name}] {ratio.label}"] = (
            f"kept {ratio.kept}  correct {ratio.correct}  P {100*ratio.precision:.1f}%  R {100*ratio.recall:.1f}%"
        )
    numbers["correct matches lost, clutter"] = rows[0][1].correct - rows[0][2].correct
    numbers["correct matches lost, bricks"] = rows[1][1].correct - rows[1][2].correct

    # Control B: shape 2's metric bug, scored two ways. Precision hides it.
    metric_bug = shape2_hidden_operator().numbers
    numbers["metric bug, right metric"] = metric_bug["norm, quiet: right + ratio"]
    numbers["metric bug, wrong metric"] = metric_bug["norm, quiet: wrong + ratio"]

    return BugReport(
        number=4,
        shape="demo with no control",
        instance="'the ratio test improves matching', run only on a scene where it cannot lose",
        detector="before running a demo, name the input that would make it print the opposite -- "
                 "here, a repeated-texture scene, and a score of recall rather than precision",
        symptom="every run of the demo agrees with the claim, and the claim still fails in "
                "production -- because the demo never contained a case that could disagree",
        numbers=numbers,
    )


# --------------------------------------------------------------------------
# Shape 5 -- the fact with an expiry date
# --------------------------------------------------------------------------
def shape5_expired_fact() -> BugReport:
    """"SIFT lives in ``cv2.xfeatures2d``, so you need contrib" -- true until 2020.

    SIFT's patent expired in **March 2020** and SIFT moved out of contrib into
    main OpenCV at release **4.4.0** in July 2020. Every tutorial written before
    then is still online, still confident, still ranking well. It is a phone
    number on a business card from 2015: never a lie, just aged.

    "X lives in package Y" is a claim about a *version*, and you have a version
    in front of you. Every line below is a question put to this interpreter
    rather than to the internet, and the answers are stamped with
    ``cv2.__version__`` so they cannot be quoted out of their context later.

    The reading of the output, once you have it:

    * ``SIFT_create`` at the ``cv2`` top level means it came from main OpenCV.
    * ``SURF`` is *bound* in contrib but does not *construct* -- the binding
      ships, the implementation is compiled out behind ``OPENCV_ENABLE_NONFREE``.
      SURF is the one people confuse with SIFT, and it really is still
      restricted.
    * Both OpenCV wheels can be installed at once, at different major versions,
      and ``import cv2`` silently resolves to one of them with no warning.
    * ``cv2.LightGlueMatcher`` exists in OpenCV 5's base wheel and not in 4.x.
      Whichever answer this prints, it has a version attached to it -- which is
      the entire habit this shape is teaching.
    """
    import importlib.metadata as md

    installed = {}
    for pkg in ("opencv-python", "opencv-contrib-python"):
        try:
            installed[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            installed[pkg] = "not installed"

    try:
        cv2.xfeatures2d.SURF_create()
        surf = "constructs"
    except AttributeError:
        surf = "cv2.xfeatures2d not present in this build"
    except cv2.error as exc:
        surf = "NO -- " + str(exc).split(") ")[-1].split(" in function")[0]

    return BugReport(
        number=5,
        shape="fact with an expiry date",
        instance="'SIFT needs opencv-contrib' -- correct until July 2020, repeated ever since",
        detector="version-stamp the claim, then ask the interpreter: hasattr(cv2, 'SIFT_create') "
                 "settles it in one line",
        symptom="you go hunting for a package you do not need, or conclude your install is broken "
                "because cv2.xfeatures2d is missing -- and the advice you followed has no date on it",
        numbers={
            "cv2.__version__": cv2.__version__,
            "opencv-python installed": installed["opencv-python"],
            "opencv-contrib-python installed": installed["opencv-contrib-python"],
            "hasattr(cv2, 'SIFT_create')": hasattr(cv2, "SIFT_create"),
            "hasattr(cv2, 'xfeatures2d')": hasattr(cv2, "xfeatures2d"),
            "SURF binding present": hasattr(getattr(cv2, "xfeatures2d", object()), "SURF_create"),
            "SURF actually constructs": surf,
            "hasattr(cv2, 'LightGlueMatcher')": hasattr(cv2, "LightGlueMatcher"),
            "hasattr(cv2, 'ALIKED')": hasattr(cv2, "ALIKED"),
        },
    )


def all_shapes() -> list[BugReport]:
    """Reproduce all five, in order. Roughly ten seconds on this machine."""
    return [
        shape1_partial_implementation(),
        shape2_hidden_operator(),
        shape3_ambiguous_unit(),
        shape4_demo_without_control(),
        shape5_expired_fact(),
    ]
