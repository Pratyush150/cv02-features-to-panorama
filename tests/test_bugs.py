"""Each of the five shapes must still misbehave in the documented direction.

A demonstration of a bug rots faster than any other kind of code: OpenCV moves,
a default changes, and the example that used to show the failure quietly starts
showing nothing while the prose around it still claims otherwise. That is bug
shape 5 applied to this repository itself, and these tests are the guard.
"""

from __future__ import annotations

import cv2
import pytest

from feat import bugs, describe, scenes


@pytest.fixture(scope="module")
def reports():
    return {report.number: report for report in bugs.all_shapes()}


def test_every_shape_carries_a_detector_and_a_symptom(reports):
    """The transferable half of each report is the prose, so it must exist."""
    assert set(reports) == {1, 2, 3, 4, 5}
    for report in reports.values():
        assert report.shape and report.instance
        assert len(report.detector) > 30, "the detector is the transferable part"
        assert len(report.symptom) > 30, "the symptom is how you recognise it in a bug report"
        assert report.numbers, "every shape must carry measurements, not just prose"
        assert report.render()


def test_shape1_naive_suppression_really_over_counts(reports):
    numbers = reports[1].numbers
    truth = numbers["true interior corners"]
    assert truth == 49
    assert numbers["component-wise NMS, interior"] == truth
    assert numbers["naive R == dilate(R), interior"] > 10 * truth
    # ...and the bug vanishes on a blurred image, which is why it reproduces on
    # the synthetic test case and not on photographs.
    assert numbers["naive NMS on the BLURRED board"] < numbers["naive R == dilate(R), all"] / 3
    # The signature really has no threshold or suppression parameter in it.
    signature = numbers["signature (3 of 5 stages)"]
    for absent in ("threshold", "quality", "minDistance", "maxCorners"):
        assert absent not in signature
    complete = numbers["the complete pipeline"]
    assert "qualityLevel" in complete and "minDistance" in complete and "maxCorners" in complete


def test_shape2_the_wrong_metric_is_silent_and_the_right_one_is_loud(reports):
    numbers = reports[2].numbers
    # The loud direction: a binary metric on float descriptors throws.
    assert "Assertion failed" in numbers["norm, loud: Hamming on float32"]
    # crossCheck cannot be combined with knnMatch(k=2).
    assert "K == 1" in numbers["crossCheck, loud: with knnMatch"]
    # The quiet direction: crossCheck changes what match() returns, with no sign.
    assert numbers["crossCheck, quiet: match() checked"] < numbers["crossCheck, quiet: match() plain"]
    # The "BF" morpheme is demonstrated with a timing, and the timing is
    # REPORTED rather than asserted. An earlier version of this test asserted
    # that doubling the descriptor count more than doubles the time -- true in
    # expectation, and it failed on a busy machine at a ratio of 1.9, because a
    # wall-clock bound on shared hardware is a flaky test rather than a better
    # one. What is asserted is that the measurement happened at all; the
    # quadratic claim stands on the algorithm, and on the number example 06
    # prints, not on a threshold that can lose a coin flip in CI.
    assert numbers["BF: 300x300 descriptors (s)"] > 0
    assert numbers["BF: 600x600 descriptors (s)"] > 0
    assert numbers["BF: cost ratio for 2x the data"] > 1.0


def test_shape2_wrong_metric_loses_correct_matches_silently():
    """Asserted on live numbers rather than the report's strings.

    L2 on ORB's uint8 descriptors runs without complaint and finds fewer of the
    true correspondences. Raw nearest neighbour keeps one match per query under
    either metric, so the two are compared at an identical kept count and only
    the correct count can move.
    """
    from feat import matching

    pair = scenes.two_views(scenes.textured_wall())
    query = describe.detect_describe(pair.img_b, "orb", 1200)
    train = describe.detect_describe(pair.img_a, "orb", 1200)
    scores = {}
    for name, norm in (("hamming", cv2.NORM_HAMMING), ("l2", cv2.NORM_L2)):
        knn = cv2.BFMatcher(norm).knnMatch(query.descriptors, train.descriptors, k=2)
        scores[name] = matching.score_matches(
            name, [p[0] for p in knn if len(p) == 2], query.points, train.points, pair.H_true
        )
    assert scores["hamming"].kept == scores["l2"].kept, "raw NN keeps one per query either way"
    assert scores["l2"].correct < 0.9 * scores["hamming"].correct, (
        f"the wrong metric should lose correct matches: "
        f"{scores['hamming'].correct} -> {scores['l2'].correct}"
    )


def test_shape3_a_fixed_threshold_does_not_travel_and_a_ratio_does(reports):
    numbers = reports[3].numbers
    assert numbers["orb: descriptor dtype"] == "uint8"
    assert numbers["sift: descriptor dtype"] == "float32"
    assert numbers["orb: kp.size (a DIAMETER)"]
    # The whole shape, as one comparison: the absolute cut's meaning changes by
    # an order of magnitude between the two detectors and the ratio test's does
    # not.
    assert numbers["spread, fixed cut (x)"] > 10
    assert numbers["spread, ratio test (x)"] < 4
    assert numbers["spread, fixed cut (x)"] > 4 * numbers["spread, ratio test (x)"]


def test_shape3_metric_is_read_from_the_dtype():
    pair = scenes.two_views(scenes.textured_wall(200, 260))
    orb = describe.detect_describe(pair.img_a, "orb", 200)
    sift = describe.detect_describe(pair.img_a, "sift", 200)
    assert describe.metric_for(orb.descriptors) == cv2.NORM_HAMMING
    assert describe.metric_for(sift.descriptors) == cv2.NORM_L2
    # kp.size is a diameter, so the radius is half of it. Stated as an assertion
    # because using it as a radius doubles every patch you crop.
    facts = describe.descriptor_facts(sift)
    assert facts["kp_size_is"] == "diameter"
    assert facts["mean_l2_norm"] == pytest.approx(512, rel=0.02)


def test_shape4_the_control_scene_flips_the_result(reports):
    """The ratio test raises precision on both scenes and loses most of the
    correct matches on the repeated-texture one. A demo that only ran the first
    scene could never have printed that."""
    numbers = reports[4].numbers
    clutter_raw = numbers["[distinctive clutter] raw nearest neighbour"]
    clutter_ratio = numbers["[distinctive clutter] ratio test 0.75"]
    bricks_raw = numbers["[repeated brick wall] raw nearest neighbour"]
    bricks_ratio = numbers["[repeated brick wall] ratio test 0.75"]
    for line in (clutter_raw, clutter_ratio, bricks_raw, bricks_ratio):
        assert "kept" in line and "correct" in line and "P " in line and "R " in line

    lost_clutter = numbers["correct matches lost, clutter"]
    lost_bricks = numbers["correct matches lost, bricks"]
    # On the easy scene almost nothing correct is lost; on the control most of
    # it is. That contrast is the shape.
    assert lost_clutter >= 0
    assert lost_bricks > 0


def test_shape4_control_measured_live():
    from feat import matching

    results = {}
    for name, scene in (("clutter", scenes.textured_wall()), ("bricks", scenes.brick_wall())):
        pair = scenes.two_views(scene)
        query = describe.detect_describe(pair.img_b, "sift", 800)
        train = describe.detect_describe(pair.img_a, "sift", 800)
        matchable = matching.count_matchable(query.points, train.points, pair.H_true)
        knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(query.descriptors, train.descriptors, k=2)
        results[name] = (
            matching.score_matches("raw", [p[0] for p in knn if len(p) == 2], query.points,
                                   train.points, pair.H_true, matchable=matchable),
            matching.score_matches("ratio", matching.ratio_test(knn, 0.75), query.points,
                                   train.points, pair.H_true, matchable=matchable),
        )

    raw_c, ratio_c = results["clutter"]
    raw_b, ratio_b = results["bricks"]
    assert ratio_c.precision > raw_c.precision, "the advertised result"
    assert ratio_b.precision > raw_b.precision, "which also holds on the control scene"
    assert ratio_c.recall > 0.7, "on the easy scene the ratio test keeps most true matches"
    assert ratio_b.correct < raw_b.correct / 2, (
        f"on the control scene it discards most of them: "
        f"{raw_b.correct} correct -> {ratio_b.correct}"
    )


def test_shape5_asks_the_interpreter(reports):
    numbers = reports[5].numbers
    assert numbers["cv2.__version__"] == cv2.__version__
    # The claim under test: SIFT is at the cv2 top level, so it came from main
    # OpenCV rather than contrib. If this ever flips, the prose in the README is
    # out of date and this test is how we find out.
    assert numbers["hasattr(cv2, 'SIFT_create')"] is True
    assert isinstance(cv2.SIFT_create(), cv2.SIFT)
    # SURF is the one that is genuinely still restricted -- bound in contrib,
    # compiled out unless OPENCV_ENABLE_NONFREE was set at build time.
    surf = numbers["SURF actually constructs"]
    assert surf.startswith("NO") or surf == "constructs"
    if surf.startswith("NO"):
        assert "NONFREE" in surf or "patented" in surf
