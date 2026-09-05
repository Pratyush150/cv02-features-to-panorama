"""Two claims are asserted here, and the second is the one the README quotes.

1. The hand-rolled brute-force matcher returns what ``cv2.BFMatcher`` returns.
2. The ratio test measurably reduces false matches on a repeated-texture pair,
   and the real numbers are in the assertion messages.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from feat import describe, matching, scenes


@pytest.fixture(scope="module")
def sift_pair():
    pair = scenes.two_views(scenes.textured_wall())
    query = describe.detect_describe(pair.img_b, "sift", 400)  # query = B, H_true maps B -> A
    train = describe.detect_describe(pair.img_a, "sift", 400)
    return pair, query, train


@pytest.fixture(scope="module")
def brick_pair():
    pair = scenes.two_views(scenes.brick_wall())
    query = describe.detect_describe(pair.img_b, "sift", 800)
    train = describe.detect_describe(pair.img_a, "sift", 800)
    return pair, query, train


def test_numpy_l2_matcher_agrees_with_opencv(sift_pair):
    _, query, train = sift_pair
    a, b = query.descriptors[:250], train.descriptors[:250]
    idx, dist = matching.knn_l2_numpy(a, b, k=2)
    reference = cv2.BFMatcher(cv2.NORM_L2).knnMatch(a, b, k=2)

    ref_idx = np.array([[m.trainIdx for m in pair] for pair in reference])
    ref_dist = np.array([[m.distance for m in pair] for pair in reference])
    assert np.array_equal(idx[:, 0], ref_idx[:, 0]), "different nearest neighbour"
    # float32 accumulation inside OpenCV against float64 here, so the distances
    # agree to single-precision and not further.
    assert np.abs(dist - ref_dist).max() < 1e-3


def test_numpy_hamming_matcher_agrees_with_opencv():
    pair = scenes.two_views(scenes.textured_wall())
    a = describe.detect_describe(pair.img_b, "orb", 400).descriptors[:200]
    b = describe.detect_describe(pair.img_a, "orb", 400).descriptors[:200]
    idx, dist = matching.knn_hamming_numpy(a, b, k=2)
    reference = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(a, b, k=2)
    ref_dist = np.array([[m.distance for m in p] for p in reference])
    ref_idx = np.array([p[0].trainIdx for p in reference])

    # The distances must agree exactly. They are integer bit counts, so there is
    # no floating-point excuse available and any difference is a real bug.
    assert np.array_equal(dist, ref_dist), "XOR + popcount must match cv2 exactly (integers)"

    # The *indices* may differ, and only where two train descriptors are tied at
    # the same Hamming distance. Integer distances make exact ties common, and
    # argpartition breaks them by memory order while OpenCV breaks them by scan
    # order -- neither is wrong, and asserting index equality here would be
    # asserting an implementation detail of a tie-break.
    disagree = idx[:, 0] != ref_idx
    if disagree.any():
        rows = np.nonzero(disagree)[0]
        tied = [np.count_nonzero(
            matching.knn_hamming_numpy(a[r:r + 1], b, k=len(b))[1][0] == dist[r, 0]
        ) > 1 for r in rows]
        assert all(tied), f"{np.count_nonzero(~np.array(tied))} disagreements were not ties"
    assert disagree.mean() < 0.05, f"{100 * disagree.mean():.1f}% of queries disagreed"


def test_ratio_test_reduces_false_matches_on_repeated_texture(brick_pair):
    """The headline claim, with the real numbers reported in the message.

    Measured on the brick wall at the time of writing:
      raw nearest neighbour  650 kept, 19 correct, precision  2.9%
      ratio test @ 0.75       20 kept,  5 correct, precision 25.0%

    Precision has to improve by a wide margin -- that is the claim. Recall has
    to fall, which is the other half of the truth and the reason example 06
    calls the single-scene version of this demo a bug.
    """
    pair, query, train = brick_pair
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(query.descriptors, train.descriptors, k=2)
    matchable = matching.count_matchable(query.points, train.points, pair.H_true)

    raw = matching.score_matches("raw", [p[0] for p in knn if len(p) == 2],
                                 query.points, train.points, pair.H_true, matchable=matchable)
    filtered = matching.score_matches("ratio", matching.ratio_test(knn, 0.75),
                                      query.points, train.points, pair.H_true, matchable=matchable)

    assert raw.kept > 300, f"expected a large raw match set, got {raw.kept}"
    assert raw.precision < 0.10, (
        f"raw nearest neighbour should be almost all wrong on this scene: "
        f"{raw.correct}/{raw.kept} = {100 * raw.precision:.1f}%"
    )
    assert filtered.precision > 4 * raw.precision, (
        f"ratio test should multiply precision: raw {100 * raw.precision:.1f}% -> "
        f"filtered {100 * filtered.precision:.1f}%"
    )
    assert filtered.kept < raw.kept / 10, "and it should throw most of the matches away"


def test_ratio_test_also_helps_on_an_easy_scene(sift_pair):
    """The control in the other direction: it is not only a repeated-texture trick."""
    pair, query, train = sift_pair
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(query.descriptors, train.descriptors, k=2)
    raw = matching.score_matches("raw", [p[0] for p in knn if len(p) == 2],
                                 query.points, train.points, pair.H_true)
    filtered = matching.score_matches("ratio", matching.ratio_test(knn, 0.75),
                                      query.points, train.points, pair.H_true)
    assert filtered.precision > 0.85
    assert filtered.precision > raw.precision + 0.3
    assert filtered.recall > 0.7, "and on an easy scene it must keep most of the true matches"


def test_the_ratio_test_is_dimensionless(sift_pair):
    """Scale every descriptor by 7 and the ratio test keeps exactly the same matches.

    This is the whole reason the ratio test transfers between ORB and SIFT while
    an absolute distance threshold does not, so it is worth an assertion rather
    than a sentence.
    """
    _, query, train = sift_pair
    a, b = query.descriptors[:300], train.descriptors[:300]
    before = matching.ratio_test(cv2.BFMatcher(cv2.NORM_L2).knnMatch(a, b, k=2), 0.75)
    after = matching.ratio_test(
        cv2.BFMatcher(cv2.NORM_L2).knnMatch(a * 7.0, b * 7.0, k=2), 0.75
    )
    assert [(m.queryIdx, m.trainIdx) for m in before] == [(m.queryIdx, m.trainIdx) for m in after]

    # ...and an absolute cut does not survive the same rescaling at all.
    cut_before = sum(1 for p in cv2.BFMatcher(cv2.NORM_L2).knnMatch(a, b, k=2) if p[0].distance < 200)
    cut_after = sum(1 for p in cv2.BFMatcher(cv2.NORM_L2).knnMatch(a * 7.0, b * 7.0, k=2)
                    if p[0].distance < 200)
    assert cut_after < cut_before / 5


def test_cross_check_makes_matching_one_to_one(sift_pair):
    """Cross-check forbids many-to-one by construction; the ratio test cannot see it."""
    _, query, train = sift_pair
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(query.descriptors, train.descriptors, k=2)
    ratio_only = matching.ratio_test(knn, 0.75)
    mutual = matching.cross_check(ratio_only, query.descriptors, train.descriptors)

    train_indices = [m.trainIdx for m in mutual]
    assert len(train_indices) == len(set(train_indices)), "cross-check output must be one-to-one"
    assert len(mutual) <= len(ratio_only)


def test_ratio_test_survives_zero_distance():
    """Bit-identical descriptors give d2 == 0, and the multiplied form must not divide.

    The divided form `d1 / d2 < r` raises ZeroDivisionError on exactly the input
    the test exists to reject -- a perfect tie, which is maximal ambiguity.
    """
    class FakeMatch:
        def __init__(self, distance):
            self.distance = distance
            self.queryIdx = self.trainIdx = 0

    assert matching.ratio_test([[FakeMatch(0.0), FakeMatch(0.0)]], 0.75) == []
    assert len(matching.ratio_test([[FakeMatch(1.0), FakeMatch(100.0)]], 0.75)) == 1


def test_short_knn_pairs_are_skipped_not_unpacked():
    """knnMatch returns fewer than k neighbours when the TRAIN set is tiny."""
    a = np.random.default_rng(0).random((10, 8), dtype=np.float32)
    b = a[:1]  # one train descriptor, so every query gets exactly one neighbour
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(a, b, k=2)
    assert all(len(p) == 1 for p in pairs)
    assert matching.ratio_test(pairs, 0.75) == []  # must not raise


def test_scoring_helpers_handle_empty_input(sift_pair):
    _, query, train = sift_pair
    pair, _, _ = sift_pair
    score = matching.score_matches("nothing", [], query.points, train.points, pair.H_true)
    assert score.kept == 0 and score.precision == 0.0 and score.recall == 0.0
    assert matching.count_matchable(np.zeros((0, 2), np.float32), train.points, pair.H_true) == 0
