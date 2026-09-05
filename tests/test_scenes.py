"""The scene generators are the ground truth everything else is scored against.

If a scene is not deterministic, every other test in this suite is measuring a
different image each run and none of their numbers mean anything. That makes
these the first tests, not an afterthought.
"""

from __future__ import annotations

import cv2
import numpy as np

from feat import ransac, scenes


def test_generators_are_deterministic():
    """Two calls with the same seed produce byte-identical images."""
    for factory in (scenes.textured_wall, scenes.brick_wall, scenes.gradient_lit_page):
        first, second = factory(seed=3), factory(seed=3)
        assert np.array_equal(first, second), f"{factory.__name__} is not deterministic"


def test_generators_do_not_touch_the_global_rng():
    """Building a scene must not advance numpy's global random state.

    A generator that used np.random.seed or np.random.rand would make every
    downstream test's result depend on how many scenes ran before it -- the kind
    of coupling that produces a suite which passes alone and fails in CI.
    """
    np.random.seed(0)
    before = np.random.rand(4)
    np.random.seed(0)
    scenes.textured_wall(120, 160)
    scenes.brick_wall(120, 160)
    after = np.random.rand(4)
    assert np.array_equal(before, after)


def test_different_seeds_give_different_scenes():
    assert not np.array_equal(scenes.textured_wall(seed=1), scenes.textured_wall(seed=2))


def test_checkerboard_reports_its_own_answer():
    board = scenes.checkerboard(squares=8, square_px=40, margin=30)
    assert board.n_interior == 49  # (8 - 1) ** 2
    x0, y0, x1, y1 = board.interior
    assert 0 < x0 < x1 < board.image.shape[1]
    assert 0 < y0 < y1 < board.image.shape[0]
    assert board.image.dtype == np.uint8
    # blur=False must leave exact plateaus, which is what makes the naive
    # non-maximum-suppression bug reproducible in example 06.
    assert len(np.unique(board.image)) == 2
    assert len(np.unique(scenes.checkerboard(blur=True).image)) > 2


def test_brick_wall_is_visible_in_luminance_not_only_in_colour():
    """The wall has to contrast in GREY, because every detector here works on grey.

    A brick colour that looks right on screen can convert to almost exactly the
    mortar's grey level, leaving a wall that is invisible to SIFT. This test
    exists because that happened.
    """
    grey = cv2.cvtColor(scenes.brick_wall(), cv2.COLOR_BGR2GRAY)
    dark, bright = np.percentile(grey, [10, 90])
    assert bright - dark > 50, f"only {bright - dark:.0f} grey levels between brick and mortar"
    # And it must actually produce keypoints, or the "repeated texture" demos
    # are measuring an empty set.
    assert len(cv2.SIFT_create(800).detect(grey, None)) > 200


def test_two_views_homography_is_exact():
    """H_true must map img_b's pixel coordinates into img_a's, to float precision.

    Checked by composing the two known scene-to-sensor transforms independently
    of how ViewPair built them: a point on the wall, projected into both photos,
    must satisfy p_a = H_true * p_b.
    """
    scene = scenes.textured_wall(400, 500, seed=2)
    out = (320, 240)
    quad_a = np.float32([[40, 40], [340, 40], [340, 290], [40, 290]])
    quad_b = np.float32([[120, 30], [430, 60], [420, 300], [110, 270]])
    pair = scenes.two_views(scene, out_size=out, quad_a=quad_a, quad_b=quad_b)

    sensor = np.float32([[0, 0], [out[0], 0], [out[0], out[1]], [0, out[1]]])
    h_a = cv2.getPerspectiveTransform(quad_a, sensor)
    h_b = cv2.getPerspectiveTransform(quad_b, sensor)

    wall_points = np.array([[60.0, 60.0], [300.0, 80.0], [200.0, 260.0], [150.0, 150.0]])
    in_a = ransac.transfer(h_a, wall_points)
    in_b = ransac.transfer(h_b, wall_points)
    assert np.abs(ransac.transfer(pair.H_true, in_b) - in_a).max() < 1e-6
    assert abs(pair.H_true[2, 2] - 1.0) < 1e-12  # normalised, so two H can be compared


def test_two_views_needs_all_eight_degrees_of_freedom():
    """The default pair must have a non-zero bottom row.

    A pair related by a pure shift or scale would let an affine or similarity
    fit look correct, so any claim this repo makes about needing a homography
    would be untested. The perspective terms are small -- they are supposed to
    be -- but they must not be zero.
    """
    pair = scenes.two_views(scenes.textured_wall(300, 400))
    assert np.abs(pair.H_true[2, :2]).max() > 1e-6


def test_scale_pair_homography_matches_the_resize():
    full, small, h = scenes.scale_pair(cv2.cvtColor(scenes.textured_wall(300, 400), cv2.COLOR_BGR2GRAY), 0.5)
    assert small.shape[0] == full.shape[0] // 2
    # H maps small -> full, so the centre of the small image lands on the centre
    # of the full one.
    centre_small = np.array([[small.shape[1] / 2, small.shape[0] / 2]])
    mapped = ransac.transfer(h, centre_small)[0]
    assert np.allclose(mapped, [full.shape[1] / 2, full.shape[0] / 2], atol=1.0)


def test_paste_intruder_does_not_mutate_its_input():
    """It must return a copy. A generator that mutates the scene passed to it
    makes every later call return a different image, and the resulting figure
    changes depending on which examples ran first."""
    original = scenes.textured_wall(200, 200)
    snapshot = original.copy()
    scenes.paste_intruder(original, 20, 20, 40)
    assert np.array_equal(original, snapshot)
