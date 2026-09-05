"""The panorama's two halves, tested separately, because they fail separately.

Estimating the homography and painting the pixels are different steps. The
overlap agreement checks the first one; the seam profile checks the second. A
single "stitch quality" number could not tell you which of the two to fix.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from feat import describe, matching, panorama, ransac, scenes

# The stated thresholds. Both are quoted in the README, so they are named here.
MIN_OVERLAP_NCC = 0.99      # how well the two warped images must agree geometrically
MIN_SEAM_IMPROVEMENT = 3.0  # how much smaller feathering must make the seam step


@pytest.fixture(scope="module")
def stitched():
    """One honest run of the whole pipeline, shared by the tests below."""
    pair = scenes.two_views(scenes.textured_wall())
    query = describe.detect_describe(pair.img_b, "sift")
    train = describe.detect_describe(pair.img_a, "sift")
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(query.descriptors, train.descriptors, k=2)
    good = matching.ratio_test(knn, 0.75)
    src, dst = matching.matched_points(good, query.points, train.points)
    result = ransac.ransac_homography(
        src.reshape(-1, 2), dst.reshape(-1, 2),
        ransac.inlier_threshold(1.0, dof=2, confidence=0.95), seed=0,
    )
    return pair, result, good


def test_the_recovered_homography_is_sub_pixel(stitched):
    pair, result, good = stitched
    corners = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])
    err = float(np.abs(ransac.transfer(result.H, corners)
                       - ransac.transfer(pair.H_true, corners)).max())
    assert err < 1.0, f"worst corner error {err:.3f} px against the known homography"
    assert result.inlier_ratio > 0.8, f"inlier ratio only {100 * result.inlier_ratio:.1f}%"
    assert len(good) > 100


def test_canvas_is_sized_from_the_corners_and_keeps_more_than_the_guess(stitched):
    """The guessed canvas (w1 + w2, max(h1, h2)) silently crops the tilted warp."""
    pair, result, _ = stitched
    base, other = pair.img_a, pair.img_b
    canvas = panorama.warp_onto_canvas(base, other, result.H)
    correct = int((panorama.paste(canvas).sum(axis=2) > 0).sum())

    guessed_size = (base.shape[1] + other.shape[1], max(base.shape[0], other.shape[0]))
    guessed = cv2.warpPerspective(other, result.H, guessed_size)
    guessed[0:base.shape[0], 0:base.shape[1]] = base
    naive = int((guessed.sum(axis=2) > 0).sum())

    assert correct > naive, f"corner-sized canvas kept {correct}, guessed kept {naive}"
    # And the warp really does stick out above row zero, which is what the
    # guessed canvas crops -- otherwise this test would pass for another reason.
    transform, _ = panorama.canvas_transform(base.shape, other.shape, result.H)
    assert transform[1, 2] > 0


def test_overlap_region_agrees_above_the_stated_similarity(stitched):
    """The claim in the README: NCC above 0.99 across the overlap.

    Measured at the time of writing: 0.9992 with our homography, against 0.9993
    with the known one -- our estimate is as good as the ground truth to four
    decimal places.
    """
    pair, result, _ = stitched
    ours = panorama.overlap_agreement(panorama.warp_onto_canvas(pair.img_a, pair.img_b, result.H))
    ideal = panorama.overlap_agreement(
        panorama.warp_onto_canvas(pair.img_a, pair.img_b, pair.H_true)
    )
    assert ours["overlap_px"] > 50_000, "the two views must actually overlap"
    assert ours["ncc"] > MIN_OVERLAP_NCC, f"NCC only {ours['ncc']:.4f}"
    assert ours["mean_abs_diff"] < 5.0
    assert ours["frac_disagree"] < 0.01
    assert abs(ours["ncc"] - ideal["ncc"]) < 0.005


def test_a_deliberately_wrong_homography_fails_the_same_check(stitched):
    """The control. If a broken H also scored above 0.99, the test above would
    be measuring nothing -- a demo with no control, which is bug shape 4."""
    pair, result, _ = stitched
    broken = result.H.copy()
    broken[0, 2] += 25.0  # a 25 px shift: small, and far outside the tolerance
    stats = panorama.overlap_agreement(panorama.warp_onto_canvas(pair.img_a, pair.img_b, broken))
    assert stats["ncc"] < MIN_OVERLAP_NCC, f"a 25 px error still scored {stats['ncc']:.4f}"


def test_feathering_reduces_the_seam_step(stitched):
    """With an exposure difference between the shots, as real auto-exposure gives."""
    pair, result, _ = stitched
    brighter = np.clip(pair.img_b.astype(np.float32) * 1.22 + 6, 0, 255).astype(np.uint8)
    canvas = panorama.warp_onto_canvas(pair.img_a, brighter, result.H)

    seam_x = int(pair.img_a.shape[1] + canvas.translation[0, 2]) - 1
    step_paste = float(np.abs(np.diff(panorama.seam_column_profile(panorama.paste(canvas), seam_x))).max())
    step_feather = float(np.abs(np.diff(panorama.seam_column_profile(panorama.feather(canvas), seam_x))).max())

    assert step_paste > 15, f"the exposure difference should make a visible step, got {step_paste:.1f}"
    assert step_paste / step_feather > MIN_SEAM_IMPROVEMENT, (
        f"paste step {step_paste:.2f} vs feathered {step_feather:.2f} -- "
        f"only {step_paste / step_feather:.1f}x better"
    )


def test_feathering_produces_no_nan_and_leaves_single_coverage_untouched(stitched):
    """Outside both masks the weights sum to zero, and 0/0 would survive the cast
    to uint8 as an arbitrary byte value rather than as an obvious error."""
    pair, result, _ = stitched
    canvas = panorama.warp_onto_canvas(pair.img_a, pair.img_b, result.H)
    blended = panorama.feather(canvas)
    assert blended.dtype == np.uint8
    assert np.isfinite(blended.astype(np.float64)).all()

    only_base = canvas.base_mask & ~canvas.other_mask
    assert only_base.sum() > 1000
    assert np.array_equal(blended[only_base], canvas.base[only_base])


def test_paste_lets_the_base_image_win(stitched):
    pair, result, _ = stitched
    canvas = panorama.warp_onto_canvas(pair.img_a, pair.img_b, result.H)
    pasted = panorama.paste(canvas)
    assert np.array_equal(pasted[canvas.base_mask], canvas.base[canvas.base_mask])


def test_masks_come_from_the_warp_not_from_thresholding_the_pixels():
    """A dark region of a real image contains legitimate zeros, so `pixel > 0`
    would punch holes in the mask exactly where the scene is dark."""
    dark = np.zeros((80, 100, 3), np.uint8)  # an entirely black image
    identity = np.eye(3)
    canvas = panorama.warp_onto_canvas(dark, dark, identity)
    assert canvas.base_mask.all(), "an all-black image still covers its own footprint"
    assert canvas.overlap.all()
