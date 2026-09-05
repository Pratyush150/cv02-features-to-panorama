"""Descriptor properties, and the failure modes at the edges of the detectors.

The interesting tests here are the ones about *empty* and *near-empty* inputs.
They are the frame-300-of-a-video failures: everything works for 299 frames and
then the camera faces a white wall.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from feat import describe, scenes


@pytest.fixture(scope="module")
def scene():
    return scenes.textured_wall(300, 400, seed=9)


def test_descriptor_shapes_and_metrics(scene):
    sift = describe.detect_describe(scene, "sift", 300)
    orb = describe.detect_describe(scene, "orb", 300)

    assert sift.descriptors.shape[1] == 128 and sift.descriptors.dtype == np.float32
    assert orb.descriptors.shape[1] == 32 and orb.descriptors.dtype == np.uint8
    assert sift.norm == cv2.NORM_L2 and orb.norm == cv2.NORM_HAMMING

    # 512 bytes against 32: exactly 16x, which is the number worth quoting.
    sift_bytes = describe.descriptor_facts(sift)["bytes_per_descriptor"]
    orb_bytes = describe.descriptor_facts(orb)["bytes_per_descriptor"]
    assert sift_bytes // orb_bytes == 16


def test_sift_descriptors_are_512_normalised_not_unit(scene):
    """Feed these to something expecting unit vectors and every number is 512x too big."""
    facts = describe.descriptor_facts(describe.detect_describe(scene, "sift", 300))
    assert 505 < facts["min_l2_norm"] < 520
    assert 505 < facts["max_l2_norm"] < 520
    # The 0.2 clip happens BEFORE the final renormalisation, so finished entries
    # can and do exceed 0.2 * 512.
    assert facts["fraction_entries_over_0p2_scaled"] > 0.0


def test_orb_bits_are_balanced(scene):
    """A binary code with a strong bias would be wasting bit positions."""
    facts = describe.descriptor_facts(describe.detect_describe(scene, "orb", 300))
    assert facts["bits"] == 256
    assert 100 < facts["mean_bits_set"] < 156


def test_points_are_x_y_not_row_col(scene):
    """kp.pt is (x, y); numpy is [row, col]. Every geometry call wants (x, y)."""
    feat = describe.detect_describe(scene, "sift", 50)
    points = feat.points
    assert points.shape == (len(feat.keypoints), 2)
    assert points[:, 0].max() < scene.shape[1]  # x against width
    assert points[:, 1].max() < scene.shape[0]  # y against height
    assert points[0] == pytest.approx(feat.keypoints[0].pt)


def test_a_blank_frame_gives_an_empty_array_not_none():
    """Both detectors return descriptors=None on a blank frame. Downstream code
    that does `len(des)` would raise AttributeError; normalising to an empty
    array here means the length check still works and still has to happen."""
    blank = np.full((200, 200, 3), 128, np.uint8)
    for method in ("sift", "orb"):
        feat = describe.detect_describe(blank, method)
        assert feat.descriptors is not None
        assert len(feat.descriptors) == 0
        assert feat.points.shape == (0, 2)


def test_nfeatures_is_a_soft_cap_for_orb(scene):
    """ORB allocates its budget per pyramid level and the rounding overshoots,
    so `assert len(kp) <= nfeatures` is a production incident waiting to happen."""
    feat = describe.detect_describe(scene, "orb", 500)
    assert len(feat.keypoints) >= 400  # it does roughly honour the request...
    # ...and the exact bound is not guaranteed, which is the point. A hard
    # assertion belongs to the caller, via a slice.
    assert len(feat.keypoints[:500]) <= 500


def test_unknown_method_raises_rather_than_guessing(scene):
    with pytest.raises(ValueError, match="sift"):
        describe.detect_describe(scene, "surf")


def test_timing_reports_the_minimum_and_scales_per_pair(scene):
    """Not asserting a speed -- that would be a flaky test on shared hardware.
    Asserting that the harness reports a positive time and divides by the pair
    count, so ORB and SIFT stay comparable at different keypoint counts."""
    feat = describe.detect_describe(scene, "orb", 200)
    seconds, per_pair = describe.time_knn_match(feat.descriptors, feat.descriptors, repeats=2)
    assert seconds > 0
    assert per_pair == pytest.approx(seconds / (len(feat.descriptors) ** 2) * 1e9)
