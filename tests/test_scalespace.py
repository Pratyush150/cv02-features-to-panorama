"""The pyramid's bookkeeping is where scale space usually goes wrong.

None of this is deep, and all of it is off-by-one arithmetic that produces a
detector which quietly searches two layers instead of three.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from feat import scalespace, scenes


@pytest.fixture(scope="module")
def scene() -> np.ndarray:
    return cv2.cvtColor(scenes.textured_wall(300, 400, seed=6), cv2.COLOR_BGR2GRAY)


def test_sigmas_are_geometric_and_cover_one_doubling(scene):
    """sigma_i = sigma0 * 2**(i/n), so the nth step is exactly a factor of two."""
    _, sigmas = scalespace.gaussian_octave(scene, n_intervals=3, sigma0=1.6)
    assert sigmas[0] == pytest.approx(1.6)
    assert sigmas[3] / sigmas[0] == pytest.approx(2.0)
    ratios = [sigmas[i + 1] / sigmas[i] for i in range(len(sigmas) - 1)]
    assert np.allclose(ratios, ratios[0])  # geometric, not arithmetic


def test_layer_bookkeeping(scene):
    """n_intervals + 3 images -> n_intervals + 2 DoG layers -> n_intervals searchable."""
    for n in (2, 3, 4):
        blurred, sigmas = scalespace.gaussian_octave(scene, n_intervals=n)
        dog, dog_sigmas = scalespace.dog_octave(blurred, sigmas)
        assert len(blurred) == n + 3
        assert len(dog) == len(blurred) - 1 == len(dog_sigmas)
        assert len(dog) - 2 == n, "searchable layers must equal the requested interval count"


def test_dog_really_is_a_difference_of_gaussians(scene):
    blurred, sigmas = scalespace.gaussian_octave(scene)
    dog, _ = scalespace.dog_octave(blurred, sigmas)
    assert np.allclose(dog[0], blurred[1] - blurred[0])


def test_pyramid_halves_and_stops_before_the_kernel_outgrows_the_image(scene):
    pyramid = scalespace.build_pyramid(scene, n_octaves=8)
    assert pyramid[0].downsample == 1
    for i in range(1, len(pyramid)):
        assert pyramid[i].downsample == 2 * pyramid[i - 1].downsample
        assert pyramid[i].blurred[0].shape[0] == pyramid[i - 1].blurred[0].shape[0] // 2
    assert min(pyramid[-1].blurred[0].shape[:2]) >= 32


def test_extrema_are_returned_in_full_image_coordinates(scene):
    """A keypoint from octave 2 must be plotted at 4x its own pixel coordinate."""
    pyramid = scalespace.build_pyramid(scene, n_octaves=3)
    for octave in pyramid:
        found = scalespace.dog_extrema(octave, contrast=8.0)
        if len(found) == 0:
            continue
        assert found[:, 0].max() < scene.shape[1] + octave.downsample
        assert found[:, 1].max() < scene.shape[0] + octave.downsample
        # The reported sigma is scaled by the same factor, or keypoint sizes
        # from different octaves are not comparable.
        assert found[:, 2].min() >= octave.dog_sigmas[1] * octave.downsample - 1e-6


def test_extrema_ignore_the_one_pixel_border(scene):
    """Border pixels' 26 neighbours include values cv2's padding invented."""
    pyramid = scalespace.build_pyramid(scene, n_octaves=1)
    found = scalespace.dog_extrema(pyramid[0], contrast=4.0)
    assert len(found) > 0
    assert found[:, 0].min() >= 1 and found[:, 1].min() >= 1


def test_a_synthetic_blob_is_found_at_its_own_scale():
    """The point of scale space, as one assertion.

    A disc of radius r is most distinctive at a blur comparable to its size, so
    the DoG extremum on it should be found at a sigma that grows with r. Two
    discs, one twice the radius, and the larger must win at the larger sigma.
    """
    sigmas = {}
    for radius in (6, 14):
        img = np.full((200, 200), 40, np.uint8)
        cv2.circle(img, (100, 100), radius, 220, -1)
        pyramid = scalespace.build_pyramid(img, n_octaves=3)
        best = None
        for octave in pyramid:
            found = scalespace.dog_extrema(octave, contrast=2.0)
            for x, y, sigma, value in found:
                if abs(x - 100) < 4 and abs(y - 100) < 4:
                    if best is None or abs(value) > best[1]:
                        best = (sigma, abs(value))
        assert best is not None, f"no extremum on the disc of radius {radius}"
        sigmas[radius] = best[0]
    assert sigmas[14] > sigmas[6], f"scale did not track size: {sigmas}"


def test_no_extrema_on_a_blank_image():
    """A flat frame must return an empty (0, 4) array, not raise and not return None."""
    pyramid = scalespace.build_pyramid(np.full((128, 128), 100, np.uint8), n_octaves=2)
    found = scalespace.dog_extrema(pyramid[0], contrast=4.0)
    assert found.shape == (0, 4)
