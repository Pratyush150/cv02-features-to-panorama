"""Descriptors: what they must be invariant to, and what each one costs.

A detector says *where*. A descriptor says *what the patch looks like*, as a
fixed-length vector you can compare with arithmetic. The same physical corner
photographed twice produces different pixels -- different scale, different
rotation, different light -- so the descriptor's whole design goal is to
produce (nearly) the same numbers anyway.

Two are used here and they sit at opposite ends of one trade:

* **SIFT** -- 128 float32 values, a 4x4 grid of cells each holding an 8-bin
  gradient-orientation histogram, unit-normalised (OpenCV then scales to a norm
  near 512), clipped at 0.2 and renormalised. 512 bytes. Compared with **L2**.
* **ORB** -- 256 bits: 256 "is pixel P brighter than pixel Q?" tests on a fixed
  pattern, rotated to the keypoint's intensity-centroid angle. 32 bytes, so
  exactly 16x smaller. Compared with **Hamming** distance, which is four XORs
  and four popcounts on 64-bit words.

The metric is not a preference, it is a property of the descriptor, and the two
ways of getting it wrong fail differently -- see :func:`metric_for` and example
06. This module is deliberately thin: OpenCV's SIFT and ORB are the production
tools and there is no teaching value in a slower reimplementation of them. What
is worth writing down is which properties they have, and measuring the cost
difference on this machine rather than repeating "ORB is faster".
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "Features",
    "detect_describe",
    "metric_for",
    "descriptor_facts",
    "time_knn_match",
]


@dataclass(frozen=True)
class Features:
    """Keypoints and descriptors from one image, with the metric they need."""

    keypoints: tuple
    descriptors: np.ndarray
    norm: int  # cv2.NORM_L2 or cv2.NORM_HAMMING -- carried with the data, not remembered
    name: str

    @property
    def points(self) -> np.ndarray:
        """Keypoint locations as an ``(N, 2)`` float32 array of ``(x, y)``.

        ``kp.pt`` is ``(x, y)``; NumPy indexing is ``[row, col] == [y, x]``.
        Every geometry function in OpenCV wants ``(x, y)``, so the conversion
        is done once, here, and never improvised at a call site. Swapping them
        produces a homography that transposed the world, with no error.
        """
        return np.float32([kp.pt for kp in self.keypoints]).reshape(-1, 2)


def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def detect_describe(img: np.ndarray, method: str = "sift", n_features: int = 0) -> Features:
    """Run SIFT or ORB and return keypoints, descriptors and the right metric.

    ``n_features`` is a *soft* cap, not a hard one. ORB allocates its budget per
    pyramid level and the per-level rounding overshoots, so asking for 1000 can
    return 1002. Any ``assert len(kp) <= n_features`` you write will eventually
    fire in production; slice if you need a hard bound.

    A blank or near-blank frame makes both detectors return zero keypoints and
    ``descriptors = None`` -- not an empty array. That is the shape of the
    classic "worked for 299 frames, crashed on frame 300 where the camera faced
    a white wall" bug. It is normalised to an empty array here so that callers
    can check ``len()`` instead of checking for ``None``, but the length check
    still has to happen: ``knnMatch`` needs at least two train descriptors.
    """
    gray = _to_gray(img)
    if method == "sift":
        det = cv2.SIFT_create(nfeatures=n_features)
        norm = cv2.NORM_L2
        empty = np.zeros((0, 128), np.float32)
    elif method == "orb":
        # ORB's default nfeatures is 500, which is low for a large image and is
        # the usual reason a first ORB run finds "suspiciously few" features.
        det = cv2.ORB_create(nfeatures=n_features or 500)
        norm = cv2.NORM_HAMMING
        empty = np.zeros((0, 32), np.uint8)
    else:
        raise ValueError(f"method must be 'sift' or 'orb', got {method!r}")

    kps, des = det.detectAndCompute(gray, None)
    return Features(
        keypoints=tuple(kps),
        descriptors=empty if des is None else des,
        norm=norm,
        name=method,
    )


def metric_for(descriptors: np.ndarray) -> int:
    """The distance metric a descriptor array requires, read off its dtype.

    ``uint8`` means a packed bit string, so Hamming. ``float32`` means a real
    vector, so L2. This is a one-line function because the point is that the
    answer is one line: the dtype is the tell, and nobody needs to remember
    which detector produced the array.

    The two mistakes are not symmetric, which is what makes one of them
    dangerous. ``NORM_HAMMING`` on float32 **throws** immediately
    (``batchDistance`` asserts on the type). ``NORM_L2`` on uint8 **runs**,
    silently, treating each byte as a coordinate -- and it keeps *more* matches
    than the correct metric while being far less precise, so a pipeline that
    watches its match count as a health signal reads the failure as an
    improvement. Example 06 measures both.
    """
    if descriptors.dtype == np.uint8:
        return cv2.NORM_HAMMING
    return cv2.NORM_L2


def descriptor_facts(feat: Features) -> dict:
    """Measured properties of one descriptor set: size, dtype, norms, spread.

    Everything here is measured rather than quoted, including the one number
    people are most often surprised by: OpenCV's SIFT descriptors are unit
    vectors *scaled by 512 and rounded*, so their L2 norm is around 512 and
    their dtype is float32 holding whole numbers. Feed them to something that
    expects unit vectors -- a cosine layer, a distance threshold from a paper --
    and every number is 512 times too big.
    """
    des = feat.descriptors
    facts = {
        "name": feat.name,
        "count": int(des.shape[0]),
        "dims": int(des.shape[1]) if des.size else 0,
        "dtype": str(des.dtype),
        "bytes_per_descriptor": int(des.itemsize * des.shape[1]) if des.size else 0,
        "metric": "hamming" if metric_for(des) == cv2.NORM_HAMMING else "L2",
    }
    if des.size == 0:
        return facts
    if des.dtype == np.uint8:
        # For a binary descriptor the meaningful "magnitude" is how many of the
        # 256 bits are set. Near 128 is what a good binary code looks like: the
        # tests are balanced, so no bit position is wasted always answering the
        # same way.
        bits = np.unpackbits(des, axis=1)
        facts["mean_bits_set"] = float(bits.sum(axis=1).mean())
        facts["bits"] = int(des.shape[1] * 8)
    else:
        norms = np.linalg.norm(des, axis=1)
        facts["mean_l2_norm"] = float(norms.mean())
        facts["min_l2_norm"] = float(norms.min())
        facts["max_l2_norm"] = float(norms.max())
        facts["fraction_entries_over_0p2_scaled"] = float((des > 0.2 * 512).mean())
    sizes = np.array([kp.size for kp in feat.keypoints])
    facts["kp_size_min"] = float(sizes.min())
    facts["kp_size_max"] = float(sizes.max())
    # kp.size is a DIAMETER. Crop a patch using it as a radius and every patch
    # is twice the intended size, which quietly wrecks any descriptor you go on
    # to compute yourself. kp.angle is in degrees, and -1 when unassigned.
    facts["kp_size_is"] = "diameter"
    return facts


def time_knn_match(
    des_a: np.ndarray, des_b: np.ndarray, repeats: int = 5
) -> tuple[float, float]:
    """Time ``knnMatch(k=2)`` on this CPU. Returns ``(best_seconds, per_pair_ns)``.

    The **minimum** over ``repeats`` runs, not the mean. A timing sample is a
    true cost plus interference from whatever else the machine is doing, and
    interference is strictly positive -- it can only make a run slower. The
    minimum is therefore the closest thing to the underlying cost; a mean
    reports the machine's mood as well.

    ``per_pair_ns`` divides by ``len(a) * len(b)`` so that ORB and SIFT can be
    compared even when they detect different numbers of keypoints. Comparing
    raw wall-clock times on different descriptor counts measures the detectors,
    not the metrics.
    """
    matcher = cv2.BFMatcher(metric_for(des_a))
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        matcher.knnMatch(des_a, des_b, k=2)
        best = min(best, time.perf_counter() - t0)
    pairs = max(len(des_a) * len(des_b), 1)
    return best, best / pairs * 1e9
