"""Matching, and the three ways of not believing it.

Matching answers one question, once per descriptor: *for this descriptor in the
query image, which descriptor in the train image is the same physical point?*

Nearest-neighbour search always returns an answer. It never returns "I do not
know". On a scene with repeated texture -- a brick wall, a row of windows, a
keyboard -- that guarantee is the whole problem: forty near-identical patches
produce forty near-identical descriptors, one of them is nearest by sensor
noise alone, and the matcher reports it with total confidence.

Three filters, and they catch different diseases:

* **Lowe's ratio test** keeps a match only when the nearest neighbour is
  clearly nearer than the runner-up. It measures *separation*, not quality: a
  match with a tiny absolute distance is discarded if the runner-up is equally
  close. It kills **ambiguity**.
* **Cross-check** keeps a match only if both images pick each other. It kills
  **many-to-one** -- without it, six query keypoints can all claim the same
  train keypoint, and at most one of them can be right.
* **RANSAC** (in :mod:`feat.ransac`) kills matches that are unambiguous,
  mutual, and still geometrically impossible. The first two filters cannot see
  those at all.

The scoring helpers here exist so that every claim about those filters in this
repository is a measured precision and recall against a known homography,
rather than an adjective.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "MatchScore",
    "knn_l2_numpy",
    "knn_hamming_numpy",
    "ratio_test",
    "cross_check",
    "matched_points",
    "score_matches",
    "count_matchable",
]


@dataclass(frozen=True)
class MatchScore:
    """What a filter kept, and how much of it was right."""

    label: str
    kept: int
    correct: int
    matchable: int  # query keypoints that HAVE a true partner, so recall has a denominator

    @property
    def precision(self) -> float:
        """Of the matches we believed, what fraction were right."""
        return self.correct / self.kept if self.kept else 0.0

    @property
    def recall(self) -> float:
        """Of the matches that existed to be found, what fraction we found.

        This is the number a precision-only demo hides. A filter can reach 100%
        precision by keeping almost nothing, and RANSAC downstream needs
        quantity: four correspondences is the bare minimum for a homography and
        thirty is where it starts being stable. Precision without recall is a
        demo that cannot come out the other way.
        """
        return self.correct / self.matchable if self.matchable else 0.0

    def __str__(self) -> str:
        return (
            f"{self.label:<28} kept {self.kept:>4}  correct {self.correct:>4}  "
            f"precision {100 * self.precision:5.1f}%  recall {100 * self.recall:5.1f}%"
        )


def knn_l2_numpy(des_a: np.ndarray, des_b: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force L2 k-nearest-neighbours in NumPy. Returns ``(indices, distances)``.

    The whole matcher is one identity:

        ||a - b||**2  ==  a.a  -  2 a.b  +  b.b

    so every pairwise distance comes out of a single matrix multiply, and no
    ``N x M x 128`` intermediate array is ever built. Written the obvious way --
    ``((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)`` -- 500 x 500 SIFT
    descriptors would allocate 128 MB and 5000 x 5000 would allocate 12 GB.
    The identity is not a micro-optimisation; it is the difference between
    running and not running.

    float64 for the accumulation. In float32, ``a.a`` and ``2 a.b`` are both
    around 5e5 for OpenCV's 512-normalised SIFT descriptors and their difference
    is small, so catastrophic cancellation puts real error into exactly the
    small distances that matter most.
    """
    a = des_a.astype(np.float64)
    b = des_b.astype(np.float64)
    d2 = (a * a).sum(1)[:, None] - 2.0 * (a @ b.T) + (b * b).sum(1)[None, :]
    # Cancellation can leave a squared distance at -1e-9; sqrt of that is NaN,
    # and a NaN sorts unpredictably, so it must be clamped before the sqrt.
    np.maximum(d2, 0.0, out=d2)

    # argpartition, not argsort: we want the k smallest, not a full ordering of
    # all M candidates. O(M) against O(M log M), per query row.
    idx = np.argpartition(d2, kth=min(k, d2.shape[1] - 1), axis=1)[:, :k]
    dist = np.take_along_axis(d2, idx, axis=1) ** 0.5
    order = np.argsort(dist, axis=1)  # k is 2 in practice, so this is cheap
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(dist, order, axis=1)


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def knn_hamming_numpy(
    des_a: np.ndarray, des_b: np.ndarray, k: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """The same thing for binary descriptors: XOR, then count the set bits.

    A 256-bit lookup table turns "count the 1 bits in this byte" into an array
    index, which is what a CPU's ``POPCNT`` instruction does in hardware and
    what makes ORB matching so much cheaper than SIFT matching. The chunking is
    there because the intermediate XOR array is ``len(a) x len(b) x 32`` bytes:
    at 2000 x 2000 descriptors that is 128 MB in one allocation.
    """
    a = des_a.astype(np.uint8)
    b = des_b.astype(np.uint8)
    n = len(a)
    dist = np.empty((n, len(b)), np.int32)
    chunk = max(1, 4_000_000 // max(len(b) * a.shape[1], 1))
    for i in range(0, n, chunk):
        block = np.bitwise_xor(a[i:i + chunk, None, :], b[None, :, :])
        dist[i:i + chunk] = _POPCOUNT[block].sum(axis=2)

    idx = np.argpartition(dist, kth=min(k, dist.shape[1] - 1), axis=1)[:, :k]
    d = np.take_along_axis(dist, idx, axis=1)
    order = np.argsort(d, axis=1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(d, order, axis=1).astype(float)


def ratio_test(knn_pairs, ratio: float = 0.75) -> list:
    """Lowe's ratio test over the output of ``BFMatcher.knnMatch(k=2)``.

    Written as ``d1 < ratio * d2``, multiplied rather than divided, for two
    reasons that are both real bugs avoided. With ORB the distances are small
    integers and ``d2`` can legitimately be **0** when two descriptors are
    bit-identical, so ``d1 / d2 < ratio`` raises ``ZeroDivisionError`` on the
    exact input the test exists to reject. And the multiplied form does the
    right thing at ``d2 == 0``: it keeps nothing, which is correct, because a
    perfect tie is maximal ambiguity.

    ``pair[0]`` is the nearer neighbour -- ``knnMatch`` returns each pair sorted
    nearest-first. Get that backwards and the test becomes ``d2 < 0.75*d1``,
    which is essentially never true, and the symptom is zero matches on two
    images that obviously overlap.

    ``len(pair) < 2`` is guarded because ``knnMatch`` returns fewer than k
    neighbours when the *train* set holds fewer than k descriptors -- a
    near-blank second image, or a restrictive mask. It has nothing to do with
    where a keypoint sits in the query image.
    """
    return [pair[0] for pair in knn_pairs if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]


def cross_check(matches: list, des_query: np.ndarray, des_train: np.ndarray) -> list:
    """Keep only matches whose reverse nearest neighbour returns home.

    ``cv2.BFMatcher(..., crossCheck=True)`` does this too, but it cannot be
    combined with ``knnMatch(k=2)`` -- OpenCV asserts ``K == 1 && update == 0``
    -- so with the flag you get the ratio test or cross-check, never both. That
    is a limitation of the flag, not of the idea: matching train->query with
    k=1 and keeping the round trips takes four lines and composes with anything.

    The two filters are not "one is stronger". The ratio test allows many-to-one
    and cannot see it; cross-check forbids many-to-one by construction and
    cannot see ambiguity. Example 05 measures both, and both together.
    """
    if len(matches) == 0 or len(des_query) == 0 or len(des_train) == 0:
        return []
    back = cv2.BFMatcher(
        cv2.NORM_HAMMING if des_query.dtype == np.uint8 else cv2.NORM_L2
    ).match(des_train, des_query)
    home = {m.queryIdx: m.trainIdx for m in back}  # train index -> its best query index
    return [m for m in matches if home.get(m.trainIdx, -1) == m.queryIdx]


def matched_points(matches: list, pts_query: np.ndarray, pts_train: np.ndarray):
    """Pull the two ``(N, 1, 2)`` float32 point arrays a homography fit wants.

    ``queryIdx`` indexes the **first** argument that was handed to the matcher,
    ``trainIdx`` the **second**. Swapping them here silently estimates the
    inverse transform -- and RANSAC still reports a healthy inlier count,
    because the inverse of a good homography is also a good homography. There
    is no error message anywhere on this path, which is why the indexing lives
    in one function instead of being retyped at each call site.
    """
    src = pts_query[[m.queryIdx for m in matches]].reshape(-1, 1, 2).astype(np.float32)
    dst = pts_train[[m.trainIdx for m in matches]].reshape(-1, 1, 2).astype(np.float32)
    return src, dst


def count_matchable(
    pts_query: np.ndarray, pts_train: np.ndarray, h_query_to_train: np.ndarray, tol: float = 3.0
) -> int:
    """How many query keypoints actually have a partner to be found.

    Project every query point through the known homography and ask whether any
    train keypoint landed within ``tol`` pixels of it. Points that fall outside
    the overlap, or whose partner the detector simply did not fire on, are not
    findable by any matcher, and counting them in the denominator would make
    every recall number in this repo pessimistic by a constant factor.
    """
    if len(pts_query) == 0 or len(pts_train) == 0:
        return 0
    proj = cv2.perspectiveTransform(pts_query.reshape(-1, 1, 2), h_query_to_train).reshape(-1, 2)
    d = np.linalg.norm(proj[:, None, :] - pts_train[None, :, :], axis=2)
    return int((d.min(axis=1) < tol).sum())


def score_matches(
    label: str,
    matches: list,
    pts_query: np.ndarray,
    pts_train: np.ndarray,
    h_query_to_train: np.ndarray,
    tol: float = 3.0,
    matchable: int | None = None,
) -> MatchScore:
    """Grade a match list against the known homography.

    A match is correct when the query keypoint, pushed through the true
    homography, lands within ``tol`` pixels of the train keypoint it was paired
    with. ``tol = 3`` px is generous on purpose: the detector's own
    localisation noise is a fraction of a pixel, so 3 px separates "the same
    physical point" from "a different point entirely" without turning a
    correct-but-imprecise match into a scored error.
    """
    if matchable is None:
        matchable = count_matchable(pts_query, pts_train, h_query_to_train, tol)
    if not matches:
        return MatchScore(label=label, kept=0, correct=0, matchable=matchable)

    q = pts_query[[m.queryIdx for m in matches]].reshape(-1, 1, 2)
    t = pts_train[[m.trainIdx for m in matches]].reshape(-1, 2)
    proj = cv2.perspectiveTransform(q.astype(np.float32), h_query_to_train).reshape(-1, 2)
    correct = int((np.linalg.norm(proj - t, axis=1) < tol).sum())
    return MatchScore(label=label, kept=len(matches), correct=correct, matchable=matchable)
