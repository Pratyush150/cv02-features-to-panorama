"""``feat`` -- corners, descriptors, matching, RANSAC and a panorama, from scratch.

The package is layered so that each module depends only on the ones before it,
which is the same order the README and the walkthrough teach in:

``scenes``      synthetic images with known ground truth -- no photographs, no downloads
``harris``      the structure tensor, the corner response, and turning a map into a list
``scalespace``  a difference-of-Gaussian pyramid, and why a fixed window is not enough
``describe``    SIFT and ORB, what each is invariant to, and what each costs
``matching``    brute force, Lowe's ratio test, cross-check, and scoring against truth
``ransac``      the DLT homography and RANSAC around it
``panorama``    one canvas, two warps, and the difference between a seam and a blend
``bugs``        five shapes of bug, each reproduced on purpose
``figures``     matplotlib defaults for the figures in docs/figures

Nothing here touches the network, reads a file, or seeds the global NumPy
generator at import time.
"""

__version__ = "1.0.0"

__all__ = [
    "scenes",
    "harris",
    "scalespace",
    "describe",
    "matching",
    "ransac",
    "panorama",
    "bugs",
    "figures",
]
