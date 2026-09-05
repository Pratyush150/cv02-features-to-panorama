"""Warping two images onto one canvas, and the two ways of filling the overlap.

Estimating H and painting pixels are two different steps, and they fail
differently. RANSAC can hand you a homography that is correct to half a pixel
while the composite still looks wrong -- because nothing about fitting a model
decides what colour an overlap pixel should be. Keeping the two apart is the
whole structure of this module: :func:`warp_onto_canvas` does the geometry,
:func:`paste` and :func:`feather` do the pixels, and
:func:`overlap_agreement` measures the geometry *through* the pixels so the two
can be diagnosed separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "Canvas",
    "canvas_transform",
    "warp_onto_canvas",
    "paste",
    "feather",
    "overlap_agreement",
    "seam_column_profile",
]


@dataclass(frozen=True)
class Canvas:
    """Both images warped onto one shared frame, with their validity masks."""

    base: np.ndarray  # the base image, translated into canvas coordinates
    other: np.ndarray  # the second image, warped by H and translated
    base_mask: np.ndarray  # bool: where `base` actually has pixels
    other_mask: np.ndarray
    translation: np.ndarray  # the 3x3 T that shifted everything into view
    size: tuple[int, int]  # (width, height)

    @property
    def overlap(self) -> np.ndarray:
        return self.base_mask & self.other_mask


def canvas_transform(
    base_shape: tuple[int, int], other_shape: tuple[int, int], h: np.ndarray
) -> tuple[np.ndarray, tuple[int, int]]:
    """Size the canvas from the transformed corners, never by guessing.

    The tutorial default ``(w_base + w_other, max(h_base, h_other))`` happens to
    survive a left-to-right pivot and nothing else. Any warp that tilts pushes
    part of the second image *above* row zero, and those rows are silently
    cropped -- silently, and the result looks deliberate, because a panorama
    with a straight-cut corner reads as a crop rather than as a bug.

    The recipe, which is worth memorising: transform the four corners of the
    source, take the bounding box together with the base image's own corners,
    and prepend the translation that moves the minimum to (0, 0). Returns
    ``(T, (width, height))``.
    """
    hb, wb = base_shape[:2]
    ho, wo = other_shape[:2]
    corners_base = np.float32([[0, 0], [wb, 0], [wb, hb], [0, hb]]).reshape(-1, 1, 2)
    corners_other = np.float32([[0, 0], [wo, 0], [wo, ho], [0, ho]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners_other, h.astype(np.float64))

    allc = np.vstack([corners_base, warped]).reshape(-1, 2)
    x_min, y_min = np.floor(allc.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(allc.max(axis=0)).astype(int)
    t = np.array([[1.0, 0.0, -x_min], [0.0, 1.0, -y_min], [0.0, 0.0, 1.0]])
    return t, (int(x_max - x_min), int(y_max - y_min))


def warp_onto_canvas(base: np.ndarray, other: np.ndarray, h: np.ndarray) -> Canvas:
    """Put both images in one frame. ``h`` maps ``other``'s pixels into ``base``'s.

    Both images are warped, not just one: the base by the pure translation T
    and the other by ``T @ h``. Warping the base with the same interpolator as
    the other, rather than slicing it in with array indexing, keeps the two
    sides of the seam comparable -- if one side is resampled and the other is
    not, the overlap agreement measured below would include a resampling
    difference that has nothing to do with the homography.

    The masks come from warping a field of 255s, not from testing ``pixel > 0``
    on the result. A dark region of a real image contains legitimate zeros, and
    ``pixel > 0`` would punch holes in the mask exactly where the scene is dark.
    """
    t, size = canvas_transform(base.shape, other.shape, h)

    warped_base = cv2.warpPerspective(base, t, size, flags=cv2.INTER_LINEAR)
    warped_other = cv2.warpPerspective(other, t @ h.astype(np.float64), size, flags=cv2.INTER_LINEAR)

    ones_base = np.full(base.shape[:2], 255, np.uint8)
    ones_other = np.full(other.shape[:2], 255, np.uint8)
    mask_base = cv2.warpPerspective(ones_base, t, size, flags=cv2.INTER_NEAREST) > 127
    mask_other = cv2.warpPerspective(
        ones_other, t @ h.astype(np.float64), size, flags=cv2.INTER_NEAREST
    ) > 127

    return Canvas(
        base=warped_base,
        other=warped_other,
        base_mask=mask_base,
        other_mask=mask_other,
        translation=t,
        size=size,
    )


def paste(canvas: Canvas) -> np.ndarray:
    """Hard overwrite: wherever the base has pixels, the base wins.

    This is the two-line compositor every tutorial ships, and it is honest
    about what it does -- which is why it is kept rather than hidden. Its
    failure is a **step**: any exposure difference between the two images
    appears as a hard edge along the boundary of the base image, because one
    column comes entirely from one exposure and the next column entirely from
    the other. Example 08 measures the size of that step in grey levels.
    """
    out = canvas.other.copy()
    out[canvas.base_mask] = canvas.base[canvas.base_mask]
    return out


def feather(canvas: Canvas) -> np.ndarray:
    """Blend the overlap with weights that fall to zero at each image's edge.

    The weight for each image is its distance to its own nearest invalid pixel,
    from ``cv2.distanceTransform``. That makes each image's contribution 1 deep
    inside it and 0 at its border, so the two weights cross over smoothly
    somewhere in the overlap and neither image's edge is ever visible as a step.

    Two details that are the difference between working and not:

    * The weights are normalised by their **sum**, not clamped independently.
      Independent weights do not add to 1 in the overlap, so the blend darkens
      or brightens there -- a soft band instead of a hard step, which is harder
      to notice and just as wrong.
    * The sum is floored before the division. Outside both masks it is zero, and
      0/0 is a NaN that survives the cast to uint8 as an arbitrary byte value.

    What feathering does *not* fix: if the two images genuinely disagree -- a
    person who moved between exposures, or parallax from a camera that
    translated -- averaging paints half of each, giving two half-strength
    ghosts. The fix for that is a seam routed around the disagreement
    (``cv2.detail_GraphCutSeamFinder``) or a per-pixel median over three or more
    frames. Feathering hides an exposure step; it cannot hide a contradiction.
    Multi-band blending over an image pyramid is the production answer, and it
    is a strictly better version of the same idea: blend low frequencies over a
    wide band and high frequencies over a narrow one.
    """
    w_base = cv2.distanceTransform(canvas.base_mask.astype(np.uint8), cv2.DIST_L2, 3)
    w_other = cv2.distanceTransform(canvas.other_mask.astype(np.uint8), cv2.DIST_L2, 3)
    total = w_base + w_other
    total = np.maximum(total, 1e-6)
    w_base, w_other = w_base / total, w_other / total

    if canvas.base.ndim == 3:
        w_base, w_other = w_base[..., None], w_other[..., None]

    blended = canvas.base.astype(np.float32) * w_base + canvas.other.astype(np.float32) * w_other
    out = np.clip(blended, 0, 255).astype(np.uint8)

    # Where only one image has pixels the weights already resolve to 1 and 0,
    # but only if that image's mask is non-zero there; the explicit fill covers
    # the strip where the distance transform is 0 on both sides of a one-pixel
    # boundary.
    only_base = canvas.base_mask & ~canvas.other_mask
    only_other = canvas.other_mask & ~canvas.base_mask
    out[only_base] = canvas.base[only_base]
    out[only_other] = canvas.other[only_other]
    return out


def overlap_agreement(canvas: Canvas, erode_px: int = 3) -> dict:
    """How well the two warped images agree where they overlap.

    This is the number that says whether the *geometry* is right, independent of
    any blending choice, and it is what turns "the seam looks fine" into
    something a test can assert. Three statistics over the overlap region:

    * ``ncc`` -- normalised cross-correlation of the two images' grey values.
      Insensitive to a uniform exposure difference between the two shots, which
      is what makes it a measure of alignment rather than of exposure.
    * ``mean_abs_diff`` -- average absolute difference in grey levels.
      Sensitive to exposure, deliberately: read next to the NCC it separates
      "misaligned" from "differently exposed".
    * ``frac_disagree`` -- the fraction of overlap pixels differing by more than
      40 grey levels. Scene motion concentrates this in a few compact blobs; a
      slightly wrong H spreads it as a thin halo along every edge.

    The overlap mask is eroded by ``erode_px`` first. The outermost ring of a
    warped image is interpolated against the border padding, so its values are
    partly invented, and including it drags every statistic in the same
    direction for a reason that has nothing to do with the homography.
    """
    mask = canvas.overlap.astype(np.uint8)
    if erode_px > 0:
        mask = cv2.erode(mask, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
    mask = mask.astype(bool)
    if mask.sum() < 100:
        return {"overlap_px": int(mask.sum()), "ncc": float("nan"),
                "mean_abs_diff": float("nan"), "frac_disagree": float("nan")}

    def grey(a: np.ndarray) -> np.ndarray:
        return a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

    a = grey(canvas.base)[mask].astype(np.float64)
    b = grey(canvas.other)[mask].astype(np.float64)
    a_c, b_c = a - a.mean(), b - b.mean()
    denom = np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
    return {
        "overlap_px": int(mask.sum()),
        "ncc": float((a_c * b_c).sum() / denom) if denom > 0 else float("nan"),
        "mean_abs_diff": float(np.abs(a - b).mean()),
        "frac_disagree": float((np.abs(a - b) > 40).mean()),
    }


def seam_column_profile(pano: np.ndarray, x: int, half_width: int = 40) -> np.ndarray:
    """Mean brightness of each column in a band around ``x``, for plotting.

    A hard paste shows up here as a step function; feathering turns the same
    data into a ramp. Plotting the profile rather than pointing at the picture
    is what makes "the seam is gone" a measurement instead of an opinion.
    """
    grey = pano if pano.ndim == 2 else cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    lo, hi = max(0, x - half_width), min(grey.shape[1], x + half_width)
    return grey[:, lo:hi].astype(np.float64).mean(axis=0)
