# Decisions

One record per real architectural choice: what was decided, what the
alternatives were, why this one, and what it costs. A decision with no cost
listed is a decision that was not really made.

---

## 1. Implement Harris by hand, then prove it equals `cv2.cornerHarris`

**Decision.** `feat/harris.py` builds the structure tensor from `cv2.Sobel` and
`cv2.boxFilter`, computes `R = det(M) − k·tr(M)²` itself, and
`tests/test_harris.py` asserts the result reproduces `cv2.cornerHarris` to a
relative difference below 1e-6 (measured: **5.481e-08**) once multiplied by the
derived factor `s⁴`, where `s = 1 / (2^(ksize−1) · blockSize)`.

**Alternatives.** (a) Call `cv2.cornerHarris` and explain the formula in prose.
(b) Reimplement the Sobel convolution too, so nothing is borrowed.

**Why this one.** (a) leaves the reader with a formula they can recite and a
function they cannot debug. The agreement *is* the lesson: after it, the library
call is visibly your own arithmetic with a scale factor on it, and the reader is
entitled to use it. (b) would re-teach convolution, which is the previous rung
(`cv01-pixels-to-edges`) and adds nothing here — the structure tensor, not the
derivative, is what this chapter is about.

**Cost.** The scale factor has to be derived and explained, which is a paragraph
of arithmetic most tutorials skip; and the test is now coupled to an OpenCV
implementation detail. That coupling is deliberate: if OpenCV changes its
internal Sobel scaling, this test fails, which is exactly when the README's
claim would have become wrong.

---

## 2. Two non-maximum-suppression implementations, one of them wrong on purpose

**Decision.** `peaks_naive_nms` (the `R == dilate(R)` version every tutorial
ships) and `peaks_component_nms` (one point per connected component) both ship,
and the naive one is documented as buggy rather than deleted.

**Alternatives.** Ship only the correct one and describe the bug in prose.

**Why this one.** The bug is *reproducible* and *measurable*: on the default
checkerboard, whose 49 interior corners we know exactly, the naive version
returns 784 inside the board and the component version returns 49. It also
disappears on a blurred image (814 → 226), which is the property that makes it
dangerous — it reproduces on the clean synthetic test and not on photographs, so
it looks like the *test* is wrong. Prose cannot make that land; two numbers can.

**Cost.** A public function that should never be used in production. It is named
`naive`, its docstring says **Buggy** in the first line, and `tests/test_harris.py`
asserts that it still over-counts — if it ever stops, the example built on it is
silently teaching nothing.

---

## 3. Build SIFT stage 1 by hand; call OpenCV for stages 2 to 4

**Decision.** `feat/scalespace.py` implements the Gaussian ladder, the
difference of Gaussians, and the 26-neighbour extremum search. Sub-pixel
quadratic refinement, edge and low-contrast rejection, orientation assignment
and the 128-D descriptor are `cv2.SIFT`.

**Alternatives.** (a) Implement all four stages. (b) Implement none, and describe
scale space in prose with a picture of a pyramid.

**Why this one.** Stage 1 carries the idea that the chapter needs: *scale
invariance comes from the way the point was found, not from a correction applied
afterwards.* Stages 2 to 4 carry no further idea — they are quality of
implementation, and a hand-rolled version would be slower and worse. Drawing the
line explicitly, and then **measuring what is on the other side of it**, is more
honest than either extreme: example 03 shows our bare extrema at 60.7%
repeatability at 0.8× against SIFT's 80.7%, and that gap is what stages 2 to 4
are worth.

**Cost.** The module cannot be described as "an implementation of SIFT", and
readers who wanted a full SIFT will not find one. The docstring says so in its
second paragraph rather than at the bottom.

---

## 4. Synthetic scenes with known ground truth, not photographs

**Decision.** Every image comes from `feat/scenes.py`. `two_views()` builds both
images from one planar scene through two chosen perspective transforms, so
`H_true` is exact.

**Alternatives.** (a) Ship a few photographs. (b) Download a standard dataset
(Oxford affine, HPatches) at first run.

**Why this one.** With a known `H_true`, every claim becomes a number: precision
and recall against a known correspondence, corner error in pixels, an inlier mask
compared against outliers we planted. Without it, a failed stitch is ambiguous —
was the code wrong or the photographs? It also means the repository has no
downloads, no dataset licence, no private study material, and runs identically
on any machine, which is what makes the README's numbers reproducible at all.

**Cost.** Real. Synthetic texture is kinder than a photograph in some ways — no
JPEG artefacts, no lens distortion, no rolling shutter, no motion blur — and
crueller in others: flat regions with genuinely zero gradient, and edges so exact
they create the response plateaus that break naive suppression. The *shapes* of
the curves transfer to real footage; the absolute counts do not. Both README and
the module docstring say so.

---

## 5. Score every match filter by precision **and** recall

**Decision.** `MatchScore` carries `kept`, `correct` and `matchable`, and every
table in the repository prints precision and recall together.

**Alternatives.** Precision alone, which is what most matching demos report.

**Why this one.** Precision alone can always be bought by keeping less. On the
brick wall the ratio test raises precision from 2.9% to 25.0% while the number of
*correct* matches falls from 19 to 5 — the filter is behaving exactly as designed
and the result is a match set too small to fit anything with. That is bug shape 4
("a demo with no control") committed inside a demo about bug shape 4, and the
only defence is to report the number that can disagree.

**Cost.** `matchable` needs a denominator, which means projecting every query
keypoint through the known homography and checking whether any train keypoint
landed near it — an `O(N·M)` distance matrix per scene. Cheap at these sizes,
and impossible without ground truth, which is another argument for decision 4.

---

## 6. Derive the RANSAC threshold from χ² with **2** degrees of freedom

**Decision.** `inlier_threshold(sigma, dof=2)` returns
`sqrt(-2·ln(1-confidence))·sigma`, which is `sqrt(5.99)·sigma ≈ 2.45σ` at 95%.
The 1-DoF case (1.96σ) is implemented too, and documented as correct for a
point-to-*line* residual such as the epipolar distance in `findFundamentalMat`.

**Alternatives.** (a) The widely quoted `1.96·sigma`. (b) A magic "3 to 5 pixels".

**Why this one.** A reprojection residual is the distance between two 2-D points,
so *both* coordinates are noisy and `r²/σ²` is chi-squared with two degrees of
freedom, not one. `1.96` is the two-sided 95% interval for a single scalar — the
right answer to a different question. Example 07 measures the cost of getting it
wrong on the same data: the 1-DoF threshold keeps 14 of 18 genuine inliers
instead of 18. And because the measured inlier ratio then sits in the *exponent*
of the iteration formula, it also inflates the iteration count.

**Cost.** It needs σ, the detector's localisation noise, which you have to
estimate. That is a feature: a magic pixel count needs the same knowledge and
hides the fact that it does.

---

## 7. Adaptive iteration count, with a hard ceiling

**Decision.** `ransac_homography` recomputes `N = ceil(log(1−p)/log(1−wˢ))` from
the best inlier ratio seen so far and stops early, with `max_iterations` as a
ceiling.

**Alternatives.** (a) A fixed 2000 iterations. (b) A fixed 100.

**Why this one.** (a) is waste: on the example-07 fixture the adaptive version
converges in 95 samples where 2000 was budgeted. (b) fails **silently** — at 20%
inliers the formula asks for 2876, so 100 iterations returns a confidently wrong
model with no warning. The ceiling stays because a pathologically contaminated
input can ask for more samples than you can afford, and hitting a bound you set
is better than running forever.

**Cost.** The estimate of `w` early in the run is low, so the adaptive bound
sometimes stops before finding the *best* consensus set. Measured on the
example-07 fixture at the tighter 95% threshold, that costs about one inlier of
18 on half the seeds. That is why the tests assert the exact recovery at the 99%
threshold and only "no outlier admitted, at least 17 of 18 kept" at 95% — which
is what a 95% confidence level *means*, stated rather than papered over.

---

## 8. Refit on the consensus set, iterated to a fixed point

**Decision.** After the sampling loop, refit the DLT on the whole inlier set,
recompute the mask, and repeat until the mask stops changing (bounded at five
passes).

**Alternatives.** (a) Return the four-point model straight out of the loop.
(b) Refit exactly once.

**Why this one.** (a) carries all four sampled points' noise at full strength;
the symptom is a homography correct in kind and a pixel or two off everywhere,
which shows up as a soft double edge in the panorama and gets blamed on the
blender. (b) is the textbook step and is nearly right: a better model admits
inliers the four-point model just missed, and those inliers make the next model
better again. Iterating is the cheap half of what the literature calls
LO-RANSAC; on the example-07 fixture it is worth about one recovered inlier and
a factor of two in corner accuracy.

**Cost.** Up to five extra SVDs per call, which is negligible next to the
sampling loop, and a loop that must be bounded so an oscillating mask cannot
spin forever.

---

## 9. Hartley normalisation inside the DLT, always

**Decision.** `dlt_homography` normalises both point sets to centroid-at-origin
with mean distance √2, solves, and un-normalises. There is no flag to skip it.

**Alternatives.** Make it optional, for speed or for teaching.

**Why this one.** Without it the columns of `A` span five orders of magnitude —
`u·x` runs to 1e5 while its neighbour is 1 — and the SVD spends its precision on
the large columns. The symptom is an `H` that fits its four sample points and
visibly misfits everything else, with no error raised. `tests/test_ransac.py`
measures both versions on points around x = 4000 and asserts the un-normalised
residual is at least 100× worse. It is the single most common reason a
hand-rolled homography "almost works", and a flag to turn it off would be a
loaded gun.

**Cost.** Two extra 3×3 matrix multiplies and an inverse per fit, inside the
RANSAC hot loop. Measured against the SVD it is noise.

---

## 10. Warp *both* images onto the canvas, and take masks from the warp

**Decision.** `warp_onto_canvas` warps the base image by the pure translation `T`
and the other by `T @ H`, and derives each image's coverage mask by warping a
field of 255s rather than testing `pixel > 0` on the result.

**Alternatives.** (a) Slice the base image into the canvas with array indexing.
(b) Build masks with `canvas.sum(axis=2) > 0`.

**Why this one.** (a) leaves one side of the seam resampled and the other not, so
`overlap_agreement` would be measuring an interpolation difference that has
nothing to do with the homography. (b) punches holes in the mask wherever the
scene is genuinely dark — a legitimate black pixel is indistinguishable from
"outside the image", and the resulting blend has speckle in the shadows.
`tests/test_panorama.py` asserts that an entirely black image still covers its
own footprint.

**Cost.** One extra warp of the base image and two extra warps of the mask
fields, per stitch. At these sizes, milliseconds.

---

## 11. Size the canvas from the transformed corners

**Decision.** `canvas_transform` transforms the source image's four corners,
takes the bounding box together with the base image's corners, and prepends the
translation that moves the minimum to (0, 0).

**Alternatives.** The tutorial default `(w_base + w_other, max(h_base, h_other))`.

**Why this one.** The default happens to survive a left-to-right pivot and
nothing else. Any warp that tilts pushes part of the second image above row zero
— measured on the example-08 pair: 31 rows, and 21,384 image pixels silently
cropped. Cropping is the dangerous half, because a panorama with a straight-cut
corner reads as a deliberate crop rather than a bug.

**Cost.** A slightly larger canvas with more empty space in the general case, and
a translation matrix that has to be carried around and applied to anything drawn
in canvas coordinates afterwards.

---

## 12. Feathering, not multi-band blending

**Decision.** `feather` weights each image by its distance to its own nearest
invalid pixel, normalised by the sum of the two weights.

**Alternatives.** (a) Hard paste only. (b) Multi-band (Laplacian pyramid)
blending. (c) `cv2.detail_MultiBandBlender`.

**Why this one.** (a) is kept alongside, because the step it produces is the
visible lesson — measured at 29.9 grey levels against feathering's 6.8. (b) is
the production answer and is a pyramid algorithm, which belongs to a later
chapter; implementing it here would add a second large idea to a chapter that
already has RANSAC in it. (c) would be a call, not a lesson.

**Cost.** Feathering cannot hide a *contradiction*. Where the two images
genuinely disagree it paints half of each, so a mover appears twice at half
strength — which is exactly the ghost the example-08 output describes and does
not fix. The README lists this under limitations rather than hiding it.

---

## 13. Every random source takes an explicit seed

**Decision.** Scene generators take `seed`; `ransac_homography` requires `seed`;
no module ever touches the global `np.random` state.

**Alternatives.** Module-level generators, or `np.random.seed()` at import.

**Why this one.** A module-level generator makes a scene's content depend on how
many other scenes were built first, which turns "the test passed" into "the test
passed in this import order". And an unseeded RANSAC is a regression test that
passes four times out of five. `tests/test_scenes.py` asserts that building a
scene does not advance the global generator, because that coupling is invisible
until it is not.

**Cost.** Every call site has to decide on a seed, and the tests that sweep seeds
(`test_ransac.py` runs 0–7) are eight times slower than a single run. Both are
worth it.

---

## 14. `float64` for matching and geometry, `float32` for images

**Decision.** `knn_l2_numpy` accumulates in float64; `dlt_homography`,
`transfer` and `residuals` work in float64. Image processing stays float32.

**Alternatives.** float32 throughout, matching OpenCV's internals.

**Why this one.** For OpenCV's 512-normalised SIFT descriptors, `a·a` and
`2·a·b` are each around 5e5 and their difference is small, so float32
cancellation puts real error into exactly the small distances that decide a
match. In the DLT the same argument applies to `u·x` terms against constants. On
the image side float32 halves the memory and matches what OpenCV's filters
produce anyway.

**Cost.** A dtype boundary that has to be crossed deliberately, and matching
distances that differ from `cv2.BFMatcher`'s in the sixth decimal — which is why
`tests/test_matching.py` compares them at 1e-3 and says why.
