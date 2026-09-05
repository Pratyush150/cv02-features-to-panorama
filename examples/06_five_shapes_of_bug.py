"""06 -- Five shapes of bug, each one reproduced on purpose.

This is the example the repository is built around. Every one of these is a real
failure that costs real afternoons, each is triggered here deliberately and
deterministically, and each comes with the ten-second check that would have
caught it -- and with the symptom, so it can be recognised from a bug report
alone, with no source in front of you.

    1. partial implementation      cv2.cornerHarris gives you a map, not corners
    2. hidden operator in the name BFMatcher(normType, crossCheck): three morphemes
    3. ambiguous unit              DMatch.distance means two different things
    4. demo with no control        "the ratio test improves matching" -- shown where it cannot lose
    5. fact with an expiry date    "SIFT needs contrib" -- correct until July 2020

The measurements come from :mod:`feat.bugs`, which is also what
``tests/test_bugs.py`` asserts against, so a demonstration of a bug cannot
quietly stop demonstrating it as the library underneath moves.

Run:  python examples/06_five_shapes_of_bug.py
Figure: docs/figures/06_five_shapes_of_bug.png
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  -- puts src/ on sys.path; see the file for why

import textwrap

import cv2
import numpy as np
from matplotlib import pyplot as plt

from feat import bugs, describe, figures, harris, matching, scenes


def text_panel(ax, title: str, lines: list[str], size: float = 7.2) -> None:
    """A monospaced block of text as a figure panel.

    Some of these shapes are *printouts*, not pictures. Shape 5 is literally
    "ask the interpreter and read the answer", and drawing a bar chart of it
    would be decoration standing in for the evidence.
    """
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=size,
            va="top", ha="left", transform=ax.transAxes, linespacing=1.45)


def main() -> None:
    figures.use_teaching_style()

    reports = bugs.all_shapes()
    for report in reports:
        print(report.render())
        print()

    # Only three of the five reports are quoted in panel titles; the other two
    # are drawn straight from the live measurements below.
    one, _, three, _, five = reports

    fig = plt.figure(figsize=(13.5, 9.5), constrained_layout=True)
    grid = fig.add_gridspec(3, 2)

    # ---- shape 1: the partial implementation ------------------------------
    ax = fig.add_subplot(grid[0, 0])
    board = scenes.checkerboard(blur=False)
    response = harris.harris_response(board.image.astype(np.float32))
    naive = harris.peaks_naive_nms(response, 0.01)
    fixed = harris.peaks_component_nms(response, 0.01)
    # A crop, because at full size the clumps are three pixels across and the
    # whole point is invisible. Two real corners, zoomed until you can count.
    x0, y0, side = 108, 108, 92
    crop = board.image[y0:y0 + side, x0:x0 + side]
    figures.show_gray(ax, crop, "")
    inside = lambda p: p[(p[:, 0] >= x0) & (p[:, 0] < x0 + side) & (p[:, 1] >= y0) & (p[:, 1] < y0 + side)]  # noqa: E731
    n_pts, f_pts = inside(naive), inside(fixed)
    ax.scatter(n_pts[:, 0] - x0, n_pts[:, 1] - y0, s=46, facecolors="none",
               edgecolors="#c1272d", linewidths=1.2, label=f"naive R == dilate(R): {len(naive)}")
    ax.scatter(f_pts[:, 0] - x0, f_pts[:, 1] - y0, s=180, marker="+",
               color="#1a9641", linewidths=1.8, label=f"one point per component: {len(fixed)}")
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
    ax.set_title("1. partial implementation -- cornerHarris returns a MAP\n"
                 f"true answer inside the board: {board.n_interior} corners; "
                 f"naive NMS finds {two_or(one, 'naive R == dilate(R), interior')}")

    # ---- shape 2: the hidden operator -------------------------------------
    ax = fig.add_subplot(grid[0, 1])
    pair = scenes.two_views(scenes.textured_wall())
    fq = describe.detect_describe(pair.img_b, "orb", 1200)
    ft = describe.detect_describe(pair.img_a, "orb", 1200)
    matchable = matching.count_matchable(fq.points, ft.points, pair.H_true)
    bars = {}
    for label, norm in (("NORM_HAMMING\n(right)", cv2.NORM_HAMMING), ("NORM_L2\n(wrong)", cv2.NORM_L2)):
        knn = cv2.BFMatcher(norm).knnMatch(fq.descriptors, ft.descriptors, k=2)
        score = matching.score_matches(label, matching.ratio_test(knn, 0.75),
                                       fq.points, ft.points, pair.H_true, matchable=matchable)
        bars[label] = score
    x = np.arange(len(bars))
    ax.bar(x - 0.17, [100 * s.precision for s in bars.values()], 0.34,
           color="#3b6ea5", label="precision")
    ax.bar(x + 0.17, [100 * s.recall for s in bars.values()], 0.34,
           color="#c1272d", label="recall")
    for i, s in enumerate(bars.values()):
        ax.text(i - 0.17, 100 * s.precision, f"{100 * s.precision:.0f}%", ha="center",
                va="bottom", fontsize=8)
        ax.text(i + 0.17, 100 * s.recall, f"{100 * s.recall:.0f}%", ha="center",
                va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(list(bars), fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("%")
    ax.set_title("2. hidden operator -- L2 on binary descriptors runs SILENTLY\n"
                 "precision barely moves; recall halves. Watch the wrong number and you see nothing.")
    ax.legend(fontsize=8, loc="upper right")

    # ---- shape 3: the ambiguous unit --------------------------------------
    ax = fig.add_subplot(grid[1, 0])
    for method, colour in (("orb", "#1a9641"), ("sift", "#3b6ea5")):
        n = 1200 if method == "orb" else 600
        a = describe.detect_describe(pair.img_b, method, n)
        b = describe.detect_describe(pair.img_a, method, n)
        knn = cv2.BFMatcher(describe.metric_for(a.descriptors)).knnMatch(
            a.descriptors, b.descriptors, k=2
        )
        d1 = np.array([p[0].distance for p in knn if len(p) == 2])
        ax.hist(d1, bins=60, alpha=0.7, color=colour, density=True,
                label=f"{method.upper()}: {'Hamming bits' if method == 'orb' else 'L2, ~512-norm'}")
    ax.axvline(50, color="#111111", ls="--", lw=1.3)
    ax.annotate("one threshold,\ntwo meanings:\nm.distance < 50", xy=(50, 0), xytext=(70, 0.006),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="#111111", lw=0.9))
    ax.set_xlabel("DMatch.distance -- the same attribute name in both cases")
    ax.set_ylabel("density")
    ax.set_title("3. ambiguous unit -- a fixed cut keeps "
                 f"{three.numbers['fixed cut d1 < 50 keeps (orb)']} of one and "
                 f"{three.numbers['fixed cut d1 < 50 keeps (sift)']} of the other\n"
                 f"a spread of {three.numbers['spread, fixed cut (x)']:.0f}x, against "
                 f"{three.numbers['spread, ratio test (x)']:.0f}x for the dimensionless ratio test")
    ax.legend(fontsize=8)

    # ---- shape 4: the demo with no control --------------------------------
    ax = fig.add_subplot(grid[1, 1])
    rows = []
    for label, scene in (("distinctive\nclutter", scenes.textured_wall()),
                         ("repeated\nbrick wall", scenes.brick_wall())):
        p = scenes.two_views(scene)
        a = describe.detect_describe(p.img_b, "sift", 800)
        b = describe.detect_describe(p.img_a, "sift", 800)
        m = matching.count_matchable(a.points, b.points, p.H_true)
        knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(a.descriptors, b.descriptors, k=2)
        raw = matching.score_matches("raw", [q[0] for q in knn if len(q) == 2],
                                     a.points, b.points, p.H_true, matchable=m)
        rat = matching.score_matches("ratio", matching.ratio_test(knn, 0.75),
                                     a.points, b.points, p.H_true, matchable=m)
        rows.append((label, raw, rat))
    x = np.arange(len(rows))
    width = 0.2
    series = [
        ("raw NN: correct matches", [r[1].correct for r in rows], "#bbbbbb"),
        ("ratio test: correct matches", [r[2].correct for r in rows], "#1a9641"),
    ]
    for i, (name, values, colour) in enumerate(series):
        ax.bar(x + (i - 0.5) * 2 * width, values, 2 * width, label=name, color=colour)
        for xi, v in zip(x + (i - 0.5) * 2 * width, values):
            ax.text(xi, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel("number of CORRECT matches")
    ax.set_title("4. demo with no control -- the ratio test raises precision on both scenes\n"
                 "and on the control scene it throws away most of the correct matches too")
    ax.legend(fontsize=8)

    # ---- shape 5: the fact with an expiry date ----------------------------
    ax = fig.add_subplot(grid[2, 0])
    def clip(value: str, width: int = 46) -> str:
        # The SURF error message is 120 characters long and would run off the
        # panel and over the neighbouring one. Truncated for the figure only --
        # the full text is in the terminal output above.
        value = str(value)
        return value if len(value) <= width else value[: width - 3] + "..."

    text_panel(
        ax,
        "5. fact with an expiry date -- ask the interpreter, not the internet",
        [f"{k:<34}{clip(v)}" for k, v in five.numbers.items()]
        + ["", "SIFT is at the cv2 TOP LEVEL: it comes from main OpenCV, not contrib.",
           "SURF is bound and does not construct: the binding ships in contrib,",
           "the implementation is compiled out behind OPENCV_ENABLE_NONFREE.",
           "Both OpenCV wheels are installed at different MAJOR versions and",
           "`import cv2` silently resolves to one of them. Nothing warns you."],
        size=6.6,
    )

    # ---- the table --------------------------------------------------------
    ax = fig.add_subplot(grid[2, 1])
    table = ["shape                     the ten-second detector",
             "-" * 78]
    for report in reports:
        # textwrap, not a slice: slicing a string every 52 characters splits
        # words in half ("paramet / ers"), which is unreadable and looks like
        # a rendering bug rather than a deliberate wrap.
        wrapped = textwrap.wrap(report.detector, width=48)
        table.append(f"{report.number}. {report.shape:<28}{wrapped[0]}")
        for line in wrapped[1:]:
            table.append(f"{'':<31}{line}")
        table.append("")
    text_panel(ax, "learn the right-hand column -- that is the transferable part", table, size=6.4)

    fig.suptitle(
        "Five shapes of bug, reproduced on purpose -- each with the ten-second check that catches it",
        fontsize=12,
    )
    print(f"wrote {figures.save(fig, '06_five_shapes_of_bug.png')}")


def two_or(report, key):
    """Small helper so the panel title can quote a measured number, not a guess."""
    return report.numbers.get(key, "?")


if __name__ == "__main__":
    main()
