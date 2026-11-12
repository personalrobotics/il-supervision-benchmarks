#!/usr/bin/env python3
"""Regenerate the three label-geometry figures of docs/studies/pusht.md.

    uv run --with matplotlib==3.10.9 python tools/pusht_v2/lesson_figures.py draw [--check]

Inputs are the committed records in docs/studies/evidence/pusht/lesson: lesson.json, the
keypoint-space preimage probe, the pose-space probe, the history fit and the demonstrations' rigidity.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EV = REPO / "docs/studies/evidence/pusht"
LESSON = EV / "lesson"

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
KEYPOINT, POSE, DEMO = "#d08a2e", "#2a6fdb", "#1baf7a"  # validated categorical slots (dataviz validate_palette)


def pair_distances(kp):
    """(N, 16) keypoints -> (N, 28) pairwise distances of the 8 points."""
    p = kp.reshape(len(kp), 8, 2)
    i, j = np.triu_indices(8, 1)
    return np.linalg.norm(p[:, i] - p[:, j], axis=-1)


FIGURES = ("pusht-label-t-shapes.svg", "pusht-label-validity.svg", "pusht-history-shift.svg")


def setup():
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt": "pusht-coordinates-lesson", "svg.fonttype": "none",
                         "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": GRID,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 8.5,
                         "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    return plt


def save(plt, fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None, "Creator": None}, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def t_patch(ax, kp, **style):
    """The T as its two rectangles (keypoints 0-3 and 4-7, gym_pusht's order)."""
    from matplotlib.patches import Polygon
    for part in (kp[:4], kp[4:]):
        ax.add_patch(Polygon(part, closed=True, **style))


def t_shapes(plt, lesson, reference):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    rows = (("keypoint", KEYPOINT, "solved in keypoint coordinates"), ("pose", POSE, "solved in pose coordinates, decoded"))
    for r, (space, color, how) in enumerate(rows):
        for c, ex in enumerate(lesson["examples"][space]):
            ax = axes[r, c]
            src, lab = np.asarray(ex["source_previous_t"]), np.asarray(ex["label_previous_t"])
            t_patch(ax, src, facecolor=GRID, edgecolor=MUTED, linewidth=0.8)
            t_patch(ax, lab, facecolor="none", edgecolor=color, linewidth=2.0)
            err = float(np.abs(pair_distances(lab.reshape(1, 16)) - reference).max())
            pts = np.vstack([src, lab])
            lo, hi = pts.min(0), pts.max(0)
            half = (hi - lo).max() / 2 + 12
            mid = (lo + hi) / 2
            ax.set_xlim(mid[0] - half, mid[0] + half)
            ax.set_ylim(mid[1] - half, mid[1] + half)
            ax.set_aspect("equal")
            ax.set_xticks([]), ax.set_yticks([])
            ax.text(0.03, 0.03, f"shape error {err:.3g} px", transform=ax.transAxes, color=INK, fontsize=7.5)
            if c == 0:
                ax.set_ylabel(how, color=INK, fontsize=8)
        axes[r, 0].text(0.03, 0.93, "label's previous T", transform=axes[r, 0].transAxes, color=color, fontsize=7.5,
                        fontweight="bold")
        axes[r, 0].text(0.03, 0.85, "recorded previous T", transform=axes[r, 0].transAxes, color=MUTED, fontsize=7.5)
    fig.suptitle("The previous-frame T of accepted CCIL labels (three drawn at random from each run)", color=INK,
                 fontsize=9, x=0.02, ha="left")
    fig.tight_layout()
    return save(plt, fig)


def validity(plt, lesson, probe, pose_probe, demo_max, tol):
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.4), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    # (a) what becomes of the proposals
    ax = axes[0]
    k = probe["results"]["newton_50"]
    kp_conv = k["converged_fraction"] * 100
    pose_use = pose_probe["usable_fraction"] * 100
    bars = (("keypoint", [(0.0, "usable", KEYPOINT), (kp_conv, "converged, T deformed", "#e9c08a"),
                          (100 - kp_conv, "no root found", GRID)]),
            ("pose", [(pose_use, "usable", POSE), (100 - pose_use, "outside action bounds", GRID)]))
    for y, (name, parts) in enumerate(bars):
        left = 0.0
        for w, label, color in parts:
            if w > 0:
                ax.barh(1 - y, w, left=left, color=color, height=0.55, edgecolor=SURFACE, linewidth=1)
            left += w
        main = max(parts, key=lambda p: p[0])
        ax.text(1.5, 1 - y, f"{name}: {main[0]:.2f}% {main[1]}", va="center", color=INK, fontsize=7.5)
    ax.text(1.5, 0.62, f"converged {kp_conv:.1f}%, every one deformed", color=MUTED, fontsize=7)
    ax.set_xlim(0, 100), ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([]), ax.set_xlabel("% of noisy_action_fe proposals")
    ax.set_title("What becomes of each proposal", loc="left")
    # (b) T-shape error of converged labels
    ax = axes[1]
    kpr, por = k["rigidity_error_converged_px"], pose_probe["rigidity_error_accepted_px"]
    for y, (q, color, name) in enumerate(((kpr, KEYPOINT, "keypoint"), (por, POSE, "pose"))):
        ax.plot([q["10"], q["90"]], [1 - y] * 2, color=color, linewidth=3, solid_capstyle="round")
        ax.plot([q["50"]], [1 - y], "o", color=color, markersize=6, markeredgecolor=SURFACE)
        ax.text(q["50"], 1 - y + 0.22, f"{name}  {q['50']:.3g} px", color=INK, fontsize=7.5, ha="center")
    for x, label, ha in ((demo_max, "demonstrations ", "right"), (tol, " tolerance", "left")):
        ax.axvline(x, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
        ax.text(x, -0.42, label, color=MUTED, fontsize=7, ha=ha)
    ax.set_xscale("log"), ax.set_ylim(-0.6, 1.6), ax.set_yticks([])
    ax.set_xlabel("max pairwise-distance error (px)")
    ax.set_title("Previous-frame T shape, converged labels", loc="left")
    # (c) conditioning
    ax = axes[2]
    for y, (space, color) in enumerate((("keypoint", KEYPOINT), ("pose", POSE))):
        p = lesson["conditioning"][space]["percentiles"]
        n = lesson["conditioning"][space]["size"]
        ax.plot([p["10"], p["90"]], [1 - y] * 2, color=color, linewidth=1.5)
        ax.plot([p["25"], p["75"]], [1 - y] * 2, color=color, linewidth=4, solid_capstyle="round")
        ax.plot([p["50"]], [1 - y], "o", color=color, markersize=6, markeredgecolor=SURFACE)
        ax.text(p["50"], 1 - y + 0.22, f"{space} ({n}x{n})  {p['50']:.3g}", color=INK, fontsize=7.5, ha="center")
    ax.set_xscale("log"), ax.set_ylim(-0.6, 1.6), ax.set_yticks([])
    ax.set_xlabel("condition number, previous-slot Jacobian")
    ax.set_title("How well posed the preimage is", loc="left")
    for a in axes:
        a.grid(axis="x", color=GRID, linewidth=0.5)
        a.set_axisbelow(True)
    fig.tight_layout()
    return save(plt, fig)


def history_shift(plt, fit):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
    for ax, key, title in ((axes[0], "history_shift_px", "The history shift (known exactly)"),
                           (axes[1], "newest_slot_px", "The newest slot (the real dynamics)")):
        for y, (model, color, name) in enumerate((("unstructured", KEYPOINT, "shift learned"),
                                                  ("structured", POSE, "shift exact"))):
            q = fit[model]["errors"]["held_out"][key]
            ax.plot([q["50"], q["90"]], [1 - y] * 2, color=color, linewidth=4, solid_capstyle="round")
            ax.plot([q["99"]], [1 - y], "|", color=color, markersize=9, markeredgewidth=1.5)
            ax.plot([q["50"]], [1 - y], "o", color=color, markersize=6, markeredgecolor=SURFACE)
            ax.text(max(q["50"], 0.0), 1 - y + 0.26, f"{name}: p50 {q['50']:.3g}, p90 {q['90']:.3g}, p99 {q['99']:.3g} px",
                    color=INK, fontsize=7.5)
        ax.set_ylim(-0.6, 1.7), ax.set_yticks([]), ax.set_xlim(left=-0.3)
        ax.set_xlabel("per-transition max error on held-out episodes (px)")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    fig.tight_layout()
    return save(plt, fig)


def draw() -> dict:
    plt = setup()
    lesson = json.loads((LESSON / "lesson.json").read_text())
    probe = json.loads((LESSON / "preimage-probe.json").read_text())
    rig = json.loads((LESSON / "rigidity.json").read_text())
    pose_probe = json.loads((LESSON / "pose-probe.json").read_text())
    fit = json.loads((LESSON / "history-fit.json").read_text())
    reference = np.asarray(rig["reference_pair_distances_px"])
    demo_max = rig["demonstration_error_px"]["100"]
    return dict(zip(FIGURES, (t_shapes(plt, lesson, reference),
                              validity(plt, lesson, probe, pose_probe, demo_max, 10 * demo_max),
                              history_shift(plt, fit))))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--check", action="store_true")
    a = ap.parse_args()
    svgs = draw()
    if a.check:
        same = {n: (EV / n).exists() and (EV / n).read_text() == t for n, t in svgs.items()}
        print(json.dumps(same, indent=1))
        return 0 if all(same.values()) else 1
    for n, t in svgs.items():
        (EV / n).write_text(t)
    print("wrote", ", ".join(FIGURES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
