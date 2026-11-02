#!/usr/bin/env python3
"""The positive control's phase plot (docs/studies/pendulum-positive-control.md, "Outcomes"): descriptive only.

    uv run --with matplotlib==3.10.9 python tools/positive_control_figure.py --run DIR --out FILE.svg [--check]

Top row: each arm's final rollouts for the first qualifying seed, on CCIL's axis contract (e in [-pi, pi) periodic,
omega in [-8, 8]), over that seed's training demonstrations, with the relabeled states on the relabel panel. Bottom
row: per-start upright occupancy, Relabel - BC and Relabel - Duplication, averaged over the qualifying seeds at each
final start (theta about hanging, theta_dot); and every seed's relabeled states. --check exits 1 unless FILE equals a
regeneration.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

from ilsb.tasks.pendulum import Task

INK, MUTED, GRID, SURFACE, DEMO, LABEL = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff", "#1baf7a", "#6a3fb5"
ARMS = ("bc", "duplication", "relabel")
FINAL = 2204


def setup():
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt": "pendulum-positive-control", "svg.fonttype": "none",
                         "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": GRID,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    return plt


def ew(s):
    return np.mod(np.arctan2(s[..., 0], s[..., 1]), 2 * np.pi) - np.pi, s[..., 2]


def contract(ax, title):
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-8, 8)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ["−π", "−π/2", "0 (upright)", "π/2", "π"])
    ax.set_xlabel("e = θ − π (rad), periodic")
    ax.set_ylabel("ω (rad/s)")
    ax.set_title(title, loc="left", color=INK, fontsize=8.5)


def paths(ax, states, color, lw=0.35, alpha=0.6):
    e, w = ew(states)
    for x, y in zip(e, w):
        jump = np.flatnonzero(np.abs(np.diff(x)) > np.pi)
        ax.plot(np.insert(x, jump + 1, np.nan), np.insert(y, jump + 1, np.nan), color=color, lw=lw, alpha=alpha)


def figure(run: Path) -> str:
    from ccil.schema import read_trajectories
    doc = json.loads((run / "positive-control.json").read_text())
    seeds = doc["qualifying_seeds"]
    s0 = seeds[0]
    split = json.loads((run / "data" / f"s{s0}" / "split.json").read_text())
    eps = {e.episode_id: e.observations["state"] for e in read_trajectories(run / "data" / f"s{s0}" / "subset.npz").episodes}
    demos = np.stack([eps[i] for i in split["train"]])
    sel = np.load(run / "select" / f"s{s0}.npz")
    plt = setup()
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 10.4), layout="constrained")
    for ax, arm in zip(axs[0], ARMS):
        st = np.load(run / "evaluate" / f"s{s0}-{arm}-rollout.npz")["states"]
        paths(ax, demos, DEMO, lw=0.5, alpha=0.7)
        paths(ax, st, INK)
        if arm == "relabel":
            e, w = ew(sel["collection/x"])
            ax.plot(e, w, ls="none", marker=".", ms=1.2, color=LABEL, alpha=0.6, rasterized=True)
        occ = doc["outcomes"][str(s0)][arm]["upright_occupancy"]
        contract(ax, f"({'abc'[ARMS.index(arm)]}) {arm}: final rollouts, seed {s0} (black); training demonstrations "
                     f"(green)\n{'relabeled collection states (purple); ' if arm == 'relabel' else ''}"
                     f"upright occupancy {occ:.3f}")
    occ = {arm: np.mean([json.loads((run / "evaluate" / f"s{s}.json").read_text())[arm]["per_start_occupancy"]
                         for s in seeds], 0) for arm in ARMS}
    starts = Task(eval_seed=FINAL).starts()
    th, thd = np.arctan2(starts[:, 0], starts[:, 1]), starts[:, 2]
    for ax, other, tag in ((axs[1, 0], "bc", "d"), (axs[1, 1], "duplication", "e")):
        sc = ax.scatter(th, thd, c=occ["relabel"] - occ[other], cmap="RdBu", vmin=-1, vmax=1, s=22,
                        edgecolors=MUTED, linewidths=0.3)
        ax.set_xlabel("θ₀ (rad, 0 = hanging)")
        ax.set_ylabel("θ̇₀ (rad/s)")
        ax.set_title(f"({tag}) Per-start upright occupancy, relabel − {other},\nmean over {len(seeds)} seeds "
                     f"(blue: relabel higher)", loc="left", color=INK, fontsize=8.5)
        fig.colorbar(sc, ax=ax, shrink=0.8, label="occupancy difference")
    ax = axs[1, 2]
    for s in seeds:
        e, w = ew(np.load(run / "select" / f"s{s}.npz")["collection/x"])
        ax.plot(e, w, ls="none", marker=".", ms=0.8, color=LABEL, alpha=0.25, rasterized=True)
    contract(ax, f"(f) Relabeled collection states, all {len(seeds)} seeds")
    fig.suptitle("Pendulum missing-state positive control (demonstrator relabels at BC-visited off-support states; "
                 f"not CCIL). Decision: {doc['decision']}", x=0.01, ha="left", color=INK, fontsize=10)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None, "Creator": None}, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if json.loads((a.run / "positive-control.json").read_text())["status"] != "completed":
        raise SystemExit("refused: the run did not complete; no phase plot is drawn from it")
    text = figure(a.run)
    if a.check:
        same = a.out.exists() and a.out.read_text() == text
        print("identical" if same else "differs")
        return 0 if same else 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
