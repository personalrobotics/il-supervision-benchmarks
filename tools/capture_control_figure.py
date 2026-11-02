#!/usr/bin/env python3
"""The capture-region positive control's phase plot (docs/studies/pendulum-capture-control.md, "Outcomes"): descriptive.

    uv run --with matplotlib==3.10.9 python tools/capture_control_figure.py --run DIR --out FILE.svg [--check]

Both rows are for the first qualifying seed, near upright (e in [-0.5, 0.5], omega in [-3, 3]), with the LQR branch
|e| < 0.1 shaded. Top row: each arm's final rollouts, with the relabeled capture states on the relabel panel. Bottom
row: each arm's action field minus the demonstrator's, pi - pi_D, on a grid. --check exits 1 unless FILE equals a
regeneration.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from positive_control_figure import GRID, INK, LABEL, MUTED, ew, paths, setup  # noqa: E402

from ilsb.learners import vector_bc  # noqa: E402
from ilsb.tasks.pendulum import expert  # noqa: E402

ARMS = ("bc", "duplication", "relabel")
E_LIM, W_LIM, LQR = 0.5, 3.0, 0.1


def near(ax, title):
    ax.set_xlim(-E_LIM, E_LIM)
    ax.set_ylim(-W_LIM, W_LIM)
    ax.axvspan(-LQR, LQR, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.set_xlabel("e = θ − π (rad); shaded: the LQR branch |e| < 0.1")
    ax.set_ylabel("ω (rad/s)")
    ax.set_title(title, loc="left", color=INK, fontsize=8.5)


def figure(run: Path) -> str:
    doc = json.loads((run / "capture-control.json").read_text())
    s0 = doc["qualifying_seeds"][0]
    sel = np.load(run / "select" / f"s{s0}.npz")
    e, w = np.meshgrid(np.linspace(-E_LIM, E_LIM, 101), np.linspace(-W_LIM, W_LIM, 121))
    grid = np.stack([np.sin(e + np.pi), np.cos(e + np.pi), w], -1).reshape(-1, 3)
    plt = setup()
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 10.4), layout="constrained")
    for i, arm in enumerate(ARMS):
        ax = axs[0, i]
        paths(ax, np.load(run / "evaluate" / f"s{s0}-{arm}-rollout.npz")["states"], INK)
        if arm == "relabel":
            xe, xw = ew(sel["collection/x"])
            ax.plot(xe, xw, ls="none", marker=".", ms=1.5, color=LABEL, alpha=0.6, rasterized=True)
        hold = doc["outcomes"][str(s0)][arm]["final_hold"]
        near(ax, f"({'abc'[i]}) {arm}: final rollouts near upright, seed {s0}"
                 f"{'; relabeled capture states (purple)' if arm == 'relabel' else ''}\nfinal hold {hold:.2f}")
        pol = vector_bc.load(run / "policies" / arm / f"s{s0}", "cpu")
        diff = (np.clip(np.asarray(pol(grid)).reshape(-1), -3, 3) - expert(grid)).reshape(e.shape)
        ax = axs[1, i]
        im = ax.pcolormesh(e, w, diff, cmap="RdBu", vmin=-6, vmax=6, shading="nearest", rasterized=True)
        near(ax, f"({'def'[i]}) {arm}: action field minus the demonstrator's, π − π_D (rad/s²), seed {s0}")
        ax.axvspan(-LQR, LQR, fill=False, ec=MUTED, lw=0.6)
    fig.colorbar(im, ax=axs[1, :], shrink=0.8, label="π − π_D (rad/s²); white: agreement")
    fig.suptitle("Pendulum capture-region positive control (exploratory; demonstrator relabels at BC-visited LQR-branch "
                 f"states; not CCIL). Decision: {doc['decision']}", x=0.01, ha="left", color=INK, fontsize=10)
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
    if json.loads((a.run / "capture-control.json").read_text())["status"] != "completed":
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
