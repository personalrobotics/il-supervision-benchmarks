#!/usr/bin/env python3
"""The Pendulum recovery-tube study's phase portraits (docs/studies/pendulum-recovery-tube.md, "Running and
artifacts"): descriptive only; the decisions are decision.json's.

    uv run --with matplotlib==3.10.9 python tools/recovery_tube_figure.py --out DIR --svg FILE.svg [--check]

DIR is a completed recovery_tube.py output (geometry and decide done). Row 1: (a) the references, anchors and
disturbance end-states by dose bin; (b) one training seed's synthetic CCIL states coloured by their displacement from
source (the displacements, q95 = 0.005, are below a pixel on these axes, so no arrows are drawn; the tangent and
transverse split is in geometry.json); (c) recovery versus achieved Delta per arm, mean over seeds, with the label
tube's q50, q95 and maximum and the tau* reference line. Rows 2-3: per-state recovery (mean over seeds, rho = q95)
at every disturbed state on the common axes, per arm, and the CCIL - BC and replay-CCIL - duplication differences.
Every plotted value is read from geometry.npz, geometry/s{seed}.npz and evaluate/{cohort}/{arm}-s{seed}.npz.
--check exits 1 unless FILE equals a regeneration.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from positive_control_figure import GRID, INK, MUTED, contract, ew, paths, setup  # noqa: E402

ARMS = {"bc": ("corrected", "#8a8984"), "ccil": ("corrected", "#1baf7a"),
        "duplication": ("matched", "#d08a2e"), "replay_ccil": ("matched", "#6a3fb5")}
TAGS = {"bc": "d", "ccil": "e", "duplication": "g", "replay_ccil": "h"}
SEED = 1201  # the synthetic-state panel's training seed


def recovery_mean(out: Path, cohort: str, arm: str) -> np.ndarray | None:
    files = sorted((out / "evaluate" / cohort).glob(f"{arm}-s*.npz"))
    return np.mean([np.load(f)["recovered"] for f in files], 0) if files else None


def figure(out: Path) -> str:
    g = json.loads((out / "geometry.json").read_text())
    dec = json.loads((out / "decision.json").read_text())
    z = np.load(out / "geometry.npz")
    refs, x0, delta, bins = z["refs"], z["x0"], z["delta"], z["bin"]
    pooled, tau = g["label_geometry"]["pooled"], g["tau_star_reference"]
    rec = {arm: recovery_mean(out, c, arm) for arm, (c, _) in ARMS.items()}
    dist = bins != "undisturbed"
    plt = setup()
    from matplotlib.colors import LogNorm
    fig, axs = plt.subplots(3, 3, figsize=(16, 15), layout="constrained")

    ax = axs[0, 0]
    paths(ax, refs, GRID, lw=0.5, alpha=1.0)
    order = sorted(set(bins[dist]), key=lambda b: (b != "null", b != "inside", b != "edge", b))
    cmap = plt.get_cmap("viridis", len(order))
    for i, b in enumerate(order):
        e, w = ew(x0[bins == b])
        ax.plot(e, w, ls="none", marker=".", ms=1.5, color=cmap(i), label=f"{b} ({(bins == b).sum()})",
                rasterized=True)
    e, w = ew(x0[~dist])
    ax.plot(e, w, ls="none", marker="o", ms=2.5, mfc="none", mec=INK, mew=0.5,
            label=f"anchors, undisturbed ({(~dist).sum()})")
    ax.legend(loc="lower left", fontsize=6.5, markerscale=3, frameon=False)
    contract(ax, "(a) π_D references (grey), anchors and disturbance end-states by dose bin")

    ax = axs[0, 1]
    paths(ax, refs, GRID, lw=0.5, alpha=1.0)
    s = np.load(out / "geometry" / f"s{SEED}.npz")
    keep = np.arange(0, len(s["d_source"]), 20)
    e, w = ew(s["z"][keep])
    sc = ax.scatter(e, w, c=np.maximum(s["d_source"][keep], 1e-7), s=0.5, cmap="viridis", rasterized=True,
                    norm=LogNorm(1e-7, pooled["max"]), lw=0)
    fig.colorbar(sc, ax=ax, label="d(synthetic state, source state)")
    contract(ax, f"(b) synthetic CCIL states, seed {SEED} (every 20th of {len(s['d_source'])}), "
                 f"by displacement\nfrom source; pooled q95 = {pooled['q95']:.4f}, max = {pooled['max']:.4f}")

    ax = axs[0, 2]
    edges = np.geomspace(delta[delta > 0].min(), delta.max() * 1.0001, 25)
    which = np.digitize(delta, edges)
    mid = np.sqrt(edges[1:] * edges[:-1])
    for arm, (_, color) in ARMS.items():
        if rec[arm] is None:
            continue
        y = [rec[arm][dist & (which == i)].mean() if (dist & (which == i)).any() else np.nan
             for i in range(1, len(edges))]
        same = arm == "replay_ccil" and rec["ccil"] is not None and np.array_equal(rec[arm], rec["ccil"])
        ax.plot(mid, y, color=color, lw=1.2 if same else 2, ls="--" if same else "-", marker="o", ms=3,
                label=f"{arm} (identical to ccil)" if same else arm, zorder=3 if same else 2)
    for i, (v, name) in enumerate(((pooled["q50"], "q50"), (pooled["q95"], "q95 = ρ"), (pooled["max"], "max"),
                                   (tau, "τ*"))):
        ax.axvline(v, color=MUTED, lw=0.8, ls=":" if name != "q95 = ρ" else "--")
        ax.text(v, 0.97 - 0.04 * (i % 2), f" {name} ", color=MUTED, fontsize=7, ha="right" if i == 1 else "left",
                va="top",
                transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("achieved Δ = d(disturbed state, reference)")
    ax.set_ylabel("recovery to ρ = q95 (mean over seeds and states)")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.9))
    ax.set_title("(c) recovery versus achieved Δ by log bin; vertical lines: label-tube quantiles and τ*",
                 loc="left", color=INK, fontsize=8.5)

    def rmap(ax, v, title, diff=False):
        paths(ax, refs, GRID, lw=0.5, alpha=1.0)
        e, w = ew(x0[dist])
        o = np.argsort(np.abs(v[dist]) if diff else -v[dist])
        im = ax.scatter(e[o], w[o], c=v[dist][o], s=2, lw=0, rasterized=True,
                        cmap="RdBu" if diff else "Greys", vmin=-1 if diff else 0, vmax=1)
        contract(ax, title)
        return im

    im = None
    for ax, arm in zip([axs[1, 0], axs[1, 1], axs[2, 0], axs[2, 1]], ("bc", "ccil", "duplication", "replay_ccil")):
        if rec[arm] is None:
            ax.set_axis_off()
            continue
        im = rmap(ax, rec[arm], f"({TAGS[arm]}) {arm}: recovery per disturbed state")
    if im is not None:
        fig.colorbar(im, ax=[axs[1, 0], axs[1, 1], axs[2, 0], axs[2, 1]], shrink=0.6,
                     label="fraction of seeds recovering (ρ = q95)")
    dm = None
    for ax, (a, b), tag in ((axs[1, 2], ("ccil", "bc"), "f"), (axs[2, 2], ("replay_ccil", "duplication"), "i")):
        if rec[a] is None or rec[b] is None:
            ax.set_axis_off()
            continue
        dm = rmap(ax, rec[a] - rec[b], f"({tag}) {a} − {b}: recovery difference per disturbed state", diff=True)
    if dm is not None:
        fig.colorbar(dm, ax=[axs[1, 2], axs[2, 2]], shrink=0.6, label="difference in recovering fraction")
    head = "; ".join(f"{k}: {dec[k]['outcome']} ({dec[k]['mean']:+.3f}, 95% [{dec[k]['interval'][0]:+.3f}, "
                     f"{dec[k]['interval'][1]:+.3f}])" for k in ("fitness", "signal", "attribution") if k in dec)
    fig.suptitle(f"Pendulum recovery tube (exploratory). {head}", x=0.01, ha="left", color=INK, fontsize=10)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None, "Creator": None}, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--svg", type=Path, required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not (a.out / "decision.json").exists():
        raise SystemExit("refused: no decision.json; no portrait is drawn before the decisions")
    text = figure(a.out)
    if a.check:
        same = a.svg.exists() and a.svg.read_text() == text
        print("identical" if same else "differs")
        return 0 if same else 1
    a.svg.parent.mkdir(parents=True, exist_ok=True)
    a.svg.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
