#!/usr/bin/env python3
"""The recovery-segment study's phase portrait (docs/studies/pendulum-recovery-segments.md, "Required Stage A
artifacts"): descriptive only; the decisions are stage-a.json's.

    uv run --with matplotlib==3.10.9 python tools/recovery_segments_figure.py --out DIR --svg FILE.svg [--check]

DIR is a completed Stage A run. For the first design seed, at the selected M (the largest ladder M when none is
selected): (a) the axis contract with the training demonstrations, the provisional BC's failure trajectories and the
failure states (segment starts); (b) the near-upright inset (e in [-0.5, 0.5], omega in [-3, 3]) with the capture
region shaded and the accepted learned-model segments coloured by check-model disagreement / tau_demo; (c) the same
inset with the exact-arm segments from the same starts, and the rejected learned starts; (d) the data-only discovery
diagnostic: the failure states of the provisional BC rolled out in the learned model beside the real held-out failures.
Bottom row: per gate and M, the number of design seeds passing, (e) learned arm, (f) exact arm. --check exits 1 unless
FILE equals a regeneration.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recovery_segments as rs  # noqa: E402
from positive_control_figure import DEMO, GRID, INK, contract, ew, paths, setup  # noqa: E402

FAIL, START, EXACT, REJECT, FOUND = "#d08a2e", "#0b0b0b", "#2a6fdb", "#d1495b", "#6a3fb5"


def inset(ax, title):
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-3, 3)
    ax.axvspan(-rs.CAPTURE, rs.CAPTURE, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.set_xlabel("e = θ − π (rad)")
    ax.set_ylabel("ω (rad/s)")
    ax.set_title(title, loc="left", color=INK, fontsize=8.5)


def segs(ax, states, color, lw=0.6, alpha=0.8, cmap=None, values=None, norm=None):
    e, w = ew(states)
    for i, (x, y) in enumerate(zip(e, w)):
        c = cmap(norm(values[i])) if cmap is not None else color
        ax.plot(x, y, color=c, lw=lw, alpha=alpha)
        ax.plot(x[-1:], y[-1:], ls="none", marker=".", ms=2, color=c)


def figure(out: Path) -> str:
    doc = json.loads((out / "stage-a.json").read_text())
    s = doc["seeds"][0]
    M = doc["selected_M"] or doc["ladder"][-1]
    D = rs.demos(out, s)
    F, _ = rs.failures(out, s)
    L = np.load(out / "generate" / f"s{s}-M{M}-learned.npz")
    X = np.load(out / "generate" / f"s{s}-M{M}-exact.npz")
    tau = float(L["tau_demo"])
    kappa = np.maximum(L["check_one_step"], L["check_composed"]) / tau
    plt = setup()
    from matplotlib.colors import LogNorm
    fig, axs = plt.subplots(2, 4, figsize=(20, 9.5), layout="constrained", height_ratios=[1.4, 1])

    ax = axs[0, 0]
    paths(ax, np.stack([D["S"][k] for k in D["train"]]), DEMO, lw=0.6, alpha=0.9)
    roll = F["collection/rollouts"]
    paths(ax, roll[F["collection/fail"]], FAIL, lw=0.35, alpha=0.5)
    e, w = ew(F["collection/x"])
    ax.plot(e, w, ls="none", marker=".", ms=1.5, color=START)
    contract(ax, f"(a) seed {s}: demonstrations (green),\nprovisional-BC failure trajectories (orange, "
                 f"{int(F['collection/fail'].sum())}) and failure states (black, {len(F['collection/x'])})")

    ax = axs[0, 1]
    acc = L["accepted"]
    norm = LogNorm(1e-2, 1e1)
    cmap = plt.get_cmap("viridis")
    segs(ax, L["states"][acc], None, cmap=cmap, values=np.clip(kappa[acc], 1e-2, 1e1), norm=norm)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label="check-model disagreement / τ_demo (gate G5 at 0.25)")
    inset(ax, f"(b) accepted learned-model segments, M = {M} ({int(acc.sum())} of {len(acc)});\n"
              f"grey band: capture region |e| < {rs.CAPTURE}")

    ax = axs[0, 2]
    xa = X["accepted"]
    segs(ax, X["states"][xa], EXACT, lw=0.5, alpha=0.6)
    e, w = ew(L["x_f"][~acc])
    ax.plot(e, w, ls="none", marker="x", ms=2.5, color=REJECT, label=f"learned starts rejected ({int((~acc).sum())})")
    ax.legend(loc="lower left", frameon=False)
    inset(ax, f"(c) exact-dynamics segments from the same starts\n({int(xa.sum())} accepted)")

    ax = axs[0, 3]
    disc = np.load(out / "discovery" / f"s{s}.npz")["rollouts"]
    _, idx = rs.failure_states(disc)
    found = disc.reshape(-1, 3)[idx]
    e, w = ew(F["held_out/x"])
    ax.plot(e, w, ls="none", marker=".", ms=2, color=FAIL, label=f"real held-out failures ({len(F['held_out/x'])})")
    e, w = ew(found)
    ax.plot(e, w, ls="none", marker=".", ms=2, color=FOUND, alpha=0.6,
            label=f"failures found in the learned model ({len(found)})")
    ax.legend(loc="lower left", frameon=False)
    r = doc["discovery"]["per_seed"][str(s)]
    cov = r["held_out_failures_covered"]
    inset(ax, f"(d) data-only discovery: real held-out failures\nwithin ρ_cov of a model failure: "
              f"{'n/a' if cov is None else f'{cov:.2f}'}")

    gates = {"learned": rs.ARM_GATES["learned"], "exact": rs.ARM_GATES["exact"]}
    for ax, arm, tag in ((axs[1, 0], "learned", "e"), (axs[1, 1], "exact", "f")):
        G = gates[arm]
        grid = np.array([[sum(doc["gates"][str(t)][str(m)][arm][g] for t in doc["seeds"]) for m in doc["ladder"]]
                         for g in G])
        ax.imshow(grid, cmap="Greys", vmin=0, vmax=len(doc["seeds"]) * 1.6, aspect="auto")
        for i in range(len(G)):
            for j in range(len(doc["ladder"])):
                ax.text(j, i, str(grid[i, j]), ha="center", va="center", color=INK, fontsize=8)
        ax.set_xticks(range(len(doc["ladder"])), [f"M={m}" for m in doc["ladder"]])
        ax.set_yticks(range(len(G)), G)
        ax.set_title(f"({tag}) {arm} arm: design seeds passing each gate (of {len(doc['seeds'])}; "
                     f"an M passes at ≥ {min(rs.PASS_SEEDS, len(doc['seeds']))} on every gate)",
                     loc="left", color=INK, fontsize=8.5)
    for ax in axs[1, 2:]:
        ax.set_axis_off()
    head = (f"selected M: {doc['selected_M']}" + (f"; {doc['stop']}" if doc["stop"] else "")
            + f"; discovery: {doc['discovery']['verdict']}")
    fig.suptitle(f"Pendulum targeted recovery segments, Stage A (exploratory; M > 1 is beyond the paper). {head}",
                 x=0.01, ha="left", color=INK, fontsize=10)
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
    if not (a.out / "stage-a.json").exists():
        raise SystemExit("refused: no stage-a.json; no portrait is drawn before the gates")
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
