#!/usr/bin/env python3
"""Regenerate the three descriptive figures in the canonical Pendulum account.

    uv run --with matplotlib==3.10.9 python tools/pendulum_figures.py draw [--check]

The compact committed records are the complete inputs. The evidence-heavy tree
that produced them is preserved at tag ``pendulum-evidence-archive-v1``.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EV = REPO / "docs/studies/evidence"
OUT = EV / "pendulum"
SUMMARY = OUT / "summary.json"
STAGE_A = EV / "pendulum-recovery-segments/stage-a.json"
CAPTURE_E = 0.1
INK, MUTED, GRID, SURFACE, DEMO = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff", "#1baf7a"
EXACT, LEARNED = "#2a6fdb", "#d08a2e"
FIGURES = ("pendulum-phase-space.svg", "pendulum-reach-trust.svg", "pendulum-supervision-contrasts.svg")


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


# ---- figures ----

def phase_space(plt, S: dict) -> str:
    from matplotlib.colors import LogNorm
    P = S["phase"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained", width_ratios=[1.35, 1])
    norm, cmap = LogNorm(1e-2, 1e1), plt.get_cmap("viridis")
    f = P["failures"]
    d = np.clip(f["d_demo_over_tau"], 1e-2, 1e1)
    for ax in axs:
        ax.axvspan(-CAPTURE_E, CAPTURE_E, color=GRID, alpha=0.8, lw=0, zorder=0)
        for e, w in P["demonstrations"]:
            e, w = np.array(e), np.array(w)
            jump = np.flatnonzero(np.abs(np.diff(e)) > np.pi)
            ax.plot(np.insert(e, jump + 1, np.nan), np.insert(w, jump + 1, np.nan), color=DEMO, lw=0.7, alpha=0.9,
                    rasterized=True)
        ax.scatter(f["e"], f["omega"], c=d, cmap=cmap, norm=norm, s=2, lw=0, rasterized=True, zorder=3)
    n, c = len(f["e"]), P["collection"]
    contract(axs[0], f"(a) {len(P['demonstrations'])} training demonstrations (green) and {n} provisional-BC failure "
                     f"states\nfrom {c['failure_trajectories']} failed of {c['episodes']} rollouts; grey band: capture "
                     f"region |e| < {CAPTURE_E}")
    ax = axs[1]
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("e = θ − π (rad)")
    ax.set_ylabel("ω (rad/s)")
    med = float(np.median(f["d_demo_over_tau"]))
    ax.set_title(f"(b) near upright: the failure states lie on the demonstrations' own trajectories\n"
                 f"(median distance to the nearest demonstration state {med:.2f} τ_demo; "
                 f"{np.mean(np.array(f['d_demo_over_tau']) >= 1):.1%} at ≥ 1 τ_demo)", loc="left", color=INK,
                 fontsize=8.5)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, shrink=0.9,
                 label="failure state's distance to the nearest demonstration state / τ_demo")
    fig.suptitle(f"Where the demonstrations and BC failures lie. Recovery segments (#21), Stage A, design seed "
                 f"{P['source']['seed']}; τ_demo = {P['tau_demo']:.3f}", x=0.01, ha="left", color=INK, fontsize=10)
    return save(plt, fig)


def reach_trust(plt, S: dict, A: dict) -> str:
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.4), layout="constrained")
    B = S["persistent_bias"]["seeds"]

    def ladder(ax, M, Y, color, marker, label, open_mask=None):
        Y = np.array([[np.nan if v is None else v for v in y] for y in Y], float)
        for i, y in enumerate(Y):
            ax.plot(M, y, color=color, lw=0.6, alpha=0.35)
            if open_mask is not None:
                o = open_mask[i]
                ax.plot(np.asarray(M)[~o], y[~o], ls="none", marker=marker, ms=4, color=color, alpha=0.8)
                ax.plot(np.asarray(M)[o], y[o], ls="none", marker=marker, ms=4, mfc="white", mec=color, alpha=0.8)
        ax.plot(M, np.nanmedian(Y, 0), color=color, lw=2, marker=marker, ms=5, label=label)

    ax = axs[0]
    Ms = sorted({int(m) for b in B.values() for m in b["ladder"]})
    eta = [[b["ladder"][str(m)]["eta_reach"] for m in Ms] for b in B.values()]
    untrusted = np.array([[not b["ladder"][str(m)]["trust_off_demonstrations"] for m in Ms] for b in B.values()])
    ladder(ax, Ms, eta, LEARNED, "o", f"median over {len(B)} seeds", untrusted)
    ax.axhline(1, color=INK, lw=0.8, ls="--")
    ax.text(Ms[0], 1.04, "beyond demonstration support", color=INK, fontsize=7.5, va="bottom")
    ax.set_ylabel("reach η = median start distance / τ_demo")
    ax.set_title("(a) persistent-bias segments, learned-dynamics CCIL (#171):\nreach grows with M but stays below 1; "
                 "open markers: the two models\ndisagree off the demonstrations (trust fails)", loc="left",
                 color=INK, fontsize=8.5)

    seeds, ladder_M = A["seeds"], A["ladder"]
    g = lambda arm, key: [[A["gates"][str(s)][str(m)][arm][key] for m in ladder_M] for s in seeds]
    ax = axs[1]
    for arm, color, marker in (("exact", EXACT, "s"), ("learned", LEARNED, "o")):
        ladder(ax, ladder_M, g(arm, "novelty_median_d_demo_over_tau"), color, marker, f"{arm}-dynamics CCIL")
    ax.axhline(1, color=INK, lw=0.8, ls="--")
    ax.text(ladder_M[0], 1.04, "novelty gate G4", color=INK, fontsize=7.5, va="bottom")
    ax.set_ylabel("novelty = median row distance to demonstrations / τ_demo")
    ax.set_title("(b) targeted segments from BC failures (#21): rows stay\non demonstration support at every M, "
                 "for exact and\nlearned dynamics alike", loc="left", color=INK, fontsize=8.5)
    ax.legend(loc="lower left", frameon=False, title="thick: median over design seeds")

    ax = axs[2]
    ladder(ax, ladder_M, g("learned", "kappa_over_tau"), LEARNED, "o", "learned-dynamics CCIL")
    ax.axhline(0.25, color=INK, lw=0.8, ls="--")
    ax.text(ladder_M[0], 0.255, "agreement gate G5", color=INK, fontsize=7.5, va="bottom")
    ax.set_ylim(0, 0.3)
    ax.set_ylabel("check-model disagreement κ / τ_demo")
    ax.set_title("(c) #21: the primary and check models agree\nwith each other at every M…", loc="left", color=INK,
                 fontsize=8.5)

    ax = axs[3]
    ladder(ax, ladder_M, g("learned", "replay_fraction"), LEARNED, "o", "learned-dynamics CCIL")
    ax.axhline(0.95, color=INK, lw=0.8, ls="--")
    ax.text(ladder_M[-1], 0.94, "exact-replay gate G1", color=INK, fontsize=7.5, va="top", ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("fraction of accepted segments that replay exactly")
    ax.set_title("(d) …yet fewer of their segments replay under the\ntrue dynamics as M grows: agreement is not trust",
                 loc="left", color=INK, fontsize=8.5)
    for ax in axs:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
        ax.set_xlabel("recovery horizon M (recorded 20 ms action intervals)")
    for ax in axs[:2]:
        ax.set_yscale("log")
    fig.suptitle("Why longer recovery segments do not add useful coverage on Pendulum: reach grows with M, "
                 "but novelty and trust do not", x=0.01, ha="left", color=INK, fontsize=10)
    return save(plt, fig)


ROWS = (  # (study label, record, contrast key, row label)
    ("pumping control", "pendulum-positive-control/positive-control.json", "relabel-bc/upright_occupancy",
     "demonstrator relabeling − BC, upright occupancy"),
    ("pumping control", "pendulum-positive-control/positive-control.json", "relabel-duplication/upright_occupancy",
     "demonstrator relabeling − duplication, upright occupancy"),
    ("capture control", "pendulum-capture-control/capture-control.json", "relabel-bc/final_hold",
     "demonstrator relabeling − BC, final hold"),
    ("capture control", "pendulum-capture-control/capture-control.json", "relabel-duplication/final_hold",
     "demonstrator relabeling − duplication, final hold"),
    ("recovery tube", "pendulum-recovery-tube/decision.json", "signal",
     "learned-dynamics CCIL − BC, edge recovery"),
    ("recovery tube", "pendulum-recovery-tube/decision.json", "attribution.exposure_duplication_minus_bc",
     "duplication − BC, edge recovery"),
    ("recovery tube", "pendulum-recovery-tube/decision.json", "attribution",
     "learned-dynamics CCIL (replayed) − duplication, edge recovery"),
)


def contrast(rec: dict, key: str) -> dict:
    if "/" in key:
        return rec["contrasts"][key]
    for k in key.split("."):
        rec = rec[k]
    return rec


def contrasts(plt) -> str:
    fig, ax = plt.subplots(figsize=(10, 4.4), layout="constrained")
    rows = [(st, lab, contrast(json.loads((EV / path).read_text()), key)) for st, path, key, lab in ROWS]
    y = np.arange(len(rows))[::-1]
    for yi, (st, lab, c) in zip(y, rows):
        lo, hi = c["interval"]
        excl = lo > 0 or hi < 0
        ax.plot([lo, hi], [yi, yi], color=INK, lw=1.4)
        ax.plot(c["mean"], yi, marker="o", ms=6, color=INK, mfc=INK if excl else "white")
        ax.text(hi + 0.02, yi, f"{c['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}], n = {c['n']}", va="center", fontsize=7.5,
                color=MUTED)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, [f"{lab}  ({st})" for st, lab, _ in rows])
    ax.set_xlim(-0.4, 1.6)
    ax.set_xlabel("paired difference in success fraction (95% interval across seeds); filled: interval excludes 0")
    for b in (1.5, 3.5):
        ax.axhline(len(rows) - 1 - b, color=GRID, lw=0.8)
    ax.set_title("Exact-dynamics CCIL has no closed-loop result: #21 stopped at Stage A, before any training.",
                 loc="left", color=MUTED, fontsize=8)
    fig.suptitle("BC, duplication and corrective supervision on Pendulum: each study's predeclared contrasts",
                 x=0.01, ha="left", color=INK, fontsize=10)
    return save(plt, fig)


def save(plt, fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None, "Creator": None}, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def draw() -> dict:
    plt = setup()
    plt.rcParams["svg.hashsalt"] = "pendulum-account"
    S = json.loads(SUMMARY.read_text())
    A = json.loads(STAGE_A.read_text())
    return dict(zip(FIGURES, (phase_space(plt, S), reach_trust(plt, S, A), contrasts(plt))))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--check", action="store_true")
    a = ap.parse_args()
    svgs = draw()
    if a.check:
        same = {n: (OUT / n).exists() and (OUT / n).read_text() == t for n, t in svgs.items()}
        print(json.dumps(same, indent=1))
        return 0 if all(same.values()) else 1
    for n, t in svgs.items():
        (OUT / n).write_text(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
