"""#66 post hoc: draw evidence/pusht-failure/pusht-failure-geometry.svg from figure_data.py's output.
    uv run --no-project --with matplotlib==3.10.9 --with numpy python tools/pusht_failure/figure.py REPO DATA.npz OUT.svg
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
sys.path.insert(0, sys.argv[1] + "/tools/pusht_v2")
import pose  # noqa: E402
z = np.load(sys.argv[2])
OUT = sys.argv[3]
INK, MUTED, GRID = "#1f2328", "#59636e", "#d0d7de"
FAIL, WIN, LAB, GOAL = "#D55E00", "#0072B2", "#7B3FA0", "#009E73"
TAU = float(z["tau"])
plt.rcParams.update({"svg.hashsalt": "pusht-failure-geometry", "svg.fonttype": "none", "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})

def tee(ax, p, **kw):
    kp = pose.keypoints(np.atleast_2d(p))[0].reshape(8, 2)
    for part in (kp[:4], kp[4:]):
        ax.add_patch(Polygon(part, closed=True, **kw))

def scene(ax):
    ax.set_xlim(0, 512); ax.set_ylim(0, 512); ax.set_aspect("equal")
    ax.set_xticks([0, 256, 512]); ax.set_yticks([0, 256, 512])
    tee(ax, z["goal"], facecolor=GOAL, alpha=0.18, edgecolor="none")

fig = plt.figure(figsize=(15, 10), facecolor="white")
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.8], hspace=0.32, wspace=0.18)

# (a) CCIL labels on one demonstration
ax = fig.add_subplot(gs[0, 0]); scene(ax)
da, dT = z["demo_agent"], z["demo_T"]
for k in range(0, len(dT), 25):
    tee(ax, dT[k], facecolor="none", edgecolor=MUTED, lw=0.8, alpha=0.6)
ax.plot(*da.T, color=INK, lw=1.2)
lr, lp = z["lab_row"], z["lab_prev_agent"]
ax.scatter(*lp.T, s=6, color=LAB, zorder=3)
ax.set_title("(a) CCIL labels hug the demonstration")
ax.text(8, 492, f"demo episode {int(z['demo_episode'])}: agent path (black), T every 25 steps\n"
        f"{len(lr)} oracle CCIL labels (purple)",
        va="top", fontsize=8.5, color=MUTED)
# inset zoom on a dense stretch
k0 = int(np.median(lr)); c = da[k0]
ins = ax.inset_axes([0.60, 0.03, 0.38, 0.38])
h = 30
ins.set_xlim(c[0] - h, c[0] + h); ins.set_ylim(c[1] - h, c[1] + h); ins.set_aspect("equal")
ins.plot(*da.T, color=INK, lw=1.2, marker="o", ms=2.5)
m = (np.abs(lp[:, 0] - c[0]) < h) & (np.abs(lp[:, 1] - c[1]) < h)
for p, r in zip(lp[m], lr[m]):
    q = da[r - 1]
    ins.plot([p[0], q[0]], [p[1], q[1]], color=LAB, lw=0.8)
ins.scatter(*lp[m].T, s=14, color=LAB, zorder=3)
ins.set_xticks([]); ins.set_yticks([])
for s in ins.spines.values():
    s.set_edgecolor(LAB)
ins.plot([c[0] + h - 2 - TAU, c[0] + h - 2], [c[1] - h + 4] * 2, color=INK, lw=2)
ins.text(c[0] + h - 2 - TAU / 2, c[1] - h + 6, "τ = 30 px", ha="center", fontsize=8, color=INK)
ins.text(0.04, 0.96, "60 px zoom", transform=ins.transAxes, va="top", fontsize=8, color=LAB)
ax.indicate_inset_zoom(ins, edgecolor=LAB)

# (b), (c) two starts with matched initial distance
for col, tag, color, word in ((1, "fail", FAIL, "fails"), (2, "win", WIN, "succeeds")):
    ax = fig.add_subplot(gs[0, col]); scene(ax)
    paths = [z[f"{tag}_path{k}"] for k in range(8)]
    tee(ax, paths[0][0, 2:5], facecolor="none", edgecolor=INK, lw=1.6)
    n0 = z[f"{tag}_nearest0"]
    tee(ax, n0[2:5], facecolor="none", edgecolor=MUTED, lw=1.0, ls="--")
    ax.scatter(*n0[:2], s=40, facecolor="none", edgecolor=MUTED, lw=1.0)
    for p in paths:
        ax.plot(p[:, 0], p[:, 1], color=color, lw=0.7, alpha=0.55)
        tee(ax, p[-1, 2:5], facecolor="none", edgecolor=color, lw=0.9, alpha=0.8)
    ax.scatter(*paths[0][0, :2], s=40, color=INK, zorder=4)
    ax.set_title(f"({'b' if col == 1 else 'c'}) Start {int(z[f'{tag}_start'])} {word} on 8/8 rollouts")
    ax.text(8, 492, f"start (black); nearest demo state (dashed), {float(z[f'{tag}_d0'][0]):.0f} px\n"
            f"8 rollouts: agent paths, final T; goal green",
            va="top", fontsize=8.5, color=MUTED)

# (d) distance to support over time
ax = fig.add_subplot(gs[1, 0:2])
ax.axhspan(0, float(z["label_dist"].max()), color=LAB, alpha=0.10, lw=0)
ax.axhline(TAU, color=INK, lw=1, ls="--")
for tag, color in (("fail", FAIL), ("win", WIN)):
    for k in range(8):
        d = z[f"{tag}_d{k}"]
        ax.plot(np.arange(len(d)), d, color=color, lw=1.0, alpha=0.75)
ax.set_yscale("log"); ax.set_ylim(1, 300); ax.set_xlim(0, 300)
ax.set_xlabel("step"); ax.set_ylabel("distance to nearest demo state (px)")
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_title("(d) Both starts begin ~70 px from support; only the succeeding one ever reaches it")
ax.text(296, TAU * 1.08, "τ: CCIL's reach", ha="right", va="bottom", fontsize=8.5, color=INK)
ax.text(296, 1.15, "where CCIL labels lie (≤ 13 px)", ha="right", va="bottom", fontsize=8.5, color=LAB)
ax.text(150, 150, f"start {int(z['fail_start'])}: 8 rollouts, never below τ", color=FAIL, fontsize=9)
ax.text(60, 1.8, f"start {int(z['win_start'])}: joins the demonstrations, then succeeds", color=WIN, fontsize=9)

# (e) closest approach, all rollouts
ax = fig.add_subplot(gs[1, 2])
bins = np.logspace(0, np.log10(300), 40)
mins, kind = z["roll_min"], z["roll_kind"]
for vals, color, name in ((z["label_dist"], LAB, "CCIL labels"),
                          (mins[kind == 0], WIN, "successes"),
                          (mins[kind == 2], FAIL, "consequential failures")):
    c, _ = np.histogram(np.clip(vals, 1, 299), bins=bins)
    ax.stairs(c / c.max(), bins, lw=1.6, color=color,
              label=f"{name} (n = {len(vals):,}): median {np.median(vals):.0f} px")
ax.axvline(TAU, ymax=0.66, color=INK, lw=1, ls="--")
ax.set_xscale("log"); ax.set_xlabel("closest approach to a demo state (px)"); ax.set_yticks([])
ax.set_title("(e) All 1,600 rollouts and 1,865 labels")
ax.set_ylim(0, 1.55); ax.set_ylabel("rollouts or labels (each scaled to its peak)")
ax.legend(fontsize=8, frameon=False, loc="upper left")
fig.suptitle("PushT (r = 0.25): consequential failures never come near the demonstrations, where CCIL's labels live",
             x=0.06, ha="left", fontsize=13, fontweight="bold", color=INK, y=0.97)
fig.savefig(OUT, format="svg", metadata={"Date": None, "Creator": None}, dpi=150, bbox_inches="tight")
print(OUT)
