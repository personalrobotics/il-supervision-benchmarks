"""#66 post hoc, read-only: initial-condition support of the Stage A rollouts.
    LEROBOT_PYTHON tools/pusht_failure/initial_support.py REPO SOURCE.npz RUN_B1 RUN_B2 OUT.json
Per start (identical initial state for both bases and all repeats): distance at t = 0 to the nearest demonstration
state, for the 41-episode r = 0.25 subset and for the 165-episode training pool; decomposed into components."""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(sys.argv[1]); sys.path.insert(0, str(REPO / "tools/pusht_failure"))
import failures as F  # noqa: E402
A, B, S = F.A, F.B, F.S
src = B.load_source(Path(sys.argv[2]))
TAU, LOW = F.TAU, F.LOW
# points: 0 prev agent, 1 cur agent, 2-9 prev T keypoints, 10-17 cur T keypoints
COMP = {"full": slice(0, 18), "current_frame": [1] + list(range(10, 18)), "agent": [1], "T": list(range(10, 18))}
def pool(eps):
    rows = np.flatnonzero(np.isin(src["episode"], eps))
    return S.points(src["state"][rows], src["env"][rows]), src["episode"][rows]
def comp_dist(x, sup, idx):
    """nearest distance using only the points idx (largest per-point distance)."""
    return np.sqrt(((x[:, None, idx] - sup[None][:, :, idx]) ** 2).sum(-1)).max(-1).min(1)
def hist_dist(x, sup):
    """history: the frame-to-frame displacement of the 9 points, against the nearest such displacement."""
    dx = x[:, [1] + list(range(10, 18))] - x[:, [0] + list(range(2, 10))]
    ds = sup[:, [1] + list(range(10, 18))] - sup[:, [0] + list(range(2, 10))]
    return np.sqrt(((dx[:, None] - ds[None]) ** 2).sum(-1)).max(-1).min(1)
sub = A.subset(A.BASE["roles"]["train"], S.R, S.SUBSET_SEED)
pools = {"subset_41": pool(sub)[0], "train_165": pool(A.BASE["roles"]["train"])[0]}
assert len(pools["subset_41"]) == 5040
per_start = {}
for b, run in ((1, sys.argv[3]), (2, sys.argv[4])):
    rec, arrs, _ = F.load_run(Path(run))
    starts = [int(s) for s in arrs["start"]]
    for i, s in enumerate(starts):
        t = F.unpack(arrs, i)
        x0 = F.states_of(t["obs_agent"][:1, 0], t["obs_kp"][:1, 0])[0]
        d = per_start.setdefault(s, {"x0": x0, "success": 0, "consequential": 0, "n": 0})
        assert np.array_equal(d["x0"], x0)  # identical initial state across bases and repeats
        for r in range(F.R_REP):
            x = rec["table"][i * F.R_REP + r]
            assert x["start"] == s
            d["success"] += x["success"]; d["consequential"] += x["consequential"]; d["n"] += 1
starts = sorted(per_start)
X = np.stack([per_start[s]["x0"] for s in starts])
D = {p: {c: comp_dist(X, sup, idx) for c, idx in COMP.items()} | {"history": hist_dist(X, sup)} for p, sup in pools.items()}
succ = np.array([per_start[s]["success"] / per_start[s]["n"] for s in starts])
cons = np.array([per_start[s]["consequential"] / per_start[s]["n"] for s in starts])
def bins(d):
    edges = [0, 15, 30, 45, 60, 90, 1e9]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        out.append({"px": f"{lo}-{hi if hi < 1e9 else 'inf'}", "starts": int(m.sum()),
                    "success": float(succ[m].mean()) if m.any() else None,
                    "consequential_failure": float(cons[m].mean()) if m.any() else None})
    return out
def auc(score, label):
    pos, neg = score[label], score[~label]
    dd = pos[:, None] - neg[None]
    return float(((dd > 0).sum() + 0.5 * (dd == 0).sum()) / dd.size)
mostly_fail = cons >= 0.5
out = {"starts": len(starts), "rollouts_per_start": 8, "tau_px": TAU, "pools": {}}
for p in pools:
    o = {}
    for c, d in D[p].items():
        o[c] = {"quantiles": {str(q): float(v) for q, v in zip((10, 50, 90), np.percentile(d, (10, 50, 90)))},
                "within_tau": int((d <= TAU).sum()),
                "auc_consequential_failure": auc(d, mostly_fail),
                "spearman_success": float(np.corrcoef(np.argsort(np.argsort(d)), np.argsort(np.argsort(succ)))[0, 1])}
    o["full_bins"] = bins(D[p]["full"])
    out["pools"][p] = o
# which component makes a start far: for starts beyond tau on the full metric
for p in pools:
    far = D[p]["full"] > TAU
    out["pools"][p]["far_starts"] = {"n": int(far.sum()),
        **{f"{c}_beyond_tau": int((D[p][c][far] > TAU).sum()) for c in ("agent", "T", "current_frame", "history")}}
# cross-classification of the subset's far, mostly failing starts against the full pool
f41, f165 = D["subset_41"]["full"] > TAU, D["train_165"]["full"] > TAU
out["mostly_failing_starts"] = {"n": int(mostly_fail.sum()),
    "far_from_41_near_165": int((mostly_fail & f41 & ~f165).sum()),
    "far_from_both": int((mostly_fail & f41 & f165).sum()),
    "near_41": int((mostly_fail & ~f41).sum())}
out["rows"] = [{"start": s, "success": float(a), "consequential": float(c),
                **{f"{p}_{k}": float(D[p][k][i]) for p in pools for k in ("full", "agent", "T", "history")}}
               for i, (s, a, c) in enumerate(zip(starts, succ, cons))]
Path(sys.argv[5]).write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
