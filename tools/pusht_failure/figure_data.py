"""#66 post hoc, read-only: data for the geometry figure (figure.py draws it).
    LEROBOT_PYTHON tools/pusht_failure/figure_data.py REPO SOURCE.npz WINDOWS.npz RUN_B1 RUN_B2 INITIAL_SUPPORT.json OUT.npz
WINDOWS.npz is #52 Stage A's oracle CCIL windows; INITIAL_SUPPORT.json is initial_support.py's output."""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(sys.argv[1]); sys.path.insert(0, str(REPO / "tools/pusht_failure")); sys.path.insert(0, str(REPO / "tools/pusht_v2"))
import failures as F  # noqa: E402
import pose  # noqa: E402
A, B, S = F.A, F.B, F.S
src = B.load_source(Path(sys.argv[2]))
demo = F.demonstrations(src)
sup = demo["pts"]
_env = A.make_env(); _env._setup()
out = {"goal": np.array(_env.goal_pose, np.float64)}
# CCIL windows (#52 Stage A oracle labels): distance of the labelled two-frame state to its demonstration state
w = np.load(sys.argv[3])
obs = w["window/observation"].astype(np.float64)  # (N, 2 frames, 18): agent 2 + keypoints 16
pts = np.concatenate([obs[:, :, 0:2], obs[:, 0, 2:18].reshape(-1, 8, 2), obs[:, 1, 2:18].reshape(-1, 8, 2)], 1)
first = {e: int(np.flatnonzero(src["episode"] == e)[0]) for e in np.unique(w["window/episode"])}
row = np.array([first[int(e)] + int(a) - 1 for e, a in zip(w["window/episode"], w["window/anchor_frame"])])
dpt = S.points(src["state"][row], src["env"][row])
assert np.allclose(dpt[:, 1], pts[:, 1], atol=1e-3) and np.allclose(dpt[:, 10:], pts[:, 10:], atol=1e-2)  # current frame = demo
out["label_dist"] = S.dist(pts, dpt)
# one demonstration episode with many labels
e0 = int(np.bincount(w["window/episode"]).argmax())
r0 = np.flatnonzero(src["episode"] == e0)
sel = np.flatnonzero(w["window/episode"] == e0)
out.update(demo_agent=src["state"][r0, 2:4].astype(np.float64), demo_T=src["pose"][r0, 3:6].astype(np.float64),
           demo_action=src["action"][r0].astype(np.float64), lab_row=row[sel] - r0[0],
           lab_prev_agent=obs[sel, 0, 0:2], lab_cur_agent=obs[sel, 1, 0:2], lab_correction=w["window/action"][sel, 1].astype(np.float64),
           lab_dist=out["label_dist"][sel], demo_episode=np.array(e0))
# all rollouts: closest approach and outcome
mins, kind = [], []
recs = {}
for b, run in ((1, sys.argv[4]), (2, sys.argv[5])):
    rec, arrs, _ = F.load_run(Path(run)); recs[b] = (rec, arrs)
    for x in rec["table"]:
        mins.append(x["d_sup"]["min"]); kind.append(0 if x["success"] else 2 if x["consequential"] else 1)
out["roll_min"], out["roll_kind"] = np.array(mins), np.array(kind)  # 0 success, 1 near miss, 2 consequential
# two example starts with similar initial distance: one always failing, one always succeeding
init = {r["start"]: r for r in json.load(open(sys.argv[6]))["rows"]}
fail = [s for s, r in init.items() if r["consequential"] == 1.0]
win = [s for s, r in init.items() if r["success"] == 1.0]
med = float(np.median([init[s]["subset_41_full"] for s in win]))
sf = min(fail, key=lambda s: abs(init[s]["subset_41_full"] - med)); sw = min(win, key=lambda s: abs(init[s]["subset_41_full"] - med))
for tag, s in (("fail", sf), ("win", sw)):
    paths, dts = [], []
    for b in (1, 2):
        rec, arrs = recs[b]
        i = [int(v) for v in arrs["start"]].index(s); t = F.unpack(arrs, i)
        for r in range(F.R_REP):
            e = int(t["end"][r])
            paths.append(t["pos"][:e + 1, r])
            d, j = F.nearest(F.states_of(t["obs_agent"][:e + 1, r], t["obs_kp"][:e + 1, r]), sup)
            dts.append(d)
            if b == 1 and r == 0:
                j0 = int(j[0]); jr = int(demo["rows"][j0])
                out[f"{tag}_nearest0"] = np.r_[src["state"][jr, 2:4], src["pose"][jr, 3:6]].astype(np.float64)
    out[f"{tag}_start"] = np.array(s)
    for k, (p, d) in enumerate(zip(paths, dts)):
        out[f"{tag}_path{k}"], out[f"{tag}_d{k}"] = p, d
out["tau"] = np.array(F.TAU)
np.savez(sys.argv[7], **out)
print("episode", e0, "labels", len(sel), "label dist q50/q99/max", np.percentile(out["label_dist"], [50, 99, 100]).round(2), "starts", sf, sw, "d0", init[sf]["subset_41_full"], init[sw]["subset_41_full"])
