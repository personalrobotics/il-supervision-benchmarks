import sys, numpy as np
from pathlib import Path
sys.path.insert(0, "tools")
import diagnostics as D
from ilsb import run
R = Path.home()/"code/il-supervision-benchmarks/runs"
before = D.entries(R/"20261031T141026Z-j-r-k/store")
ctx = run.Context(Path("studies/pendulum-j-r-k/study.yaml"), R/"20261031T160500Z-novelty-probe-root", R/"20261031T141026Z-j-r-k/store", allow_dirty=False)
s, c, _ = D.arm_view(ctx, "ccil_exact", 0.1, 1, 1301, 1e-5)
demo, syn = s.source == "demonstration", s.source == "synthetic"
zd, ad = s.observations[demo].astype(np.float64), s.actions[demo].astype(np.float64)
zs, as_ = s.observations[syn].astype(np.float64), s.actions[syn].astype(np.float64)
print("shapes", zd.shape, ad.shape, zs.shape, as_.shape, "bound", ctx.task.action_bound, s.observations.dtype)
scale = zd.std(axis=0); b = ctx.task.action_bound
X = lambda z, a: np.floor(np.hstack([z/scale, a/b])/0.1).astype(np.int64)
cd, cs = X(zd, ad), X(zs, as_)
by = {(str(e), int(f)): i for i, (e, f) in enumerate(zip(s.episode_id[demo], s.frame[demo]))}
src = np.array([by[(str(e), int(f))] for e, f in zip(s.episode_id[syn], s.frame[syn])])
diff = (cs != cd[src]).any(axis=1)
print("rows whose cell differs from own source:", diff.sum(), "of", len(diff))
print("which columns differ:", (cs != cd[src])[diff].sum(axis=0))
i = np.flatnonzero(diff)[:3]
print("z syn", zs[i]); print("z src", zd[src[i]]); print("a syn", as_[i].ravel(), "a src", ad[src[i]].ravel())
print("novel cells", len({tuple(x) for x in cs} - {tuple(x) for x in cd}))
print("store unchanged", D.entries(R/"20261031T141026Z-j-r-k/store") == before)
