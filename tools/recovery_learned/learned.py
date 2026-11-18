"""#73 Stage 3 on Pendulum: the learned models, the frozen policy, and one start's solve and evaluation
(recovery-learned.md, stage-3.json).

Stage 2's Pendulum system (tools/recovery_feasibility/pendulum.py) supplies the starts, metric, envelope, target and
the exact and refined maps; Stage 2's LM solver and scheduler are reused unchanged. A start is solved (LM through
model 0, the only use of the solver) and then evaluated from its actions alone, so the audit recomputes everything
but the solve from the stored actions.

Branches. From x_0 at anchor time t, n = 500 - t steps. Continuation is pi from x_0; a recovery branch executes its
prefix actions, then pi. Upright is counted on the state before each action (ilsb.tasks.pendulum's score). A branch
under model k steps x + delta_k(x, u); under the exact map it steps the evaluator map. Actions are clipped to the
bound, as the task applies them.

Precision. The learned models are CCIL's float32 networks, evaluated here in float64 (the float32 weights are
exactly representable). Stage 2's LM uses a 1e-6 forward-difference step, below float32 resolution, so a float32
evaluation would give the solver a noise Jacobian. Every model rollout (construction, model validity, the estimated
advantages, model error) uses the same float64 evaluation.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "recovery_feasibility"))  # after this directory: run and audit are Stage 3's
import common as C  # noqa: E402
import pendulum as P2  # noqa: E402

SPEC = json.loads((HERE / "stage-3.json").read_text())
SID = C.SYSTEMS["pendulum"]
T = SPEC["relevance"]["horizon"]
HOLD_FROM = T - 100
CHECK = tuple(SPEC["models"]["check"])
USEFUL = SPEC["relevance"]["useful_threshold_steps"]


def key(smoke: bool, *rest) -> list:
    return [171, 3, SID, *([C.SMOKE_KEY] if smoke else []), *rest]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ccil_commit() -> str | None:
    """The installed CCIL's git commit (prlccil, or ccil before the rename), from its direct_url.json."""
    from importlib import metadata
    for name in ("prlccil", "ccil"):
        try:
            text = metadata.distribution(name).read_text("direct_url.json")
        except metadata.PackageNotFoundError:
            continue
        return json.loads(text or "{}").get("vcs_info", {}).get("commit_id")
    return None


def upright(x) -> np.ndarray:
    from ilsb.tasks.pendulum import UPRIGHT_COS
    x = np.asarray(x)
    return np.cos(np.arctan2(x[..., 0], x[..., 1]) - np.pi) >= UPRIGHT_COS


def gym_cost(x, u) -> np.ndarray:
    e = P2.wrap(P2.theta(x) - np.pi)
    return e ** 2 + 0.1 * np.asarray(x)[..., 2] ** 2 + 0.001 * np.asarray(u) ** 2


# --- the models and the policy -----------------------------------------------------------------------------------

def fit_models(sy, out: Path) -> dict:
    """Fit the five members in one deterministic CPU group (stage-3.json "models"); write models/k.npz and fit
    records. Returns their identities."""
    from ccil.models.mlp_batched import FitJob, fit_group, model_sha256
    train = sy.identity["split"]["train"]
    jobs, eps = [], []
    for k in range(SPEC["models"]["members"]):
        chosen = list(train) if k == SPEC["models"]["builder"] else \
            [train[i] for i in C.rng(171, 3, k, 1).choice(len(train), len(train), replace=True)]
        z = np.concatenate([sy.ep[e][0][:-1] for e in chosen])
        a = np.concatenate([sy.ep[e][1][:-1] for e in chosen])[:, None]
        zn = np.concatenate([sy.ep[e][0][1:] for e in chosen])
        seed = int(np.random.SeedSequence([171, 3, k]).generate_state(1)[0])
        jobs.append(FitJob(z.astype(np.float32), a.astype(np.float32), zn.astype(np.float32), seed))
        eps.append(chosen)
    res = fit_group(jobs, SPEC["models"]["settings"], device="cpu", deterministic=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    ident = {}
    for k, (model, rec) in enumerate(res):
        path = out / "models" / f"{k}.npz"
        tmp = path.with_name(f".{k}.tmp.npz")
        model.save(tmp)
        tmp.replace(path)
        C.write_atomic(path.with_suffix(".json"), C.dumps(rec))
        ident[str(k)] = {"episodes": [str(e) for e in eps[k]], "seed": jobs[k].seed, "weights_sha256": model_sha256(model),
                         "file_sha256": sha256(path)}
    return ident


def load_models(out: Path, ident: dict) -> list:
    from ccil.models.mlp import MLPTransitionModel
    from ccil.models.mlp_batched import model_sha256
    models = []
    for k in range(SPEC["models"]["members"]):
        path = out / "models" / f"{k}.npz"
        if sha256(path) != ident[str(k)]["file_sha256"]:
            sys.exit(f"refused: model {k} is not the run's")
        m = MLPTransitionModel.load(path, "cpu")
        if model_sha256(m) != ident[str(k)]["weights_sha256"]:
            sys.exit(f"refused: model {k}'s weights are not the run's")
        models.append(m)
    return models


def nets64(models: list) -> list:
    """Each model's dynamics network, in float64 and eval mode."""
    import copy
    out = []
    for m in models:
        if m.structured:
            sys.exit("refused: a structured (history) model; Pendulum's is not")
        out.append(copy.deepcopy(m.net).double().eval())
    return out


def delta64(net, z, a) -> np.ndarray:
    import torch
    with torch.no_grad():
        return net.predict(torch.as_tensor(np.asarray(z, np.float64)),
                           torch.as_tensor(np.asarray(a, np.float64).reshape(len(z), 1))).numpy()


def load_policy(path: Path):
    if sha256(path) != SPEC["policy"]["sha256"]:
        sys.exit("refused: the base policy is not the protocol's frozen BC")
    from ilsb.learners.vector.vector_bc import VectorBCPolicy
    pol, _, _ = VectorBCPolicy.load(path)
    return pol


def validation_error(sy, models: list) -> dict:
    """Each member's one-step error d(x + delta(x, u), x') over the validation episodes: q50, q95, max."""
    val = sy.identity["split"]["validation"]
    z = np.concatenate([sy.ep[e][0][:-1] for e in val])
    a = np.concatenate([sy.ep[e][1][:-1] for e in val])
    zn = np.concatenate([sy.ep[e][0][1:] for e in val])
    out = {}
    for k, net in enumerate(nets64(models)):
        d = sy.metric(z + delta64(net, z, a), zn)
        out[str(k)] = {"q50": float(np.percentile(d, 50)), "q95": float(np.percentile(d, 95)), "max": float(d.max())}
    return out


# --- the study object ------------------------------------------------------------------------------------------

class Study:
    def __init__(self, smoke: bool, pendulum: Path, policy: Path, run: Path, stage2: Path):
        import torch
        torch.set_num_threads(1)
        self.smoke = smoke
        self.sy = P2.System(smoke, pendulum)
        self.sy.smoke = smoke
        self.pol = load_policy(policy)
        from ilsb.tasks.pendulum import expert
        self.expert = expert
        self.stage2 = stage2
        self.stage2_starts = json.loads((stage2 / "record.json").read_text())["starts_sha256"]
        ident =json.loads((run / "identity.json").read_text())["identity"]
        self.models = load_models(run, ident["models"])
        self.nets = nets64(self.models)
        self.units = C.start_ids(len(self.sy.anchors))

    # one-step maps on (B, 3) states and (B,) actions; actions clipped as the task applies them
    def step(self, dyn, x, u):
        u = np.clip(u, -self.sy.ubar, self.sy.ubar)
        if dyn == "exact":
            return self.sy.task.step(x, u)[0], u
        return x + delta64(self.nets[dyn], x, u), u

    def pi(self, x) -> np.ndarray:
        return np.asarray(self.pol.predict(np.asarray(x, np.float32)), np.float64).reshape(len(x))

    def branches(self, dyn, x0, prefixes: list, n: int, policy="pi") -> list:
        """Roll every prefix (an (M,) action array; length 0 is continuation) then the policy, n steps from x0,
        all rows in one batch. Per row: upright steps, final hold and Gymnasium return."""
        follow = self.pi if policy == "pi" else self.expert
        B = len(prefixes)
        L = np.array([len(p) for p in prefixes])
        U = np.zeros((B, max(1, L.max())))
        for i, p in enumerate(prefixes):
            U[i, :len(p)] = p
        x = np.repeat(np.asarray(x0, np.float64)[None], B, 0)
        up = np.zeros(B, np.int64)
        hold = np.ones(B, bool)
        gym = np.zeros(B)
        for k in range(n):
            ok = upright(x)
            up += ok
            if k >= HOLD_FROM - (T - n):
                hold &= ok
            u = np.where(k < L, U[:, min(k, U.shape[1] - 1)], follow(x))
            x_next, applied = self.step(dyn, x, u)
            gym += gym_cost(x, applied)
            x = x_next
        return [{"upright": int(up[i]), "hold": bool(hold[i]), "gym_return": float(-gym[i])} for i in range(B)]

    def model_rollout(self, k: int, x0, U) -> np.ndarray:
        """(R, M) actions -> (R, M + 1, 3) states under model k."""
        U = np.atleast_2d(U)
        x = np.repeat(np.asarray(x0, np.float64)[None], len(U), 0)
        out = [x]
        for j in range(U.shape[1]):
            x, _ = self.step(k, x, U[:, j])
            out.append(x)
        return np.stack(out, 1)

    # --- the start ---------------------------------------------------------------------------------------------
    def start(self, unit):
        s, a, r, d = unit
        """The Stage 2 start, from its start file. Provenance: the file's sha256 is the one Stage 2's record lists.
        Content: the start recomputed here agrees within 1e-12 (the last bits differ across hosts)."""
        path = self.stage2 / "starts" / f"{s:03d}.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.stage2_starts[f"{s:03d}"]:
            raise ValueError(f"start {s}'s Stage 2 file is not the one Stage 2's record lists")
        rec2 = json.loads(path.read_text())
        x0, _ = C.available(lambda g: self.sy.propose(a, r, g), C.rng(*C.key(SID, self.smoke, s)))
        if x0 is None or rec2["anchor_key"] != list(self.sy.anchors[a]) or \
                np.abs(np.asarray(rec2["x0"]) - x0).max() > 1e-12:
            raise ValueError(f"start {s} does not reproduce its Stage 2 record")
        return np.asarray(rec2["x0"], np.float64), rec2

    def solve(self, unit) -> dict:
        """LM through model 0 at every M: {M: (8, M) actions}."""
        s, a, _, _ = unit
        x0, _ = self.start(unit)
        out = {}
        for M in self.sy.ladder:
            y, demo = self.sy.target(a, M), self.sy.anchor(a)[1][:M]
            res = lambda U: self.sy.residual(self.model_rollout(0, x0, U)[:, -1], y)  # noqa: E731
            seed = lambda k, M=M: C.rng(*key(self.smoke, s, M, k))  # noqa: E731
            out[str(M)] = C.floats(C.lm(res, M, self.sy.ubar, C.lm_starts(demo, self.sy.ubar, seed)))
        return out

    def evaluate(self, unit, actions: dict) -> dict:
        """Everything from the start and its stored actions: checks, branches, advantages, selections."""
        s, a, r, d = unit
        sy = self.sy
        x0, rec2 = self.start(unit)
        t = sy.anchors[a][1]
        n = T - t
        rec = {"start": s, "anchor": a, "anchor_key": list(sy.anchors[a]), "radius": r, "direction": d,
               "x0": C.floats(x0), "t": int(t), "n": int(n)}
        rungs, rows = {}, []  # rows: (M, restart) of every model-valid restart, in branch order
        for M in sy.ladder:
            y = sy.target(a, M)
            segs = []
            for i, U in enumerate(np.asarray(actions[str(M)], np.float64)):
                ms = self.model_rollout(0, x0, U[None])[0]
                es = sy.rollout(x0, U[None])[0]
                fine = sy.refined_rollout(x0, U)
                checks = {"bounds": sy.check_bounds(U), "model_envelope": sy.check_envelope(ms),
                          "model_endpoint": bool(sy.metric(ms[-1], y) <= sy.eps)}
                model_valid = all(checks.values())
                exact = {"bounds": checks["bounds"], "envelope": sy.check_envelope(es),
                         "endpoint": bool(sy.metric(es[-1], y) <= sy.eps),
                         "refinement": bool(sy.metric(fine[-1], y) <= sy.eps)}
                ends = [self.model_rollout(k, x0, U[None])[0, -1] for k in range(len(self.nets))]
                segs.append({"actions": C.floats(U), "model_states": C.floats(ms), "exact_states": C.floats(es),
                             "model_miss": float(sy.metric(ms[-1], y)), "exact_miss": float(sy.metric(es[-1], y)),
                             "refined_miss": float(sy.metric(fine[-1], y)), "model_checks": checks,
                             "model_valid": model_valid, "exact_checks": exact, "valid": all(exact.values()),
                             "model_error": [float(sy.metric(e, es[-1])) for e in ends]})
                if model_valid:
                    rows.append((M, i))
            rungs[str(M)] = {"segments": segs}
        # Stage 2's exact segments: each M's first valid restart
        s2 = {}
        for M in sy.ladder:
            rung2 = rec2["rungs"][str(M)]["lm_shooting"]
            k = rung2["first_valid"]
            s2[str(M)] = None if k is None else {"restart": k, "actions": rung2["segments"][k]["actions"]}
        # pi's own actions as the recovery: taken from pi's continuation under each dynamics
        pi_prefix = {dyn: self.pi_actions(dyn, x0, max(sy.ladder)) for dyn in ("exact", *CHECK)}

        def prefixes(dyn):
            out = [np.zeros(0)]  # continuation
            out += [np.asarray(rungs[str(M)]["segments"][i]["actions"]) for M, i in rows]
            out += [pi_prefix[dyn][:M] for M in sy.ladder]
            if dyn == "exact":
                out += [np.asarray(s2[str(M)]["actions"])[:, 0] for M in sy.ladder if s2[str(M)]]
                out += [np.zeros(M) for M in sy.ladder]
            return out

        res = {dyn: self.branches(dyn, x0, prefixes(dyn), n) for dyn in ("exact", *CHECK)}
        demo_rel = self.branches("exact", x0, prefixes("exact"), n, policy="demonstrator")
        R, L = len(rows), len(sy.ladder)
        base = {dyn: res[dyn][0]["upright"] for dyn in res}
        rec["continuation"] = {dyn if dyn == "exact" else f"model_{dyn}": res[dyn][0] for dyn in res}
        rec["demonstrator_continuation"] = demo_rel[0]
        for j, (M, i) in enumerate(rows):
            g = rungs[str(M)]["segments"][i]
            g["delta"] = res["exact"][1 + j]["upright"] - base["exact"]
            g["delta_hat"] = [res[k][1 + j]["upright"] - base[k] for k in CHECK]
            g["delta_hat_min"] = min(g["delta_hat"])
            g["hold"] = res["exact"][1 + j]["hold"]
            g["gym_return"] = res["exact"][1 + j]["gym_return"]
            g["demonstrator_delta"] = demo_rel[1 + j]["upright"] - demo_rel[0]["upright"]
        rec["pi_reference"] = {str(M): {"exact": res["exact"][1 + R + m]["upright"] - base["exact"],
                                        "models": [res[k][1 + R + m]["upright"] - base[k] for k in CHECK]}
                               for m, M in enumerate(sy.ladder)}
        off = 1 + R + L
        ceiling = None
        stage2 = {}
        for M in sy.ladder:
            if s2[str(M)] is None:
                stage2[str(M)] = None
                continue
            b = res["exact"][off]
            stage2[str(M)] = {"restart": s2[str(M)]["restart"], "delta": b["upright"] - base["exact"],
                              "hold": b["hold"], "demonstrator_delta": demo_rel[off]["upright"] - demo_rel[0]["upright"]}
            off += 1
            if ceiling is None and stage2[str(M)]["delta"] >= USEFUL:
                ceiling = {"M": M, "restart": s2[str(M)]["restart"], "delta": stage2[str(M)]["delta"]}
        rec["zero_reference"] = {str(M): res["exact"][off + m]["upright"] - base["exact"]
                                 for m, M in enumerate(sy.ladder)}
        rec["stage2"] = stage2
        rec["ceiling"] = ceiling
        rec["selection"] = self.select(rungs)
        rec["rungs"] = rungs
        return rec

    def pi_actions(self, dyn, x0, M: int) -> np.ndarray:
        x, out = np.asarray(x0, np.float64)[None], []
        for _ in range(M):
            u = np.clip(self.pi(x), -self.sy.ubar, self.sy.ubar)
            out.append(u[0])
            x, _ = self.step(dyn, x, u)
        return np.array(out)

    def select(self, rungs: dict):
        """The learned selection: shortest M, then restart order; model-valid with delta_hat_min >= 1. Uses no exact
        outcome."""
        for M in self.sy.ladder:
            for i, g in enumerate(rungs[str(M)]["segments"]):
                if g["model_valid"] and g["delta_hat_min"] >= USEFUL:
                    return {"M": M, "restart": i, "delta_hat_min": g["delta_hat_min"]}
        return None
