"""#52 Stage B2: discover the base policy's deviations from paired demonstration prefixes (pusht-recovery.md, stage-b.json).

    LEROBOT_PYTHON tools/pusht_recovery/stage_b2.py --source SOURCE.npz --checkpoint CKPT --seed S --run RUN_DIR \
        [--episodes all|smoke] [--out RECORD.json]
    LEROBOT_PYTHON tools/pusht_recovery/stage_b2.py audit --source SOURCE.npz --checkpoint CKPT --seed S --run RUN_DIR

For every anchor (support.json's rule; its identities are checked against stage-b.json), the anchor row's replayed full
state is loaded fresh and the frozen base is rolled out from it. The policy's observation queue holds the row's two
recorded frames, so its first chunk sees a real history. Its observations pass through lerobot-eval's own path
(preprocess_observation, the checkpoint's processors). Each 16-action chunk is generated after seeding torch with
SeedSequence([52, seed, episode, anchor row, chunk]), so every chunk can be regenerated on its own.

Per rollout it records c_0, c_D and c_pi (gym_pusht coverage) and the first k >= 1 at which the rollout's state is
farther than tau from every state of the same episode's continuation. A rollout yields a state only when it misses the
demonstrated progress (c_pi < c_D), deviates, and the matched row j* has j* + 8 inside the episode. RUN_DIR holds
trajectories.npz (every rollout, padded, with lengths) and selection.npz (the frozen states with their full simulator
states); the record holds counts, digests and the per-rollout table.

The audit runs in a fresh process: it regenerates every chunk from the saved observations and seeds and requires the
executed actions bit for bit; it recomputes coverage (within 1e-9), D(k), missed progress and the selection from the
saved states and requires the selection to be identical; and it replays the saved actions from each anchor in the
simulator and reports the largest position difference as a measurement, not a gate.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import stage_a as A  # noqa: E402
import support as S  # noqa: E402

SPEC = json.loads((HERE / "stage-b.json").read_text())
D_ = SPEC["discovery"]
TAU = D_["tau_px"]
CHUNK, SLACK, REJOIN = 8, 50, 8
COV_TOL = 1e-9


def git(*a):
    return subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True, check=True).stdout.strip()


def flat(s: tuple) -> np.ndarray:
    """A full simulator state (oracle_probe.full_state) as 10 floats, exactly."""
    return np.array([*s[0], *s[1], *s[2], s[3], *s[4], s[5]], np.float64)


def unflat(z: np.ndarray) -> tuple:
    return ((z[0], z[1]), (z[2], z[3]), (z[4], z[5]), float(z[6]), (z[7], z[8]), float(z[9]))


def seed_of(*key) -> int:
    return int(np.random.SeedSequence([52, *key]).generate_state(1, np.uint32)[0])


def anchors(src: dict, eps: list) -> tuple:
    """support.py's anchors, recomputed, with each episode's rows, replayed full states and recorded coverage."""
    cur = np.concatenate([src["state"][:, 2:4], src["pose"][:, 3:6]], 1).astype(np.float64)
    act = src["action"].astype(np.float64)
    out, data = [], {}
    for e in eps:
        rows = np.flatnonzero(src["episode"] == e)
        _, err, states = A.replay(A.make_env(), cur[rows], act[rows])
        cov = S.coverages(cur[rows])
        data[e] = {"rows": rows, "states": states, "cov": cov,
                   "pts": S.points(src["state"][rows], src["env"][rows])}
        n = len(rows)
        for k in range(S.ANCHOR["first_row"], n - S.ANCHOR["rows_before_end"] + 1, S.ANCHOR["stride"]):
            if err[k] <= A.REPLAY_TOL and cov[k:].max() > cov[k]:
                out.append((int(e), int(k)))
    return out, data


class Policy:
    """The checkpoint driven as lerobot-eval drives it, on one environment."""

    def __init__(self, ckpt: Path, device: str = "cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.utils import preprocess_observation
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.utils.import_utils import register_third_party_plugins
        register_third_party_plugins()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        self.torch, self.prep = torch, preprocess_observation
        cfg = PreTrainedConfig.from_pretrained(ckpt)
        self.policy = get_policy_class(cfg.type).from_pretrained(ckpt).to(device).eval()
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=str(ckpt),
                                                       preprocessor_overrides={"device_processor": {"device": device}})

    def batch(self, agent: np.ndarray, kp: np.ndarray) -> dict:
        obs = self.prep({"agent_pos": agent[None].astype(np.float64), "environment_state": kp[None].astype(np.float64)})
        obs["task"] = [""]
        b = self.pre(obs)
        b.pop("action", None)  # the processors add an empty action, which populate_queues would queue
        return b

    def start(self, agent_prev, kp_prev):
        """A fresh queue holding the previous frame (populate_queues fills both slots with it)."""
        from lerobot.policies.utils import populate_queues
        self.policy.reset()
        self.policy._queues = populate_queues(self.policy._queues, self.batch(agent_prev, kp_prev))

    def act(self, agent, kp, seed: int | None) -> np.ndarray:
        """One step's action; seed is given when the action queue is empty (a new chunk)."""
        from lerobot.utils.constants import ACTION
        with self.torch.inference_mode():
            b = self.batch(agent, kp)
            if len(self.policy._queues[ACTION]) == 0:
                self.torch.manual_seed(seed)
            a = self.policy.select_action(b)
            return self.post(a).cpu().numpy()[0].astype(np.float64)

    def needs_chunk(self) -> bool:
        from lerobot.utils.constants import ACTION
        return len(self.policy._queues[ACTION]) == 0


def obs_of(env) -> tuple:
    o = env.get_obs()
    return np.asarray(o["agent_pos"], np.float64), np.asarray(o["environment_state"], np.float64)


def rollout(pol: Policy, env, seed: int, e: int, k0: int, d: dict, row_obs: tuple) -> dict:
    """One rollout from anchor row k0; step i's state is the simulator after i actions (step 0 is the restored anchor)."""
    rows, n = d["rows"], len(d["rows"])
    A.load_fresh(env, d["states"][k0])
    steps = (n - 1 - k0) + SLACK
    agent_prev, kp_prev, agent0, kp0 = row_obs
    pol.start(agent_prev, kp_prev)
    agent, kp = agent0, kp0  # the policy's first current observation is the recorded frame
    obs_agent, obs_kp = [agent0], [kp0]
    sim_agent, sim_kp = obs_of(env)
    pos, full, acts, cov, chunk_seeds = [A.positions(env)], [flat(A.full_state(env))], [], [], []
    for i in range(steps):
        s = None
        if pol.needs_chunk():
            s = seed_of(seed, e, k0, len(chunk_seeds))
            chunk_seeds.append((i, s))
        a = pol.act(agent, kp, s)
        _, _, _, _, info = env.step(a)
        acts.append(a)
        cov.append(float(info["coverage"]))
        pos.append(A.positions(env))
        full.append(flat(A.full_state(env)))
        agent, kp = obs_of(env)
        obs_agent.append(agent)
        obs_kp.append(kp)
        if info["is_success"]:
            break
    return {"pos": np.array(pos), "full": np.array(full), "action": np.array(acts), "coverage": np.array(cov),
            "obs_agent": np.array(obs_agent), "obs_kp": np.array(obs_kp),
            "chunk_seeds": np.array(chunk_seeds, np.int64).reshape(-1, 2), "sim_agent0": sim_agent, "sim_kp0": sim_kp}


def select(traj: dict, d: dict, k0: int, coverage: np.ndarray) -> dict:
    """c_0, c_D, c_pi, and the first deviation, from a rollout's saved observations and coverage."""
    n = len(d["rows"])
    c0, cD = float(d["cov"][k0]), float(d["cov"][k0:].max())
    cpi = float(coverage.max()) if len(coverage) else c0
    cont = d["pts"][k0:]
    out = {"c_0": c0, "c_D": cD, "c_pi": cpi, "shortfall": cD - cpi, "missed_progress": cpi < cD,
           "k": -1, "j_star": -1, "d_px": float("nan"), "selected": False, "reason": "no_missed_progress"}
    if cpi >= cD:
        return out
    ag, kp = traj["obs_agent"], traj["obs_kp"]
    for k in range(1, len(ag)):
        x = np.concatenate([np.stack([ag[k - 1], ag[k]]), np.concatenate([kp[k - 1], kp[k]]).reshape(16, 2)])
        dist = S.dist(x[None], cont)
        j = int(dist.argmin())
        if dist[j] > TAU:
            out.update(k=k, j_star=k0 + j, d_px=float(dist[j]))
            if k0 + j + REJOIN <= n - 1:
                out.update(selected=True, reason="selected")
            else:
                out["reason"] = "rejoin_beyond_episode"
            return out
    out["reason"] = "no_deviation"
    return out


def row_obs(src: dict, row: int) -> tuple:
    st, en = src["state"][row].astype(np.float64), src["env"][row].astype(np.float64)
    return st[0:2], en[0:16], st[2:4], en[16:32]


def pack(trajs: list) -> dict:
    """Ragged per-rollout arrays as one flat array each, with offsets."""
    out = {}
    for key in ("pos", "full", "action", "coverage", "obs_agent", "obs_kp", "chunk_seeds"):
        parts = [t[key] for t in trajs]
        out[key] = np.concatenate(parts) if parts else np.zeros(0)
        out[f"{key}_offsets"] = np.cumsum([0] + [len(p) for p in parts]).astype(np.int64)
    return out


def unpack(arrs: dict, i: int) -> dict:
    return {key: arrs[key][arrs[f"{key}_offsets"][i]:arrs[f"{key}_offsets"][i + 1]]
            for key in ("pos", "full", "action", "coverage", "obs_agent", "obs_kp", "chunk_seeds")}


def episodes_of(which: str) -> list:
    sub = A.subset(A.BASE["roles"]["train"], S.R, S.SUBSET_SEED)
    if which == "all":
        return sub
    # smoke: the first two training episodes outside the subset; nothing of the study's population is read
    return sorted(e for e in A.BASE["roles"]["train"] if e not in set(sub))[:2]


def load_source(path: Path) -> dict:
    with np.load(path) as z:
        src = {k: z[k] for k in z.files}
    if A.source_module.content_sha256(src) != S.SOURCE_CONTENT:
        sys.exit("refused: the source rows are not the protocol's by content")
    return src


def discover(args) -> dict:
    src = load_source(args.source)
    eps = episodes_of(args.episodes)
    anc, data = anchors(src, eps)
    anc_arr = np.array(anc, np.int64).reshape(-1, 2)
    if args.episodes == "all":
        a = D_["anchors"]
        if (len(anc), hashlib.sha256(anc_arr.tobytes()).hexdigest()) != (a["count"], a["sha256"]):
            sys.exit("refused: the anchors are not stage-b.json's")
    pol, env = Policy(args.checkpoint), A.make_env()
    trajs, table = [], []
    for e, k0 in anc:
        t = rollout(pol, env, args.seed, e, k0, data[e], row_obs(src, int(data[e]["rows"][k0])))
        trajs.append(t)
        table.append({"episode": e, "anchor_row": k0, "steps": len(t["action"]), **select(t, data[e], k0, t["coverage"])})
    arrs = {**pack(trajs), "anchor": anc_arr}
    sel = [i for i, r in enumerate(table) if r["selected"]]
    selection = {"rollout": np.array(sel, np.int64),
                 "episode": np.array([table[i]["episode"] for i in sel], np.int64),
                 "anchor_row": np.array([table[i]["anchor_row"] for i in sel], np.int64),
                 "k": np.array([table[i]["k"] for i in sel], np.int64),
                 "j_star": np.array([table[i]["j_star"] for i in sel], np.int64),
                 "full_k": np.array([trajs[i]["full"][table[i]["k"]] for i in sel]).reshape(-1, 10),
                 "full_k_minus_1": np.array([trajs[i]["full"][table[i]["k"] - 1] for i in sel]).reshape(-1, 10)}
    args.run.mkdir(parents=True, exist_ok=False)
    np.savez(args.run / "trajectories.npz", **arrs)
    np.savez(args.run / "selection.npz", **selection)
    reasons = {}
    for r in table:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    return {"schema": "ilsb.pusht_recovery.stage_b2 1",
            "code": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))},
            "seed": args.seed, "episodes": args.episodes, "checkpoint_model_sha256": A.sha(args.checkpoint / "model.safetensors"),
            "tau_px": TAU, "rollouts": len(table), "by_reason": reasons,
            "selected": len(sel), "selected_episodes": int(len(np.unique(selection["episode"]))),
            "trajectories_content_sha256": A.content_sha(arrs), "selection_content_sha256": A.content_sha(selection),
            "table": table}


def load_run(run: Path) -> tuple:
    record = json.loads((run / "record.json").read_text())
    with np.load(run / "trajectories.npz") as z:
        arrs = {k: z[k] for k in z.files}
    with np.load(run / "selection.npz") as z:
        selection = {k: z[k] for k in z.files}
    return record, arrs, selection


def audit(args, run=None, only=None, pol=None) -> dict:
    """Fresh process: regenerate the chunks, recompute the selection, measure a simulator replay. run: (record,
    trajectories, selection) in place of RUN_DIR's; only: audit these rollouts alone (the smoke's faults)."""
    src = load_source(args.source)
    eps = episodes_of(args.episodes)
    _, data = anchors(src, eps)
    record, arrs, selection = run if run is not None else load_run(args.run)
    problems = []
    if A.content_sha(arrs) != record["trajectories_content_sha256"]:
        problems.append("the trajectories are not the record's")
    if A.content_sha(selection) != record["selection_content_sha256"]:
        problems.append("the selection is not the record's")
    pol, env = pol or Policy(args.checkpoint), A.make_env()
    worst_cov, worst_replay, chunk_mismatch, sel_rows = 0.0, 0.0, 0, []
    for i, (e, k0) in enumerate(arrs["anchor"]):
        if only is not None and i not in only:
            continue
        e, k0 = int(e), int(k0)
        t, d = unpack(arrs, i), data[e]
        # coverage from the saved full states
        cov = []
        for z in t["full"][1:]:
            A.load_fresh(env, unflat(z))
            cov.append(env._get_coverage())
        cov = np.array(cov)
        worst_cov = max(worst_cov, float(np.abs(cov - t["coverage"]).max()) if len(cov) else 0.0)
        # every chunk regenerated from the saved observations and seed
        ap, kp_prev, _, _ = row_obs(src, int(d["rows"][k0]))
        for step, s in t["chunk_seeds"]:
            step = int(step)
            prev = (ap, kp_prev) if step == 0 else (t["obs_agent"][step - 1], t["obs_kp"][step - 1])
            pol.start(*prev)
            got = [pol.act(t["obs_agent"][step], t["obs_kp"][step], int(s))]
            for m in range(1, CHUNK):
                if step + m >= len(t["action"]):
                    break
                got.append(pol.act(t["obs_agent"][step + m], t["obs_kp"][step + m], None))
            if not np.array_equal(np.array(got), t["action"][step:step + len(got)]):
                chunk_mismatch += 1
        r = select(t, d, k0, cov)
        if r["selected"]:
            sel_rows.append((i, r["k"], r["j_star"]))
        # an independent replay of the saved actions: a measurement
        A.load_fresh(env, d["states"][k0])
        rep = [A.positions(env)]
        for a in t["action"]:
            env.step(a)
            rep.append(A.positions(env))
        worst_replay = max(worst_replay, float(A.distance(np.array(rep), t["pos"]).max()))
    if worst_cov > COV_TOL:
        problems.append(f"coverage recomputed from the saved states differs by {worst_cov:.3g}")
    if chunk_mismatch:
        problems.append(f"{chunk_mismatch} chunks do not regenerate bit for bit")
    want = np.array(sel_rows, np.int64).reshape(-1, 3)
    have = np.stack([selection["rollout"], selection["k"], selection["j_star"]], 1) if len(selection["rollout"]) \
        else np.zeros((0, 3), np.int64)
    if only is not None:
        have = have[np.isin(have[:, 0], list(only))]
    if not np.array_equal(want, have):
        problems.append("the selection does not reproduce from the saved states")
    return {"problems": problems, "pass": not problems, "coverage_max_abs_diff": worst_cov,
            "chunks_mismatched": chunk_mismatch, "simulator_replay_max_px": worst_replay}


def main():
    argv = sys.argv[1:]
    mode = "audit" if argv[:1] == ["audit"] else "discover"
    if mode == "audit":
        argv = argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--episodes", choices=("all", "smoke"), default="all")
    args = p.parse_args(argv)
    A.simulator_check()
    if mode == "audit":
        out = audit(args)
        print(json.dumps(out))
        sys.exit(0 if out["pass"] else 2)
    if args.episodes == "all" and git("status", "--porcelain"):
        sys.exit("refused: the tree is dirty; B2 runs from a clean commit")
    if args.run.exists():
        sys.exit(f"refused: {args.run} exists; a run is never overwritten")
    rec = discover(args)
    (args.run / "record.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: rec[k] for k in ("seed", "rollouts", "by_reason", "selected", "selected_episodes")}))


if __name__ == "__main__":
    main()
