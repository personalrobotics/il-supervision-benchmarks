"""#62 Stage A: the two-member disagreement probe (pusht-uncertainty.md, stage-a.json).

    LEROBOT_PYTHON tools/pusht_uncertainty/probe.py --members CKPT1 CKPT2 --driver D --run RUN_DIR \
        [--starts development|smoke] [--no-probe]
    LEROBOT_PYTHON tools/pusht_uncertainty/probe.py audit --members CKPT1 CKPT2 --run RUN_DIR
    LEROBOT_PYTHON tools/pusht_uncertainty/probe.py report --runs RUN_1 RUN_2 [--out RECORD.json]

The rerun. Member D drives every start: env.reset(seed = start), driven as lerobot-eval drives it (B2's driver), each
16-action chunk generated after seeding torch with SeedSequence([62, D, start, chunk]). Convolutions run in TF32, as
LeRobot trains and evaluates (PyTorch's cuDNN default; B2's driver forces strict FP32); every other setting stays
deterministic. Bitwise reproducibility is required within the pinned environment the record names, not across hardware. An episode runs at least 80
steps, even past success, then ends at success or step 300. Every step's observation, full simulator state, action,
reward and coverage is saved. An episode is scored up to its first success, where lerobot-eval ends it; the steps run
past success only supply decision points.

The probe. At each chunk boundary (steps 0, 8, ...), each member generates one batch of S = 16 chunks from the driver's
two-frame observation, after torch is seeded with SeedSequence([62, 0, D, start, chunk]), so both members denoise the
same noise. LeRobot's generate_actions returns the executed slots (horizon indices 1-8), in the normalized action space;
those are saved, and the probe never acts. U_epi is the mean over samples of the variance across members, and U_ale the
mean over members of the variance across samples, each averaged over the 8 slots and 2 action dimensions.

The audit runs in a fresh process. It regenerates every driver chunk and every probe batch bit for bit from the saved
observations and seeds, recomputes coverage, labels and every statistic from the saved states and outputs, and replays
the saved actions from each start as a measurement. report combines the two drivers' runs into the decision.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "pusht_recovery"))
import stage_a as A  # noqa: E402
import stage_b2 as B  # noqa: E402

SPEC = json.loads((HERE / "stage-a.json").read_text())
B1 = json.loads((ROOT / SPEC["members"]["source"]).read_text())
S_PROBE, CHUNK, MAX_STEPS, MIN_STEPS, WINDOW = 16, 8, 300, 80, 80
AUC_MIN, MARGIN, MIN_CLASS = 0.70, 0.05, 10
STARTS = {"development": (SPEC["rerun"]["starts"]["start_seed"], SPEC["rerun"]["starts"]["n_episodes"]),
          "smoke": (SPEC["smoke"]["starts"]["start_seed"], SPEC["smoke"]["starts"]["n_episodes"])}
NORMALIZERS = ("policy_preprocessor_step_3_normalizer_processor.safetensors",
               "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
TRAJ_KEYS = ("obs_agent", "obs_kp", "full", "pos", "action", "reward", "coverage", "chunk_seeds")
COV_DIGITS = B.COV_DIGITS


def git(*a):
    return subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True, check=True).stdout.strip()


def seed_of(*key) -> int:
    return int(np.random.SeedSequence([62, *key]).generate_state(1, np.uint32)[0])


def log(msg: str):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}", flush=True)


def check_members(paths: list) -> list:
    """The two pinned checkpoints, in member order, with byte-identical normalizer stats."""
    want = [SPEC["members"][str(m)]["model_sha256"] for m in (1, 2)]
    got = [A.sha(p / "model.safetensors") for p in paths]
    if got != want:
        sys.exit(f"refused: the members are not stage-a.json's ({[g[:12] for g in got]})")
    for f in NORMALIZERS:
        if A.sha(paths[0] / f) != A.sha(paths[1] / f):
            sys.exit(f"refused: the members' {f} differ")
    return got


class Policy(B.Policy):
    """B2's driver with LeRobot's default numerical mode: TF32 cuDNN convolutions, strict FP32 matmuls."""

    def __init__(self, ckpt: Path, device: str = "cuda"):
        super().__init__(ckpt, device)
        self.torch.backends.cudnn.allow_tf32 = True


def environment(torch) -> dict:
    """What bitwise reproducibility is pinned to."""
    import importlib.metadata as md
    smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                         text=True)  # host-wide; nvidia-smi's indices ignore CUDA_VISIBLE_DEVICES
    return {"gpu": torch.cuda.get_device_name(), "driver": smi.stdout.split()[0], "cuda": torch.version.cuda,
            "torch": torch.__version__, "cudnn": torch.backends.cudnn.version(), "lerobot": md.version("lerobot"),
            "numpy": np.__version__, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


class Probe:
    """Every member's S chunks at one two-frame observation, from shared noise."""

    def __init__(self, pols: list):
        self.pols = pols

    def __call__(self, prev: tuple, cur: tuple, seed: int) -> np.ndarray:
        from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE
        torch, out = self.pols[0].torch, []
        with torch.inference_mode():
            for pol in self.pols:
                bp, bc = pol.batch(*prev), pol.batch(*cur)
                batch = {k: torch.stack([bp[k], bc[k]], 1).expand(S_PROBE, -1, -1).contiguous()
                         for k in (OBS_STATE, OBS_ENV_STATE)}
                torch.manual_seed(seed)
                out.append(pol.policy.diffusion.generate_actions(batch).float().cpu().numpy())
        return np.stack(out)  # (members, S, 8, 2), normalized


def u_of(p: np.ndarray) -> tuple:
    """(members, S, 8, 2) -> U_epi, U_ale."""
    p = p.astype(np.float64)
    return float(p.var(axis=0).mean()), float(p.var(axis=1).mean())


def rollout(pol, probe, env, driver: int, start: int) -> tuple:
    """One episode; step i's state is the simulator after i actions."""
    env.reset(seed=start)
    pol.policy.reset()
    agent, kp = B.obs_of(env)
    t = {k: [] for k in TRAJ_KEYS}
    t["obs_agent"].append(agent), t["obs_kp"].append(kp)
    t["full"].append(B.flat(A.full_state(env))), t["pos"].append(A.positions(env))
    points, success_step = [], -1
    for i in range(MAX_STEPS):
        s = None
        if pol.needs_chunk():
            c = len(t["chunk_seeds"])
            s = seed_of(driver, start, c)
            t["chunk_seeds"].append((i, s))
            if probe is not None:
                prev = (t["obs_agent"][max(i - 1, 0)], t["obs_kp"][max(i - 1, 0)])
                ps = seed_of(0, driver, start, c)
                points.append((i, ps, probe(prev, (agent, kp), ps)))
        a = pol.act(agent, kp, s)
        _, reward, _, _, info = env.step(a)
        agent, kp = B.obs_of(env)
        t["action"].append(a), t["reward"].append(round(float(reward), COV_DIGITS))  # coverage / 0.95: same drift
        t["coverage"].append(round(float(info["coverage"]), COV_DIGITS))
        t["obs_agent"].append(agent), t["obs_kp"].append(kp)
        t["full"].append(B.flat(A.full_state(env))), t["pos"].append(A.positions(env))
        if info["is_success"] and success_step < 0:
            success_step = i + 1
        if success_step > 0 and i + 1 >= MIN_STEPS:
            break
    t = {k: np.array(v, np.int64 if k == "chunk_seeds" else np.float64) for k, v in t.items()}
    t["chunk_seeds"] = t["chunk_seeds"].reshape(-1, 2)
    return t, points


def pack(trajs: list) -> dict:
    out = {}
    for key in TRAJ_KEYS:
        parts = [t[key] for t in trajs]
        out[key] = np.concatenate(parts)
        out[f"{key}_offsets"] = np.cumsum([0] + [len(p) for p in parts]).astype(np.int64)
    return out


def unpack(arrs: dict, i: int) -> dict:
    return {k: arrs[k][arrs[f"{k}_offsets"][i]:arrs[f"{k}_offsets"][i + 1]] for k in TRAJ_KEYS}


def end_of(coverage: np.ndarray) -> tuple:
    """(success, scored steps): the first success, as lerobot-eval ends the episode, or all steps."""
    hit = np.flatnonzero(coverage > 0.95)
    return (True, int(hit[0]) + 1) if len(hit) else (False, len(coverage))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank AUC of pos over neg, ties counted half."""
    if not len(pos) or not len(neg):
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def q(x) -> dict:
    x = np.asarray(x, np.float64)
    return {str(p): float(v) for p, v in zip((10, 50, 90), np.percentile(x, (10, 50, 90)))} if len(x) else {}


def statistics(arrs: dict, probes: dict, starts: list) -> dict:
    """Labels, the early-window episode scores, the decision for this driver and the descriptive summaries."""
    n = len(starts)
    episodes, early, by_chunk, before = [], [], {}, {}
    for i, start in enumerate(starts):
        t = unpack(arrs, i)
        success, end = end_of(t["coverage"])
        sel = probes["episode"] == i
        steps, ue, ua = probes["step"][sel], probes["u_epi"][sel], probes["u_ale"][sel]
        w = steps < WINDOW
        if w.sum() != WINDOW // CHUNK:
            raise SystemExit(f"start {start}: {int(w.sum())} early-window points, not {WINDOW // CHUNK}")
        scored = steps < end
        episodes.append({"start": int(start), "success": success, "scored_steps": end, "steps": len(t["action"]),
                         "max_reward": float(t["reward"][:end].max()),
                         "score_epi": float(ue[w].mean()), "score_ale": float(ua[w].mean()),
                         "whole_epi": float(ue[scored].mean())})
        early += [(success, float(x)) for x in ue[w]]
        for c, (s, x) in enumerate(zip(steps[scored], ue[scored])):
            by_chunk.setdefault((success, c), []).append(float(x))
            if success:
                before.setdefault(int((end - s - 1) // CHUNK), []).append(float(x))
    fail = np.array([not e["success"] for e in episodes])
    col = lambda k: np.array([e[k] for e in episodes])  # noqa: E731
    a_epi, a_ale = auc(col("score_epi")[fail], col("score_epi")[~fail]), auc(col("score_ale")[fail], col("score_ale")[~fail])
    informative = bool(fail.sum() >= MIN_CLASS and (~fail).sum() >= MIN_CLASS)
    ef = np.array([not s for s, _ in early])
    ev = np.array([x for _, x in early])
    return {
        "episodes": episodes,
        "n_failed": int(fail.sum()), "n_successful": int((~fail).sum()),
        "avg_max_reward": float(col("max_reward").mean()), "pc_success": 100.0 * float((~fail).sum()) / n,
        "auc_epi": a_epi, "auc_ale": a_ale, "auc_margin": a_epi - a_ale,
        "informative": informative,
        "pass": bool(informative and a_epi >= AUC_MIN and a_epi - a_ale >= MARGIN),
        "descriptive": {
            "u_epi_by_outcome": {"failed": q(col("score_epi")[fail]), "successful": q(col("score_epi")[~fail])},
            "u_ale_by_outcome": {"failed": q(col("score_ale")[fail]), "successful": q(col("score_ale")[~fail])},
            "u_epi_over_u_ale_early": q(probes["u_epi"][probes["step"] < WINDOW] / probes["u_ale"][probes["step"] < WINDOW]),
            "state_auc_early": auc(ev[ef], ev[~ef]),
            "whole_episode_auc": auc(col("whole_epi")[fail], col("whole_epi")[~fail]),
            "u_epi_by_chunk": {f"{'successful' if s else 'failed'}/{c}": float(np.mean(v))
                               for (s, c), v in sorted(by_chunk.items())},
            "u_epi_by_chunks_before_success": {str(k): float(np.mean(v)) for k, v in sorted(before.items())},
        },
    }


def starts_of(which: str) -> list:
    s0, n = STARTS[which]
    return list(range(s0, s0 + n))


def discover(args) -> dict:
    members = check_members(args.members)
    pols = [Policy(p) for p in args.members]
    pol, probe, env = pols[args.driver - 1], (None if args.no_probe else Probe(pols)), A.make_env()
    starts = starts_of(args.starts)
    trajs, pts, t0 = [], [], time.time()
    log(f"driver {args.driver}: {len(starts)} starts from {starts[0]}, probe {'off' if args.no_probe else 'on'}")
    for i, start in enumerate(starts):
        t, p = rollout(pol, probe, env, args.driver, start)
        trajs.append(t)
        pts += [(i, *x) for x in p]
        if (i + 1) % 5 == 0 or i + 1 == len(starts):
            log(f"driver {args.driver}: {i + 1}/{len(starts)} episodes, {time.time() - t0:.0f} s")
    arrs = {**pack(trajs), "start": np.array(starts, np.int64)}
    args.run.mkdir(parents=True, exist_ok=False)
    np.savez(args.run / "trajectories.npz", **arrs)
    rec = {"schema": "ilsb.pusht_uncertainty.stage_a_run 1",
           "code": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))},
           "driver": args.driver, "starts": args.starts, "probe": not args.no_probe,
           "members_model_sha256": members, "environment": environment(pol.torch),
           "trajectories_content_sha256": A.content_sha(arrs)}
    if not args.no_probe:
        probes = {"episode": np.array([x[0] for x in pts], np.int64), "step": np.array([x[1] for x in pts], np.int64),
                  "seed": np.array([x[2] for x in pts], np.int64), "output": np.stack([x[3] for x in pts])}
        np.savez(args.run / "probes.npz", **probes)
        rec["probes_content_sha256"] = A.content_sha(probes)
        rec.update(statistics(arrs, {**probes, **u_arrays(probes["output"])}, starts))
    return rec


def u_arrays(output: np.ndarray) -> dict:
    u = np.array([u_of(p) for p in output]).reshape(-1, 2)
    return {"u_epi": u[:, 0], "u_ale": u[:, 1]}


def load_run(run: Path) -> tuple:
    rec = json.loads((run / "record.json").read_text())
    with np.load(run / "trajectories.npz") as z:
        arrs = {k: z[k] for k in z.files}
    with np.load(run / "probes.npz") as z:
        probes = {k: z[k] for k in z.files}
    return rec, arrs, probes


def audit(args, run=None, only=None, pols=None) -> dict:
    """Fresh process: regenerate the chunks and probes, recompute labels and statistics, measure a simulator replay.
    run: (record, trajectories, probes) in place of RUN_DIR's; only: audit these episodes alone (the smoke's faults)."""
    rec, arrs, probes = run if run is not None else load_run(args.run)
    problems = []
    if A.content_sha(arrs) != rec["trajectories_content_sha256"]:
        problems.append("the trajectories are not the record's")
    if A.content_sha(probes) != rec["probes_content_sha256"]:
        problems.append("the probes are not the record's")
    if rec["members_model_sha256"] != check_members(args.members):
        problems.append("the members are not the record's")
    pols = pols or [Policy(p) for p in args.members]
    if rec["environment"] != environment(pols[0].torch):
        problems.append("the environment is not the record's: bitwise reproducibility holds only within it")
    pol, probe, env = pols[rec["driver"] - 1], Probe(pols), A.make_env()
    starts = [int(s) for s in arrs["start"]]
    worst_cov, worst_replay, chunk_bad, probe_bad, point_bad = 0.0, 0.0, 0, 0, 0
    t0 = time.time()
    for i, start in enumerate(starts):
        if only is not None and i not in only:
            continue
        t = unpack(arrs, i)
        cov = []
        for z in t["full"][1:]:
            A.load_fresh(env, B.unflat(z))
            cov.append(env._get_coverage())
        cov = np.array(cov)
        worst_cov = max(worst_cov, float(np.abs(cov - t["coverage"]).max()))
        if end_of(np.round(cov, COV_DIGITS)) != end_of(t["coverage"]):
            problems.append(f"start {start}: the label does not reproduce from the saved states")
        # decision points: the chunk boundaries, and nothing else
        steps = t["chunk_seeds"][:, 0]
        sel = np.flatnonzero(probes["episode"] == i)
        if not (np.array_equal(steps, np.arange(len(steps)) * CHUNK) and np.array_equal(probes["step"][sel], steps)):
            point_bad += 1
        for j, (step, s) in enumerate(t["chunk_seeds"]):
            step, s = int(step), int(s)
            prev = (t["obs_agent"][max(step - 1, 0)], t["obs_kp"][max(step - 1, 0)])
            pol.start(*prev)
            got = [pol.act(t["obs_agent"][step], t["obs_kp"][step], s)]
            for m in range(1, CHUNK):
                if step + m >= len(t["action"]):
                    break
                got.append(pol.act(t["obs_agent"][step + m], t["obs_kp"][step + m], None))
            if not np.array_equal(np.array(got), t["action"][step:step + len(got)]):
                chunk_bad += 1
            if j < len(sel):
                k = sel[j]
                ps = seed_of(0, rec["driver"], start, j)
                if int(probes["seed"][k]) != ps or not np.array_equal(
                        probe(prev, (t["obs_agent"][step], t["obs_kp"][step]), ps), probes["output"][k]):
                    probe_bad += 1
        env.reset(seed=start)
        rep = [A.positions(env)]
        for a in t["action"]:
            env.step(a)
            rep.append(A.positions(env))
        worst_replay = max(worst_replay, float(A.distance(np.array(rep), t["pos"]).max()))
        if only is None and ((i + 1) % 5 == 0 or i + 1 == len(starts)):
            log(f"audit driver {rec['driver']}: {i + 1}/{len(starts)} episodes, {time.time() - t0:.0f} s")
    if worst_cov > B.COV_TOL:
        problems.append(f"coverage recomputed from the saved states differs by {worst_cov:.3g}")
    if point_bad:
        problems.append(f"{point_bad} episodes' decision points are not the chunk boundaries")
    if chunk_bad:
        problems.append(f"{chunk_bad} chunks do not regenerate bit for bit")
    if probe_bad:
        problems.append(f"{probe_bad} probe batches do not regenerate bit for bit")
    if only is None:
        stats = statistics(arrs, {**probes, **u_arrays(probes["output"])}, starts)
        if json.dumps(stats, sort_keys=True) != json.dumps({k: rec.get(k) for k in stats}, sort_keys=True):
            problems.append("the statistics do not reproduce from the saved outputs")
    elif any(rec["episodes"][i]["success"] != end_of(unpack(arrs, i)["coverage"])[0] for i in only):
        problems.append(f"start {starts[min(only)]}: the label does not reproduce from the saved states")
    return {"problems": problems, "pass": not problems, "coverage_max_abs_diff": worst_cov,
            "chunks_mismatched": chunk_bad, "probes_mismatched": probe_bad, "simulator_replay_max_px": worst_replay}


def report(args) -> dict:
    """Both drivers' runs -> the Stage A record and decision."""
    recs = [json.loads((r / "record.json").read_text()) for r in args.runs]
    if [r["driver"] for r in recs] != [1, 2] or any(r["starts"] != "development" or not r["probe"] for r in recs):
        sys.exit("refused: report takes driver 1's and driver 2's development runs, with the probe")
    if len({r["code"]["commit"] for r in recs}) != 1 or any(r["code"]["dirty"] for r in recs):
        sys.exit("refused: the runs are not from one clean commit")
    b1 = {d: B1["units"][f"lerobot_diffusion/s{d}"] for d in (1, 2)}
    drivers = {}
    for r in recs:
        d, u = r["driver"], b1[r["driver"]]
        failed = {e["start"] for e in r["episodes"] if not e["success"]}
        drivers[str(d)] = {
            **{k: r[k] for k in ("n_failed", "n_successful", "auc_epi", "auc_ale", "auc_margin", "informative", "pass",
                                 "descriptive", "trajectories_content_sha256", "probes_content_sha256")},
            "reconciliation": {"rerun": {"avg_max_reward": r["avg_max_reward"], "pc_success": r["pc_success"]},
                               "b1": u["evaluation"],
                               "episodes_with_other_success": len(failed ^ set(u["unsuccessful_start_seeds"]))},
            "run_files_sha256": {f: A.sha(run / f) for run in [args.runs[d - 1]]
                                 for f in ("record.json", "trajectories.npz", "probes.npz")},
        }
    ok = all(v["pass"] for v in drivers.values())
    uninformative = [d for d, v in drivers.items() if not v["informative"]]
    decision = ("refuse: fewer than 10 failed or successful episodes on driver " + ", ".join(uninformative)
                if uninformative else "pass: Stage B may be proposed" if ok else "stop: the frozen separation rule fails")
    return {"schema": "ilsb.pusht_uncertainty.stage_a 1", "code": recs[0]["code"],
            "members_model_sha256": recs[0]["members_model_sha256"],
            "rule": {"auc_min": AUC_MIN, "margin_over_u_ale": MARGIN, "min_per_class": MIN_CLASS,
                     "window_steps": WINDOW, "samples": S_PROBE},
            "drivers": drivers, "pass": ok, "decision": decision}


def main():
    argv = sys.argv[1:]
    mode = argv[0] if argv[:1] and argv[0] in ("audit", "report") else "run"
    argv = argv[1:] if mode != "run" else argv
    p = argparse.ArgumentParser()
    if mode == "report":
        p.add_argument("--runs", type=Path, nargs=2, required=True)
        p.add_argument("--out", type=Path, default=ROOT / "docs/studies/evidence/pusht-uncertainty/stage-a.json")
        args = p.parse_args(argv)
        out = report(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"pass": out["pass"], "decision": out["decision"]}))
        return
    p.add_argument("--members", type=Path, nargs=2, required=True)
    p.add_argument("--run", type=Path, required=True)
    if mode == "run":
        p.add_argument("--driver", type=int, choices=(1, 2), required=True)
        p.add_argument("--starts", choices=tuple(STARTS), default="development")
        p.add_argument("--no-probe", action="store_true")
    args = p.parse_args(argv)
    A.simulator_check()
    if mode == "audit":
        out = audit(args)
        print(json.dumps(out))
        sys.exit(0 if out["pass"] else 2)
    if args.starts == "development" and git("status", "--porcelain"):
        sys.exit("refused: the tree is dirty; Stage A runs from a clean commit")
    if args.run.exists():
        sys.exit(f"refused: {args.run} exists; a run is never overwritten")
    rec = discover(args)
    (args.run / "record.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    log(json.dumps({k: rec.get(k) for k in ("driver", "starts", "n_failed", "n_successful", "auc_epi", "auc_ale",
                                            "pass")}))


if __name__ == "__main__":
    main()
