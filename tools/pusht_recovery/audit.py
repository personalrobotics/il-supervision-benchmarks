"""#52 Stage A audit: re-derive every accepted recovery window and the decision in a fresh process, read-only.

    LEROBOT_PYTHON tools/pusht_recovery/audit.py --source SOURCE.npz --candidates CANDIDATES.npz \
        --proposals PROPOSALS.npz --run RUN_DIR [--record docs/studies/evidence/pusht-recovery/stage-a.json]

It recomputes the limits (action change, speed envelope) and the real and needed window counts from the inputs. For
every accepted proposal it replays the chain from the stored predecessor and preceding action and requires the stored
positions bit for bit, then recomputes the deviations, the loss mask and every validity check. It recomputes the
selection, and builds each selected window's slots independently of stage_a.windows: the anchor and source frames
from ilsb.temporal's offsets, each action from the chain's own actions and the source rows, each observation from the
predecessor and the anchor frame. It refuses any observation equal to a held-out row, and recomputes the window
digests and the decision. Rejected proposals are not re-solved here; a second clean run reproducing the record byte
for byte covers them. Exit 0 with a JSON summary when everything reproduces; exit 2 listing every problem otherwise.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_a as A  # noqa: E402


def check_proposal(env, src, rows, t, z, a_corr, stored_chain, stored_mask, limits) -> tuple:
    """(problems, chain actions, k_max, mask) for one accepted proposal, recomputed from its stored z."""
    n = len(rows)
    cur = np.concatenate([src["state"][rows][:, 2:4], src["pose"][rows][:, 3:6]], 1).astype(np.float64)
    act = src["action"][rows].astype(np.float64)
    if t < 1 or t > n - A.DROP - 1:
        return ["the anchor is not eligible"], None, None, None
    k_max = min(A.H - 1, n - t)
    acts = [z[10:12], a_corr] + [act[t + k - 1] for k in range(2, k_max + 1)]
    p = A.chain(env, z, acts)
    out = []
    if not np.array_equal(p, stored_chain[:len(p)]):
        return ["the chain does not reproduce bit for bit"], acts, k_max, None
    dev = np.array([A.distance(p[1 + m], cur[t + m])[0] for m in range(min(len(p) - 1, n - t))])
    if dev[0] > A.TOL or dev[1] > A.TOL:
        out.append("the chain misses frame t or t + 1")
    mask = np.zeros(A.H, bool)
    mask[:2] = True
    for k in range(2, k_max + 1):
        if not (dev[1:k] <= A.MASK["continuation_px"]).all():
            break
        mask[k] = True
    if not np.array_equal(mask, stored_mask):
        out.append("the loss mask does not follow the rule")
    if not mask[A.MASK["required_slots"]].all():
        out.append("the executed chunk is not supervised")
    if not np.isfinite(p).all():
        out.append("a chain state is not finite")
    sup = np.array(acts)[mask[:len(acts)]]
    if not ((sup >= A.LO) & (sup <= A.HI)).all():
        out.append("a supervised action is outside the action bounds")
    if np.linalg.norm(z[10:12] - act[t - 1]) > limits["action_change"]:
        out.append("b is beyond the action-change limit")
    if (A.speeds(A.state_of(z)) > limits["speed"]).any():
        out.append("P is beyond the speed envelope")
    pen, ext = A.envelope_bad(p[:int(np.flatnonzero(mask)[-1]) + 1])
    if pen or ext:
        out.append(f"a chain state is beyond the {'penetration' if pen else 'extent'} envelope")
    return out, acts, k_max, mask


def check_window(src, rows, e, t, z, acts, k_max, mask, w, j) -> list:
    """One selected window's slots, built here from the chain's actions, the source rows and ilsb.temporal."""
    out = []
    n = len(rows)
    anchor = t + 1
    off = A.temporal().WindowSpec(A.N_OBS, A.H, A.N_ACT, A.DROP, False).action_offsets
    raw = anchor + off
    frame, pad = np.clip(raw, 0, n), raw > n  # dataset frames 0..n (row r is frame r + 1)
    if int(w["episode"][j]) != e or int(w["anchor_frame"][j]) != anchor:
        out.append("its episode or anchor frame is not the proposal's")
    want_frame = frame.copy()
    want_frame[0], want_frame[1] = -1, -2  # the synthetic history action and the correction
    if not np.array_equal(w["slot_frame"][j], want_frame):
        out.append("a slot's source frame is not the anchor's offset frame")
    if not np.array_equal(w["action_pad"][j][2:], pad[2:]) or w["action_pad"][j][:2].any():
        out.append("a slot's padding flag is wrong")
    want = np.empty((A.H, 2), np.float32)
    want[0], want[1] = z[10:12], acts[1]
    for k in range(2, A.H):
        want[k] = src["action"][rows[frame[k] - 1]]
        if k <= k_max and not np.array_equal(np.float32(acts[k]), want[k]):
            out.append(f"slot {k} is not the action the chain applied")
    if not np.array_equal(w["action"][j], want):
        out.append("a window action is not its slot's action")
    if not np.array_equal(w["loss_mask"][j], mask):
        out.append("the window's loss mask is not the proposal's")
    kp = A.pose.keypoints(np.array([[z[2], z[3], z[4] / A.L]]))[0]
    o_prev = np.concatenate([z[0:2], kp]).astype(np.float32)
    o_t = np.concatenate([src["state"][rows[t], 2:4], src["env"][rows[t], 16:32]]).astype(np.float32)
    if not np.array_equal(w["observation"][j][0], o_prev):
        out.append("the previous observation is not the predecessor's")
    if not np.array_equal(w["observation"][j][1], o_t):
        out.append("the current observation is not the anchor frame's")
    return out


def audit(src: dict, eps: list, props: dict, arrays: dict, record: dict) -> list:
    problems = []
    rows_of = {e: np.flatnonzero(src["episode"] == e) for e in eps}
    rec = {k[len("proposal/"):]: v for k, v in arrays.items() if k.startswith("proposal/")}
    w = {k[len("window/"):]: v for k, v in arrays.items() if k.startswith("window/")}
    if not np.array_equal(rec["candidate_row"], props["candidate_row"]):
        return ["the run's proposals are not the population"]
    limits = A.limits_of(src, eps)
    if (record["limits"]["action_change_px"] != limits["action_change"]
            or record["limits"]["speed"] != limits["speed"].tolist()):
        problems.append("the record's limits are not the subset's")
    n_real, n_syn = A.real_windows(src, eps)
    if (record["mass"]["n_real"], record["mass"]["n_synthetic"]) != (n_real, n_syn):
        problems.append("the record's window counts are not the subset's")
    accepted = np.flatnonzero(rec["status"] == 0)
    env = A.make_env()
    derived = {}
    for i in accepted:
        e, t = int(props["episode"][i]), int(props["t"][i])
        got, acts, k_max, mask = check_proposal(env, src, rows_of[e], t, rec["z"][i], props["action"][i],
                                                rec["chain"][i], rec["mask"][i], limits)
        problems += [f"proposal {int(props['candidate_row'][i])}: {m}" for m in got]
        derived[int(i)] = (acts, k_max, mask)
    src_row = np.array([rows_of[int(e)][int(t)] for e, t in zip(props["episode"], props["t"])], np.int64)
    chosen = A.select(props, accepted, src_row)
    if not np.array_equal(chosen, arrays["selected"]):
        problems.append("the selection does not reproduce")
    if len(w["episode"]) != len(arrays["selected"]):
        return problems + ["the window count is not the selection's"]
    for j, i in enumerate(arrays["selected"]):
        e, t = int(props["episode"][i]), int(props["t"][i])
        acts, k_max, mask = derived.get(int(i), (None, None, None))
        if mask is None:
            problems.append(f"window {j}: its proposal is not an accepted, reproducing one")
            continue
        problems += [f"window {j}: {m}" for m in check_window(src, rows_of[e], e, t, rec["z"][i], acts, k_max, mask,
                                                               w, j)]
    held = np.isin(src["episode"], A.BASE["roles"]["held_out"])
    held_obs = {r.tobytes() for r in np.concatenate([src["state"][held][:, 2:4], src["env"][held][:, 16:32]], 1)
                .astype(np.float32)}
    held_obs |= {r.tobytes() for r in np.concatenate([src["state"][held][:, 0:2], src["env"][held][:, 0:16]], 1)
                 .astype(np.float32)}
    bad = sorted({j for j in range(len(w["episode"])) for o in w["observation"][j] if o.tobytes() in held_obs})
    if bad:
        problems.append(f"windows {bad[:5]}: an observation equals a held-out row")
    if not set(np.unique(w["episode"]).tolist()) <= set(eps):
        problems.append("a window's episode is not a subset episode")
    if record.get("windows_content_sha256") != A.content_sha(w):
        problems.append("the record's window digest is not the run's")
    if record.get("windows_npz_content_sha256") != A.content_sha(arrays):
        problems.append("the record's run digest is not the run's")
    n_eps = len(np.unique(w["episode"]))
    passed = len(arrays["selected"]) >= n_syn and n_eps >= A.SPEC["yield"]["min_episodes"]
    if record.get("decision") is not None and record["decision"].startswith("pass") != passed:
        problems.append("the decision does not follow from the counts")
    return problems


def main():
    p = argparse.ArgumentParser()
    for k in ("source", "candidates", "proposals", "run"):
        p.add_argument(f"--{k}", type=Path, required=True)
    p.add_argument("--record", type=Path, default=A.ROOT / "docs/studies/evidence/pusht-recovery/stage-a.json")
    args = p.parse_args()
    A.simulator_check()
    src, cand, orc = A.load_inputs(args)
    eps = A.subset(A.BASE["roles"]["train"], A.SPEC["cell"]["r"], A.SPEC["cell"]["subset_seed"])
    props = A.population(src, cand, orc, eps)
    if A.real_windows(src, eps) != (A.SPEC["expected"]["n_real"], A.SPEC["expected"]["n_synthetic"]):
        sys.exit("refused: the real window count is not the protocol's")
    with np.load(args.run / "windows.npz") as z:
        arrays = {k: z[k] for k in z.files}
    record = json.loads(args.record.read_text())
    problems = audit(src, eps, props, arrays, record)
    print(json.dumps({"accepted": int((arrays["proposal/status"] == 0).sum()), "windows": int(len(arrays["selected"])),
                      "problems": problems, "pass": not problems}))
    sys.exit(2 if problems else 0)


if __name__ == "__main__":
    main()
