"""#55 Stage A: can the existing r = 0.25 CCIL pool define a correction-specific gate? Read-only (pusht-refiner.md).

    LEROBOT_PYTHON tools/pusht_refiner/stage_a.py --source SOURCE.npz --labels LABELS.npz \
        --candidates CANDIDATES.npz --selection SELECTION.npz [--out docs/studies/evidence/pusht-refiner/stage-a.json]

The inputs are the PushT source rows and the surface study's r = 0.25 pool (store entry c30c0f98...): its labels,
candidates and selection, each checked by hash. No policy is loaded or trained, no simulator runs, and no held-out or
evaluation data is read. A state is its 18 points (agent and 8 T keypoints, previous and current frame); the distance
between two states is the largest Euclidean distance between corresponding points, in px.

The record holds the evidence fields the pool carries; the geometry of the labels against their source rows and the
subset's demonstration rows; the relative gate g = d_real^2 / (d_real^2 + d_CCIL^2 + eps) in sample and with each of
the 41 subset episodes held out against the other 40 (quantiles and the rank AUC of labels against real rows); and
the gate on the duplication targets. Computation is float64 on the CPU, so a rerun on the same host is byte-identical.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "pusht_v2"))
import source as source_module  # noqa: E402
from oracle_probe import subset  # noqa: E402

BASE = json.loads((ROOT / "tools/pusht_v2/surface.json").read_text())
INPUTS = {
    "source_content_sha256": "333a9b17ebfc6657591eb2b14a7f314200dd2ae1c743691802a319c1746e561d",
    "labels_sha256": "6024b739dff0551221d13c670a2153087006a0c83ee664221e713e1cc55cbf25",
    "candidates_sha256": "6f8c9c97f85d08e775430f67284b964ebe3b68572c06ae8db8dd8f915202747f",
    "selection_sha256": "ed5beb46e6aa29326b27faccd44683989a314f3ad1b936fa48ba8fd50db3e1d0",
}
# the frozen bases #55 names (the surface study's r = 0.25 BC units); identities only, never loaded here
BASES = {"seed_1": {"unit": "bc/r0.25/a0/s1", "entry": "78a06a4cc266edd75f3a37881c29fe062a9efdd7b0ddfb45dfac694d2189a4dc",
                    "model_sha256": "1c44a71ac60ff26a5492a6f2020fb782fc5cc9d39b79c4d8b3ac24d9181cda4e"},
         "seed_2": {"unit": "bc/r0.25/a0/s2", "entry": "77b8234f73055190efd16391d7051b8bcbb178efaf152815b110c53e1a0960eb",
                    "model_sha256": "8e098a53fc319f4d523f08ddac0e27bf2de36fc2f9d7b0eedb53a06591133f85"}}
R, SUBSET_SEED = 0.25, 0
EXPECTED = {"subset_episodes": 41, "subset_rows": 5040, "candidates": 19996, "labels": 19982}
EPS = 1e-6  # px^2, the gate's denominator floor
QS = (10, 50, 90, 99, 100)
PREV = [0] + list(range(2, 10))  # agent previous, keypoints previous
CUR = [1] + list(range(10, 18))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True, check=True).stdout.strip()


def points(state: np.ndarray, env: np.ndarray) -> np.ndarray:
    """(N, 4) agent history and (N, 32) keypoint history -> (N, 18, 2) points, float64."""
    return np.concatenate([state.reshape(-1, 2, 2), env.reshape(-1, 16, 2)], 1).astype(np.float64)


def nearest(q: np.ndarray, p: np.ndarray, chunk: int = 16) -> np.ndarray:
    """For each state of q, the distance to the nearest state of p."""
    out = np.empty(len(q))
    for i in range(0, len(q), chunk):
        d = np.sqrt(((q[i:i + chunk, None] - p[None]) ** 2).sum(-1)).max(-1)
        out[i:i + chunk] = d.min(1)
    return out


def gate(d_real: np.ndarray, d_ccil: np.ndarray) -> np.ndarray:
    return d_real ** 2 / (d_real ** 2 + d_ccil ** 2 + EPS)


def quantiles(x: np.ndarray) -> dict:
    return {str(q): float(v) for q, v in zip(QS, np.percentile(x, QS))}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) + P(tie) / 2: the Mann-Whitney statistic with average ranks."""
    v = np.concatenate([pos, neg])
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v))
    ranks[order] = np.arange(1, len(v) + 1)
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, ranks) / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def held_out(job):
    """One subset episode's real rows and labels, against the other episodes' real rows and labels."""
    e, real, rep, lab, lep = job
    r, lb, ro, lo = real[rep == e], lab[lep == e], real[rep != e], lab[lep != e]
    return gate(nearest(r, ro), nearest(r, lo)), gate(nearest(lb, ro), nearest(lb, lo))


def load(args) -> tuple:
    with np.load(args.source) as z:
        src = {k: z[k] for k in z.files}
    if source_module.content_sha256(src) != INPUTS["source_content_sha256"]:
        sys.exit("refused: the source rows are not the protocol's by content")
    for k in ("labels", "candidates", "selection"):
        if sha(getattr(args, k)) != INPUTS[f"{k}_sha256"]:
            sys.exit(f"refused: {k} is not the r = 0.25 pool's")
    with np.load(args.labels) as z:
        lab = {k: z[k] for k in z.files}
    with np.load(args.candidates) as z:
        cand = {k: z[k] for k in z.files if k != "meta"}
    with np.load(args.selection) as z:
        sel = {k: z[k] for k in z.files if k != "meta"}
    return src, lab, cand, sel


def evidence(cand: dict, sel: dict) -> dict:
    mask = sel["mask"]
    return {"candidate_fields": sorted(cand), "selection_fields": sorted(sel),
            "mask_rows": int(mask.sum()), "weight_values_on_mask": sorted(set(sel["weight"][mask].tolist())),
            "finite_error_rows": int(np.isfinite(sel["error"]).sum()),
            "certificates": "none: the selection is acceptance only (data_only, uncertified)"}


def main():
    p = argparse.ArgumentParser()
    for k in ("source", "labels", "candidates", "selection"):
        p.add_argument(f"--{k}", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "docs/studies/evidence/pusht-refiner/stage-a.json")
    p.add_argument("--workers", type=int, default=min(41, os.cpu_count() or 1))
    args = p.parse_args()
    if git("status", "--porcelain"):
        sys.exit("refused: the tree is dirty; Stage A runs from a clean commit")
    src, lab, cand, sel = load(args)
    eps = subset(BASE["roles"]["train"], R, SUBSET_SEED)
    rows = np.flatnonzero(np.isin(src["episode"], eps))
    if not np.isin(src["episode"][lab["src_row"]], eps).all():
        sys.exit("refused: a label is sourced outside the subset")
    counts = {"subset_episodes": len(eps), "subset_rows": len(rows), "candidates": len(cand["accepted"]),
              "labels": len(lab["src_row"])}
    if counts != EXPECTED:
        sys.exit(f"refused: the counts {counts} are not the protocol's")
    if not np.array_equal(sel["mask"], cand["accepted"]) or int(sel["mask"].sum()) != EXPECTED["labels"]:
        sys.exit("refused: the selection is not the accepted labels")

    lab_pts = points(lab["syn_state"], lab["syn_env"])
    src_pts = points(src["state"][lab["src_row"]], src["env"][lab["src_row"]])
    real = points(src["state"][rows], src["env"][rows])
    rep, lep = src["episode"][rows], src["episode"][lab["src_row"]]
    shift = np.sqrt(((lab_pts - src_pts) ** 2).sum(-1))  # (N, 18) per-point shift from the source row

    d_lab_real = nearest(lab_pts, real)
    d_real_lab = nearest(real, lab_pts)
    with ProcessPoolExecutor(args.workers) as ex:
        parts = list(ex.map(held_out, [(e, real, rep, lab_pts, lep) for e in eps]))
    g_real = np.concatenate([a for a, _ in parts])
    g_lab = np.concatenate([b for _, b in parts])
    dup_rows = np.unique(lab["src_row"])
    g_dup = gate(nearest(points(src["state"][dup_rows], src["env"][dup_rows]), real),
                 nearest(points(src["state"][dup_rows], src["env"][dup_rows]), lab_pts))

    separation = auc(g_lab, g_real)
    out = {
        "schema": "ilsb.pusht_refiner.stage_a 1",
        "code": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
                 "files_sha256": {f: sha(ROOT / f) for f in ("tools/pusht_refiner/stage_a.py",
                                                             "tools/pusht_v2/source.py", "tools/pusht_v2/oracle_probe.py",
                                                             "tools/pusht_v2/surface.json")}},
        "inputs": {**INPUTS, "pool_entry": "c30c0f98b1741162feda2165c83f47bd000ba81aabe22e003eef2702e5e65691",
                   "bases": BASES},
        "counts": counts,
        "evidence": evidence(cand, sel),
        "geometry_px": {
            "label_to_source": quantiles(shift.max(1)),
            "label_to_source_previous_frame": quantiles(shift[:, PREV].max(1)),
            "label_to_source_current_frame": quantiles(shift[:, CUR].max(1)),
            "label_to_nearest_real": quantiles(d_lab_real),
            "real_to_nearest_label": quantiles(d_real_lab),
        },
        "relative_gate": {
            "definition": "g = d_real^2 / (d_real^2 + d_CCIL^2 + 1e-6 px^2); d to the nearest subset real row / label",
            "in_sample": {"real_rows": quantiles(gate(np.zeros(len(real)), d_real_lab)),
                          "labels": quantiles(gate(d_lab_real, np.zeros(len(lab_pts))))},
            "held_out_episode": {"real_rows": quantiles(g_real), "labels": quantiles(g_lab),
                                 "auc_labels_over_real": separation},
            "duplication_targets": {"rows": int(len(dup_rows)), "gate_zero": int((g_dup == 0).sum())},
        },
        "decision": ("refused: on held-out episodes the relative gate scores demonstration states nearly as high as CCIL "
                     "labels, so this state representation does not define a correction-specific domain; no refiner "
                     "is trained"),
        "decision_basis": ("the stop rule set in #55's review before any refiner trained: stop if the relative gate cannot "
                           "distinguish the two populations; no numeric threshold was pre-registered"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"counts": counts, "label_to_source_median": out["geometry_px"]["label_to_source"]["50"],
                      "held_out": out["relative_gate"]["held_out_episode"]}))


if __name__ == "__main__":
    main()
