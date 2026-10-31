#!/usr/bin/env python3
"""Descriptive label diagnostics of a study's synthetic supervision, per (r, K): no rollout, score or test.

    uv run python tools/diagnostics.py SPEC STORE --root NEW_DIR [--out FILE]

The study's candidate sets are looked up in STORE by the harness's own identities (a new run root on the study's
store; nothing is generated, and a lookup that would create an entry is refused). For every (r, K) and every seed of
that dose, on the K view each CCIL arm trains on (its training episodes' rows):
- accepted fraction: accepted rows over the view's rows;
- action displacement |a~ - a| (N m) and state displacement ||z~ - z|| (standardized by the training demonstrations'
  per-coordinate standard deviation) from each accepted row's source demonstration;
- nearest-demonstration distance: the standardized distance from each accepted synthetic state to the nearest
  training demonstration state;
- effective distinct synthetic examples at resolution delta: the number of delta-grid cells of standardized
  (state, action / action bound) space occupied by accepted synthetic rows and by no training demonstration, for
  delta in DELTAS; and the Kish effective count (sum w)^2 / sum w^2 of the accepted rows' weights;
- exact versus learned (same proposals row for row): the fraction of rows with identical proposed actions, the
  fraction whose acceptance differs, and the standardized distance between the two arms' accepted states, alone and
  over the exact arm's own state displacement.
Each quantity is reported per seed (median and 90th percentile over rows where it is a distribution) and summarized
over seeds by the median and range.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ilsb import run
from ilsb.supervision import ccil as sup

DELTAS = (1e-3, 1e-2, 1e-1)


def entries(store: Path) -> int:
    return sum(1 for kind in store.iterdir() if kind.is_dir() for _ in kind.iterdir()) if store.exists() else 0


def q(x) -> dict:
    x = np.asarray(x, np.float64)
    return {"median": float(np.median(x)), "p90": float(np.quantile(x, 0.9)), "max": float(x.max())} if len(x) \
        else {"median": None, "p90": None, "max": None}


def arm_view(ctx, arm, r, k, seed, sigma):
    from ccil.schema import read_candidates
    subset_p, split_p = run.data_units(ctx, r, seed)
    view = sup.view(ctx.store, run.full_candidates(ctx, arm, r, seed, sigma), k) / "candidates.npz"
    s = sup.supervision(arm, subset_p, split_p, view, "train")
    c = read_candidates(view)
    train = set(json.loads(split_p.read_text())["train"])
    in_train = np.isin(c.source_episode_id.astype(str), sorted(train))
    return s, c, in_train


def unit(ctx, r, k, seed, sigma) -> dict:
    bound = ctx.task.action_bound
    out, views = {}, {}
    for arm in ("ccil_learned", "ccil_exact"):
        s, c, in_train = arm_view(ctx, arm, r, k, seed, sigma)
        demo, syn = s.source == "demonstration", s.source == "synthetic"
        zd, ad = s.observations[demo].astype(np.float64), s.actions[demo].astype(np.float64)
        scale = zd.std(axis=0)
        by_source = {(str(e), int(f)): i for i, (e, f) in enumerate(zip(s.episode_id[demo], s.frame[demo]))}
        src = np.array([by_source[(str(e), int(f))] for e, f in zip(s.episode_id[syn], s.frame[syn])], np.int64)
        zs, as_ = s.observations[syn].astype(np.float64), s.actions[syn].astype(np.float64)
        nearest, _ = cKDTree(zd / scale).query(zs / scale)
        cells = {d: ({tuple(x) for x in np.floor(np.hstack([zs / scale, as_ / bound]) / d).astype(np.int64)}
                     - {tuple(x) for x in np.floor(np.hstack([zd / scale, ad / bound]) / d).astype(np.int64)})
                 for d in DELTAS}
        w = s.weights[syn]
        out[arm] = {
            "accepted_fraction": float(np.asarray(c.accepted, bool)[in_train].mean()),
            "synthetic_rows": int(syn.sum()), "demonstration_rows": int(demo.sum()),
            "action_displacement": q(np.linalg.norm(as_ - ad[src], axis=1)),
            "state_displacement": q(np.linalg.norm((zs - zd[src]) / scale, axis=1)),
            "nearest_demonstration_distance": q(nearest),
            "distinct_novel_examples": {str(d): len(cells[d]) for d in DELTAS},
            "kish_effective_rows": float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0}
        [z] = c.observations.values()
        rows = np.flatnonzero(in_train)
        z_src = zd[[by_source[(str(e), int(f))] for e, f in zip(c.source_episode_id[rows], c.source_frame[rows])]]
        views[arm] = {"z": np.asarray(z, np.float64)[rows], "a": np.asarray(c.actions)[rows],
                      "accepted": np.asarray(c.accepted, bool)[rows], "z_src": z_src, "scale": scale}
    lv, ev = views["ccil_learned"], views["ccil_exact"]
    both = lv["accepted"] & ev["accepted"]
    gap = np.linalg.norm((lv["z"][both] - ev["z"][both]) / ev["scale"], axis=1)
    exact_disp = np.linalg.norm((ev["z"][both] - ev["z_src"][both]) / ev["scale"], axis=1)
    moved = exact_disp > 0
    out["exact_vs_learned"] = {
        "identical_proposals": float((lv["a"] == ev["a"]).all(axis=1).mean()),
        "acceptance_differs": float((lv["accepted"] != ev["accepted"]).mean()),
        "state_disagreement": q(gap),
        "disagreement_over_exact_displacement": q(gap[moved] / exact_disp[moved]),
        "exact_rows_with_zero_displacement": float((~moved).mean()) if len(moved) else None}
    return out


def summarize(per_seed: dict) -> dict:
    """Median and range over seeds of every scalar leaf."""
    def leaves(d, prefix=()):
        for k_, v in d.items():
            if isinstance(v, dict):
                yield from leaves(v, prefix + (k_,))
            elif v is not None:
                yield prefix + (k_,), v
    acc = {}
    for rec in per_seed.values():
        for path, v in leaves(rec):
            acc.setdefault("/".join(path), []).append(v)
    return {p: {"median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v)), "seeds": len(v)}
            for p, v in sorted(acc.items())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", type=Path)
    ap.add_argument("store", type=Path)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.root.exists():
        raise SystemExit(f"refused: {a.root} exists; use a new root")
    before = entries(a.store)
    ctx = run.Context(a.spec, a.root, a.store, allow_dirty=False)
    sp = ctx.spec
    fixed = sp["generation"].get("sigma")
    if fixed is None or not {"ccil_learned", "ccil_exact"} <= set(sp["arms"]):
        raise SystemExit("refused: the diagnostics need a fixed sigma per (r, K) and both CCIL arms")
    out = {}
    for r in sp["real_fractions"]:
        for k in sorted(sp["synthetic_doses"]):
            sigma = fixed[f"r{r}/k{k}"]
            per_seed = {str(seed): unit(ctx, r, k, seed, sigma) for seed in ctx.seeds(k)}
            out[f"r{r}/k{k}"] = {"sigma": sigma, "summary": summarize(per_seed), "per_seed": per_seed}
    if entries(a.store) != before:
        raise SystemExit("refused: a lookup created store entries, so the diagnostics did not read the study's own "
                         "candidates")
    doc = {"definitions": __doc__.split("\n\n", 2)[2].strip(), "deltas": list(DELTAS), "study": sp["study"],
           "spec_sha256": ctx.protocol["spec_sha256"], "coordinates": out}
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    print(json.dumps({k_: v["summary"].get("ccil_learned/nearest_demonstration_distance/median") for k_, v in out.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
