#!/usr/bin/env python3
"""The pilot decisions of the J(r, K | alpha = 0.5) protocol (docs/studies/pendulum-j-r-k.md), by declared rules.

    uv run python tools/pilot.py stopping ROOT --store STORE [--out FILE]   # patience and the step ceiling
    uv run python tools/pilot.py precision ROOT [--out FILE]                # the number of study seeds

The pilots use non-study seeds and non-study evaluation starts, and choose only the stopping rule's patience, the
failure ceiling and the seed count. Neither output carries an arm mean or a contrast mean.

stopping: every unit of the stopping pilot trained its whole schedule and recorded its validation history. For each
candidate patience P, the trainer's rule (stop after P evaluations without an improvement) is replayed on every
history; its regret is the relative excess of the checkpoint it would select over the history's best loss. The
chosen P is the smallest candidate that fires on every unit with a 95th-percentile regret of at most 1%; the
ceiling is twice the latest step at which that P stops, rounded up to a multiple of 1000.

precision: for each primary contrast (every contrast at K = K_max, at every real fraction), the standard deviation
of the paired per-seed differences. The seed count is the smallest n whose normal-approximation half-width
z_c * sd / sqrt(n) is at most 0.05 upright occupancy for every primary contrast, clipped to [20, 50].
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

PATIENCE = (5, 10, 20, 40)
REGRET_Q, REGRET_MAX = 0.95, 0.01
HALF_WIDTH, N_MIN, N_MAX = 0.05, 20, 50


def replay(losses: list[float], patience: int) -> tuple[int | None, int]:
    """(index at which the rule stops, or None; index of the checkpoint selected by then): the trainer's rule with
    min_delta 0, an evaluation improving when strictly below the best so far."""
    best, best_i, stale = math.inf, 0, 0
    for i, v in enumerate(losses):
        if v < best:
            best, best_i, stale = v, i, 0
        else:
            stale += 1
            if stale >= patience:
                return i, best_i
    return None, best_i


def stopping(root: Path, store: Path) -> dict:
    units = json.loads((root / "stages" / "train.json").read_text())["units"]
    per_p = {}
    for p in PATIENCE:
        regrets, stops, never = [], [], []
        for key, u in units.items():
            ev = json.loads((store / "policy" / u["entry"] / "training.json").read_text())["validation"]["evaluations"]
            losses, steps = [e["loss"] for e in ev], [e["step"] for e in ev]
            at, sel = replay(losses, p)
            if at is None:
                never.append(key)
                continue
            regrets.append((losses[sel] - min(losses)) / min(losses))
            stops.append(steps[at])
        per_p[str(p)] = {"fires_on_every_unit": not never, "never_fired": len(never),
                         "regret_q95": float(np.quantile(regrets, REGRET_Q)) if regrets else None,
                         "regret_max": float(max(regrets)) if regrets else None,
                         "latest_stop_step": max(stops) if stops else None,
                         "median_stop_step": float(np.median(stops)) if stops else None}
    ok = [p for p in PATIENCE if per_p[str(p)]["fires_on_every_unit"] and per_p[str(p)]["regret_q95"] <= REGRET_MAX]
    chosen = min(ok) if ok else None
    ceiling = None if chosen is None else int(math.ceil(2 * per_p[str(chosen)]["latest_stop_step"] / 1000) * 1000)
    return {"rule": f"smallest P in {list(PATIENCE)} firing on every unit with q{int(REGRET_Q * 100)} regret <= "
                    f"{REGRET_MAX}; ceiling = 2 x latest stop step, rounded up to 1000",
            "units": len(units), "candidates": per_p, "patience": chosen, "step_budget": ceiling,
            "pass": chosen is not None}


def precision(root: Path) -> dict:
    spec = json.loads((root / "root.json").read_text())["spec"]
    scores = json.loads((root / "stages" / "evaluate.json").read_text())["units"]
    metric, k_max, seeds = spec["analysis"]["metric"], max(spec["synthetic_doses"]), spec["seeds"]
    z = float(norm.ppf(1 - (1 - spec["analysis"]["confidence"]) / 2))
    sds = {}
    for a_, b_ in spec["analysis"]["contrasts"]:
        for r in spec["real_fractions"]:
            kb = 0 if b_ == "bc" else k_max
            d = [scores[f"{a_}/r{r}/k{k_max}/s{s}"]["scores"][metric] - scores[f"{b_}/r{r}/k{kb}/s{s}"]["scores"][metric]
                 for s in seeds]
            sds[f"{a_}-{b_}/r{r}/k{k_max}"] = float(np.std(d, ddof=1))
    need = max(math.ceil((z * sd / HALF_WIDTH) ** 2) for sd in sds.values())
    return {"rule": f"smallest n with z_c * sd / sqrt(n) <= {HALF_WIDTH} for every primary contrast, in "
                    f"[{N_MIN}, {N_MAX}]", "z_c": z, "pilot_seeds": len(seeds), "sd_of_paired_differences": sds,
            "n_required": need, "seeds": min(max(need, N_MIN), N_MAX), "capped": need > N_MAX}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("stopping", "precision"))
    ap.add_argument("root", type=Path)
    ap.add_argument("--store", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = stopping(a.root, a.store or a.root / "store") if a.part == "stopping" else precision(a.root)
    text = json.dumps(out, indent=1, sort_keys=True) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
