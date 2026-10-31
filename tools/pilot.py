#!/usr/bin/env python3
"""The pilot decisions of the J(r, K | alpha = 0.5) protocol (docs/studies/pendulum-j-r-k.md), by declared rules.

    uv run python tools/pilot.py stopping ROOT [--out FILE]    # the convergence check and the step ceiling
    uv run python tools/pilot.py precision ROOT [--out FILE]   # the number of study seeds

The pilots use non-study seeds and non-study evaluation starts, and choose only the training procedure's convergence
settings, the failure ceiling and the seed count. Neither output carries an arm mean or a contrast mean.

stopping: ROOT is the convergence sentinel (studies/pendulum-j-r-k/pilot-stopping.yaml): the extremes of the design
(the smallest and the full real fraction; BC and CCIL at K_max and the largest sigma) on a few non-study seeds,
trained with the study's convergence rule and then on to twice each unit's stop step (extend_after_stop). It passes
when the rule fired on every unit before the ceiling and no unit's extension improved its selected validation loss
by more than 2%. The step ceiling is three times the latest stop step, rounded up to a multiple of 10,000 (a safety
limit only: a study unit that reaches it is an incomplete fit).

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

GAIN_MAX = 0.02  # a doubling of training that improves validation loss by more than this is still converging
CEILING_FACTOR, CEILING_ROUND = 3, 10_000
CONVERGENCE_KEYS = ("validation_interval_epochs", "validation_min_steps", "lr_schedule", "lr_factor", "lr_patience", "min_lr", "patience")
HALF_WIDTH, N_MIN, N_MAX = 0.05, 20, 50


def rung(root: Path, store: Path) -> dict:
    spec = json.loads((root / "root.json").read_text())["spec"]
    units = json.loads((root / "stages" / "train.json").read_text())["units"]
    fired, gains, stops, steps = [], [], [], []
    for key, u in units.items():
        st = json.loads((store / "policy" / u["entry"] / "training.json").read_text())["stopping"]
        steps.append(st["steps"])
        if st["outcome"] != "patience" or not st.get("extension"):
            continue
        fired.append(key)
        gains.append(st["extension"]["relative_gain"])
        stops.append(st["stopped_at_step"])
    ok = len(fired) == len(units) and max(gains) <= GAIN_MAX
    return {"root": str(root), "settings": {k: spec["learner"].get(k) for k in CONVERGENCE_KEYS},
            "units": len(units), "fired": len(fired), "gain_max": max(gains) if gains else None,
            "gain_q95": float(np.quantile(gains, 0.95)) if gains else None,
            "latest_stop_step": max(stops) if stops else None,
            "median_stop_step": float(np.median(stops)) if stops else None, "steps_trained": int(sum(steps)),
            "pass": ok}


def stopping(root: Path, store: Path) -> dict:
    out = rung(root, store)
    ceiling = (int(math.ceil(CEILING_FACTOR * out["latest_stop_step"] / CEILING_ROUND) * CEILING_ROUND)
               if out["pass"] else None)
    return {"rule": f"the rule fires on every unit and the extension to twice the stop step gains at most {GAIN_MAX} "
                    f"relative validation loss on every unit; ceiling = {CEILING_FACTOR} x the latest stop step, "
                    f"rounded up to {CEILING_ROUND}", **out, "step_budget": ceiling}


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
