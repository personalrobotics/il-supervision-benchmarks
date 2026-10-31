#!/usr/bin/env python3
"""The pilot decisions of the J(r, K | alpha = 0.5) protocol (docs/studies/pendulum-j-r-k.md), by declared rules.

    uv run python tools/pilot.py stopping ROOT [--out FILE]    # the convergence check and the step ceiling
    uv run python tools/pilot.py precision ROOT [--out FILE]   # the number of study seeds
    uv run python tools/pilot.py sigma ROOT [--out FILE]       # the study's sigma per (r, K)

The pilots use non-study seeds and non-study evaluation starts, and choose only the training procedure's convergence
settings, the failure ceiling and the seed count. Neither output carries an arm mean or a contrast mean.

stopping: ROOT is the convergence sentinel (studies/pendulum-j-r-k/pilot-stopping.yaml): the extremes of the design
(the smallest and the full real fraction; BC and CCIL at K_max and the largest sigma) on a few non-study seeds,
trained with the study's convergence rule and then on to twice each unit's stop step (extend_after_stop). It passes
when the rule fired on every unit before the ceiling, every CCIL unit had drawn each of its synthetic rows by its
stop, and every unit's extension passes the mixed criterion: it improved the selected validation loss by at most 2%,
or it lowered the validation RMSE in raw action units (N m) by at most 1e-3. The absolute tolerance was added after
the sentinel failed the relative-only criterion on near-zero-loss fits, and before the study ran. The step ceiling is
three times the latest stop step, rounded up to a multiple of 10,000 (a safety limit only: a study unit that reaches it
is an incomplete fit).

precision: for each primary contrast (every contrast at K = K_max, at every real fraction), the standard deviation
of the paired per-seed differences. The seed count is the smallest n whose normal-approximation half-width
z_c * sd / sqrt(n) is at most 0.05 upright occupancy for every primary contrast, clipped to [20, 50].

sigma: from the precision pilot's train stage (validation data only, no evaluation), one sigma per (r, K) for every
study seed: among the grid's sigmas feasible on every pilot seed, the one with the least mean validation error over
the pilot seeds; ties go to the smaller sigma.
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
RMSE_GAIN_MAX = 1e-3  # ... unless it lowers the raw validation RMSE (N m) by no more than this: 0.033% of the 3 N m bound
CEILING_FACTOR, CEILING_ROUND = 3, 10_000
CONVERGENCE_KEYS = ("validation_interval_epochs", "validation_min_steps", "lr_schedule", "lr_factor", "lr_patience", "min_lr", "patience")
HALF_WIDTH, N_MIN, N_MAX = 0.05, 20, 50


def rung(root: Path, store: Path) -> dict:
    spec = json.loads((root / "root.json").read_text())["spec"]
    units = json.loads((root / "stages" / "train.json").read_text())["units"]
    fired, gains, rmse_gains, passed, stops, steps, coverage = [], [], [], [], [], [], {}
    for key, u in units.items():
        st = json.loads((store / "policy" / u["entry"] / "training.json").read_text())["stopping"]
        steps.append(st["steps"])
        if st["outcome"] != "patience" or not st.get("extension"):
            continue
        fired.append(key)
        ext = st["extension"]
        gains.append(ext["relative_gain"])
        rmse_gains.append(raw_rmse_gain(store / "policy" / u["entry"] / "policy.npz",
                                        ext["selected_loss_at_stop"], ext["best_loss_extended"]))
        passed.append(gains[-1] <= GAIN_MAX or rmse_gains[-1] <= RMSE_GAIN_MAX)
        stops.append(st["stopped_at_step"])
        rec = json.loads((store / "policy" / u["entry"] / "training.json").read_text())["exposure"]
        coverage[key] = {"demonstration_epoch_steps": rec["demonstration_epoch_steps"],
                         "demonstration_epochs": rec["demonstration_epochs"],
                         "synthetic_at_stop": (rec.get("synthetic_coverage") or {}).get("at_stop")}
    ok = (len(fired) == len(units) and all(passed)
          and all(c["synthetic_at_stop"] is None or c["synthetic_at_stop"]["all_rows_drawn"] for c in coverage.values()))
    return {"root": str(root), "settings": {k: spec["learner"].get(k) for k in CONVERGENCE_KEYS},
            "units": len(units), "fired": len(fired), "gain_max": max(gains) if gains else None,
            "gain_q95": float(np.quantile(gains, 0.95)) if gains else None,
            "relative_only_pass": len(fired) == len(units) and max(gains) <= GAIN_MAX,
            "raw_rmse_gain_max": max(rmse_gains) if rmse_gains else None,
            "extension_failures": [k for k, g, d in zip(fired, gains, rmse_gains)
                                   if g > GAIN_MAX and d > RMSE_GAIN_MAX],
            "latest_stop_step": max(stops) if stops else None,
            "median_stop_step": float(np.median(stops)) if stops else None, "steps_trained": int(sum(steps)),
            "exposure": coverage,
            "pass": ok}


def raw_rmse_gain(policy: Path, loss_stop: float, loss_extended: float) -> float:
    """The extension's drop in validation RMSE in raw action units. The loss is the mean squared error in the min-max
    scaled action space ([-1, 1]), so for a one-dimensional action RMSE_raw = (max - min) / 2 * sqrt(loss)."""
    with np.load(policy) as z:
        lo, hi = z["scaler/act_min"], z["scaler/act_max"]
    if lo.shape != (1,):
        raise SystemExit(f"{policy}: the raw RMSE criterion is defined for a one-dimensional action, not {lo.shape}")
    return float((hi[0] - lo[0]) / 2 * (math.sqrt(loss_stop) - math.sqrt(loss_extended)))


def stopping(root: Path, store: Path) -> dict:
    out = rung(root, store)
    ceiling = (int(math.ceil(CEILING_FACTOR * out["latest_stop_step"] / CEILING_ROUND) * CEILING_ROUND)
               if out["pass"] else None)
    return {"rule": f"the rule fires on every unit and, on every unit, the extension to twice the stop step gains at "
                    f"most {GAIN_MAX} relative validation loss or at most {RMSE_GAIN_MAX} N m raw validation RMSE; ceiling = {CEILING_FACTOR} x the latest stop step, "
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


def sigma(root: Path) -> dict:
    spec = json.loads((root / "root.json").read_text())["spec"]
    sel = json.loads((root / "stages" / "train.json").read_text())["selection"]
    seeds, out = spec["seeds"], {}
    for r in spec["real_fractions"]:
        for k in sorted(spec["synthetic_doses"]):
            per_seed = [sel[f"r{r}/k{k}/s{s}"]["candidates"] for s in seeds]
            common = sorted(set.intersection(*(set(c) for c in per_seed)), key=float)
            mean = {s_: float(np.mean([c[s_] for c in per_seed])) for s_ in common}
            out[f"r{r}/k{k}"] = {"mean_validation_mse": mean,
                                 "sigma": float(min(common, key=lambda s_: (mean[s_], float(s_)))) if common else None}
    return {"rule": "per (r, K): among sigmas feasible on every pilot seed, the least mean validation error over the "
                    "pilot seeds; ties go to the smaller sigma", "pilot_seeds": seeds, "selection": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("stopping", "precision", "sigma"))
    ap.add_argument("root", type=Path)
    ap.add_argument("--store", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = (stopping(a.root, a.store or a.root / "store") if a.part == "stopping"
           else precision(a.root) if a.part == "precision" else sigma(a.root))
    text = json.dumps(out, indent=1, sort_keys=True) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
