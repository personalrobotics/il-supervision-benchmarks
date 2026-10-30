#!/usr/bin/env python3
"""Check the Pendulum task adapter against CCIL's originals, bit for bit.

    uv run python tools/port_check.py --ccil PATH_TO_CCIL_CHECKOUT [--out FILE]

- the transition map: ilsb.tasks.pendulum_map.step against CCIL tools/vector_env/pendulum.step on random states and
  actions (both variants, including states inside the discontinuity bands);
- the scorer: ilsb.tasks.pendulum.Task.score against CCIL tools/pendulum_eval/profiles.pendulum_swingup_v2
  (upright occupancy, normalized height, final hold) on rollouts of the corrected demonstrator;
- the evaluation starts: Task.starts against CCIL tools/m3_pendulum/env.initial_states.
The CCIL checkout must be at the commit the harness pins.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ilsb.tasks import pendulum, pendulum_map


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ccil", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    for p in ("tools/vector_env", "tools/pendulum_eval", "tools/m3_pendulum"):
        sys.path.insert(0, str(a.ccil / p))
    import env as ccil_env
    import pendulum as ccil_map
    import profiles
    rng = np.random.default_rng(0)
    out = {"ccil_commit": subprocess.run(["git", "-C", str(a.ccil), "rev-parse", "HEAD"], capture_output=True,
                                         text=True).stdout.strip()}
    th = np.concatenate([rng.uniform(-np.pi, np.pi, 4000), np.pi / 6 + rng.uniform(-0.03, 0.03, 500),
                         np.pi / 3 + rng.uniform(-0.03, 0.03, 500)])
    s = np.stack([np.sin(th), np.cos(th), rng.uniform(-8, 8, len(th))], 1)
    act = rng.uniform(-3, 3, len(th))
    for disc in (False, True):
        mine, fl = pendulum_map.step(s, act, disc)
        theirs, fl2 = ccil_map.step(s, act, disc)
        out[f"map_{'discontinuous' if disc else 'continuous'}"] = {
            "states": len(s), "bit_identical": bool(np.array_equal(mine, theirs) and np.array_equal(fl, fl2)),
            "flips": int(fl.sum())}
    task = pendulum.Task(eval_episodes=20)
    out["starts_identical"] = bool(np.array_equal(task.starts(), ccil_env.initial_states(task.eval_seed)[:20]))
    [ro] = task.rollouts([lambda o: pendulum.expert(o)])
    mine = task.score(ro)
    theirs = profiles.pendulum_swingup_v2(ro["states"][:, :-1], ro["applied"][..., None])
    out["scorer"] = {k: {"ilsb": mine[k], "ccil": float(np.mean(theirs[k])), "equal": mine[k] == float(np.mean(theirs[k]))}
                     for k in ("upright_occupancy", "normalized_height", "final_hold")}
    out["all_pass"] = bool(out["map_continuous"]["bit_identical"] and out["map_discontinuous"]["bit_identical"]
                           and out["map_discontinuous"]["flips"] > 0 and out["starts_identical"]
                           and all(v["equal"] for v in out["scorer"].values()))
    text = json.dumps(out, indent=1, sort_keys=True) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    print(text)
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
