"""The PushT v2 BC feasibility gate (gate.json): velocity-aware single-step BC, three seeds, pass or fail.

    python tools/pusht_v2/gate.py --root ROOT --source DATA --lerobot-python PY [--smoke] [stages...]

DATA is lerobot/pusht_keypoints at the protocol's revision; PY is a LeRobot 0.6.1 environment with this commit's
plugin installed (tools/pusht_v2/plugin). Stages:
  derive    the velocity-aware dataset, full and training-only (derive.py), and the adapter check (check.py)
  train     bc on the training-only dataset for each seed, with the pilot's supervised scheduler
  select    each seed's checkpoint with the lowest held-out action error, on the full derived dataset
  evaluate  each selected checkpoint through an unchanged lerobot-eval, env pusht_velocity
  decide    the gate statistic against the predeclared threshold
A rerun resumes as the pilot's runner does. No CCIL code runs here.
"""
import hashlib
import importlib.metadata as md
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pusht_pilot"))
from run import Pilot, PilotError, now, sha256, tree, write_json  # noqa: E402

REPO = HERE.parents[1]
STAGES = ("derive", "train", "select", "evaluate", "decide")
PLUGIN = HERE / "plugin" / "lerobot_env_pusht_velocity.py"


class Gate(Pilot):
    PROTOCOL = HERE / "gate.json"

    def identity(self) -> dict:
        git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
        freeze = subprocess.run(["uv", "pip", "freeze", "--python", str(self.py)], capture_output=True, text=True).stdout
        plugin = subprocess.run([str(self.py), "-c", "import lerobot_env_pusht_velocity as m; print(m.__file__)"],
                                capture_output=True, text=True).stdout.strip()
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip() if shutil.which("nvidia-smi") else None
        return {"protocol_sha256": sha256(self.PROTOCOL), "smoke": self.smoke, "ilsb_commit": git("rev-parse", "HEAD"),
                "ilsb_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml", "uv.lock")),
                "ccil": md.version("ccil"), "lerobot_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
                "lerobot": next((ln.split("==")[1] for ln in freeze.splitlines() if ln.startswith("lerobot==")), None),
                "plugin_installed_sha256": sha256(plugin) if plugin else None, "plugin_sha256": sha256(PLUGIN),
                "source_sha256": tree(self.source), "host": platform.node(), "gpus": gpus,
                "python": platform.python_version(), "numpy": np.__version__}

    def check_root(self) -> None:
        ident = self.identity()
        if ident["lerobot"] != self.protocol["lerobot"]["version"]:
            raise PilotError(f"lerobot is {ident['lerobot']}, the protocol pins {self.protocol['lerobot']['version']}")
        if ident["plugin_installed_sha256"] != ident["plugin_sha256"]:
            raise PilotError("the installed lerobot_env_pusht_velocity is not this commit's plugin; reinstall it")
        path = self.root / "root.json"
        if path.exists():
            old = json.loads(path.read_text())
            keep = ("protocol_sha256", "smoke", "ilsb_commit", "lerobot_freeze_sha256", "plugin_sha256", "source_sha256",
                    "host")
            diff = [k for k in keep if old.get(k) != ident.get(k)]
            if diff:
                raise PilotError(f"{path} was created under another identity ({diff}); use a new root")
        else:
            if ident["ilsb_dirty"] and not self.smoke:
                raise PilotError("the gate runs from a clean, committed ilsb checkout")
            write_json(path, {**ident, "created": now(), "command": sys.argv})
            self.log(f"root created: ilsb {ident['ilsb_commit'][:12]}, lerobot {ident['lerobot']}, "
                     f"plugin {ident['plugin_sha256'][:12]}, smoke {self.smoke}, gpus {ident['gpus']}")

    def held_out(self) -> list:
        return self.protocol["split"]["held_out"]

    def selection_source(self) -> Path:
        return self.root / "derived" / "full"

    def derive(self) -> None:
        if self.marker("arms"):
            return
        d = self.root / "derived"
        if d.exists():
            d.rename(self.root / f"derived.incomplete-{int(time.time())}")
        d.mkdir()
        (d / "held_out.json").write_text(json.dumps(self.held_out()))
        for name, extra in (("full", []), ("train", ["--exclude", str(d / "held_out.json")])):
            r = subprocess.run([sys.executable, str(HERE / "derive.py"), str(self.source), str(d / name), *extra],
                               capture_output=True, text=True)
            if r.returncode:
                raise PilotError(f"derive {name} exited {r.returncode}:\n{r.stderr[-1500:]}")
            self.log(f"derive: {r.stdout.strip()}")
        r = subprocess.run([str(self.py), str(HERE / "check.py"), str(self.source), str(d / "full"), str(d / "check.json")],
                           capture_output=True, text=True)
        check = json.loads((d / "check.json").read_text()) if (d / "check.json").exists() else {}
        if r.returncode or not check.get("pass"):
            raise PilotError(f"the adapter check failed (exit {r.returncode}): {check.get('checks')}\n{r.stderr[-1500:]}")
        if check["plugin_sha256"] != sha256(PLUGIN):
            raise PilotError("the adapter check ran another plugin")
        full, train = (json.loads((d / f"{n}.json").read_text()) for n in ("full", "train"))
        if train["episodes"] != full["episodes"] - len(self.held_out()):
            raise PilotError(f"training-only dataset has {train['episodes']} episodes")
        write_json(self.root / "arms.json", {"arms": {"bc": {"path": "derived/train", "frames": train["frames"],
                                                             "episodes": train["episodes"], "files": tree(d / "train")}},
                                             "check": check["checks"], "completed": now()})
        self.log(f"derive: adapter check passed; bc trains on {train['episodes']} episodes, {train['frames']} frames")

    def decide(self) -> None:
        g = self.protocol["gate"]
        res = {}
        for arm, seed in self.units():
            unit = self.root / "units" / f"{arm}-s{seed}"
            ev, sel = (json.loads((unit / f"{n}.json").read_text()) for n in ("eval", "select"))
            if sha256(unit / "eval" / "eval_info.json") != ev["eval_info_sha256"]:
                raise PilotError(f"{arm}-s{seed}: eval_info.json changed")
            res[seed] = {"avg_max_reward": ev["avg_max_reward"], "pc_success": ev["pc_success"],
                         "selected_step": sel["selected"]["step"],
                         "held_out_mean_l2": sel["scores"]["mean_l2"][str(sel["selected"]["step"])]}
        stat = float(np.mean([v["avg_max_reward"] for v in res.values()]))
        ref = g["references"]
        fraction = (stat - ref["hold"]) / (ref["demonstration_replay"] - ref["hold"])
        passed = fraction >= g["fraction"]
        write_json(self.root / "decision.json", {"per_seed": res, "statistic": stat, "gap_fraction": fraction,
                                                 "threshold": g["threshold"], "pass": bool(passed),
                                                 "rule": g["rule"], "smoke": self.smoke, "completed": now()})
        self.log(f"decide: mean coverage {stat:.4f} recovers {fraction:.3f} of the gap (needs {g['fraction']}); "
                 f"gate {'PASSED' if passed else 'FAILED'}")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True, help="lerobot/pusht_keypoints at the protocol's revision")
    ap.add_argument("--lerobot-python", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("stages", nargs="*", default=list(STAGES))
    a = ap.parse_args()
    g = Gate(a.root.resolve(), a.source.resolve(), a.lerobot_python, a.smoke)
    try:
        g.check_root()
        for s in STAGES:
            if s in a.stages:
                g.log(f"stage {s}: start")
                getattr(g, s)()
                if s in ("train", "select", "evaluate"):
                    write_json(g.root / f"stage-{s}.json", {"stage": s, "completed": now()})
                g.log(f"stage {s}: done")
    except PilotError as e:
        g.log(f"STOPPED: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
