"""Pre-freeze development evidence for PushT v2: the transition model on the two-frame history state, fitted only.

    uv run python tools/pusht_v2/history_fit.py DERIVED_FULL OUT_DIR

DERIVED_FULL is derive.py's full dataset. CCIL converts it, then `ccil diagnose` (data.prepare, transition.fit,
labels.diagnose: no candidates) runs twice on the training episodes: with v2's configuration (pusht.yaml:
transition_model.structure history_shift), and with the structure removed (generation.method noisy_action_fe kept). For
each fitted model, on the training and the held-out transitions, the per-row max-abs error in pixels of
- the history shift: delta's older-slot rows against z[newer slot] - z[older slot], which is exact by construction;
- the newest slot: delta's newest-slot rows against the recorded next frame.
No label is generated. Writes OUT_DIR/history-fit.json.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
PREV = np.r_[0:2, 4:20]  # state [x_{t-1}(2), x_t(2)] then environment_state [x_{t-1}(16), x_t(16)]
CUR = np.r_[2:4, 20:36]


def fit(out: Path, cfg: dict) -> dict:
    out.mkdir()
    (out / "pusht.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    r = subprocess.run([sys.executable, "-m", "ccil", "diagnose", "--config", "pusht.yaml"], cwd=out,
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"ccil diagnose exited {r.returncode}:\n{r.stderr[-1500:]}")
    from ccil.data.prepare import load_dataset
    from ccil.models.mlp import MLPTransitionModel
    fitted, = (out / "runs").glob("transition-fit-*")
    prepared, = (out / "runs").glob("data-prepare-*")
    model = MLPTransitionModel.load(fitted / "checkpoints" / "transition_model.npz")
    ds = load_dataset(prepared / "data" / "dataset.npz")
    val = set(json.loads((prepared / "data" / "split.json").read_text())["validation"])
    q = lambda x: {p: float(np.percentile(x, p)) for p in (50, 90, 99, 100)}  # noqa: E731
    res = {}
    for name, keep in (("held_out", lambda e: e in val), ("train", lambda e: e not in val)):
        z, a, zn = (np.concatenate(x) for x in zip(*[
            (np.concatenate([ep.observations["state"], ep.observations["state.environment_state"]], 1)[:-1],
             ep.actions[:-1],
             np.concatenate([ep.observations["state"], ep.observations["state.environment_state"]], 1)[1:])
            for ep in ds.episodes if keep(ep.episode_id)]))
        z, a, zn = z.astype(np.float64), a.astype(np.float64), zn.astype(np.float64)
        d = model.delta(z, a)
        shift = np.abs(d[:, PREV] - (z[:, CUR] - z[:, PREV])).max(1)
        newest = np.abs(d[:, CUR] - (zn[:, CUR] - z[:, CUR])).max(1)
        res[name] = {"transitions": len(z), "history_shift_px": q(shift), "newest_slot_px": q(newest),
                     "history_shift_within_1e-3_px": float((shift <= 1e-3).mean())}
    metrics = json.loads((fitted / "metrics.json").read_text())["values"]
    return {"structure": cfg["transition_model"].get("structure", "none"), "errors": res,
            "held_out_mean_l2": metrics["held_out"]["mean_l2"], "lipschitz_measured": metrics["lipschitz_measured"],
            "checkpoint_sha256": hashlib.sha256((fitted / "checkpoints" / "transition_model.npz").read_bytes()).hexdigest()}


def main(derived: Path, out: Path) -> None:
    import importlib.metadata as md
    from ccil.lerobot import convert
    out.mkdir(parents=True)
    convert(derived, out / "pusht.npz", out / "conversion.json")
    base = yaml.safe_load((HERE / "pusht.yaml").read_text())
    base["data"]["path"] = str((out / "pusht.npz").resolve())
    plain = json.loads(json.dumps(base))
    for k in ("structure", "history_steps"):
        plain["transition_model"].pop(k)
    doc = {"ccil": json.loads(md.distribution("ccil").read_text("direct_url.json"))["vcs_info"]["commit_id"],
           "config": "tools/pusht_v2/pusht.yaml", "note": "development evidence observed before v2's freeze; no labels",
           "structured": fit(out / "structured", base), "unstructured": fit(out / "unstructured", plain)}
    (out / "history-fit.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    for k in ("structured", "unstructured"):
        e = doc[k]["errors"]["held_out"]
        print(f"{k}: held-out shift p50 {e['history_shift_px'][50]:.3g} max {e['history_shift_px'][100]:.3g}; "
              f"newest p50 {e['newest_slot_px'][50]:.3g} max {e['newest_slot_px'][100]:.3g}; "
              f"mean L2 {doc[k]['held_out_mean_l2']:.3f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
