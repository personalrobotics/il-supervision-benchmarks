"""#82: every arm on every unit (recovery-fixed-horizon.md, fixedh.json, inventory.json).

    PY tools/recovery_fixed_horizon/run.py --set smoke|development|study --run-dir DIR --workers N \
        --pendulum pendulum_cont_corrected.npz --policy s3321/policy.npz --stage2 S2 --stage3 S3 --stage4 S4

S2, S3 and S4 are the Stage 2 Pendulum, #73 and #76 run directories: the smokes' for the smoke set, the studies'
(from their releases) for the development replay and the study. The study set runs the frozen fresh inventory and
needs --study-go; the development set is #76's 400 starts with the fixed-H learned arm only (descriptive). Each
unit is written atomically to DIR/units/, so a rerun resumes and two runs compare byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixedh as F  # noqa: E402

C = F.C
STUDY = None
_spec = importlib.util.spec_from_file_location("run79", F.W.HERE / "run.py")
R79 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R79)


def name(unit) -> str:
    return f"{unit[0]}-{unit[1]:04d}.json"


def init(set_name, pendulum, policy, stage3, stage2, stage4) -> None:
    global STUDY
    STUDY = F.Study(set_name, pendulum, policy, stage3, stage2, stage4)


def work(unit):
    t0 = time.time()
    rec = STUDY.run_unit(unit)
    return rec, time.time() - t0


# --- the record ---------------------------------------------------------------------------------------------------

q, count = R79.q, R79.count


def gate(recs: list, hits: list) -> dict:
    radii = sorted({r["radius"] for r in recs if r["radius"] is not None})
    n = {str(x): sum(1 for r in recs if r["radius"] == x) for x in radii}
    by = {str(x): sum(1 for r in hits if r["radius"] == x) for x in radii}
    frac = {x: by[x] / n[x] for x in n if n[x]}
    d = F.SPEC["decision"]
    return {"recovered": len(hits), "by_radius": by, "fraction_by_radius": frac,
            "passes": len(hits) >= d["min_recovered"] and bool(frac) and max(frac.values()) >= d[
                "min_fraction_at_some_radius"]}


def sf_recovered(a: dict) -> bool:
    tr = a["transitions"]
    return bool(len(tr) >= 2 and tr[0]["to"] == "recover" and tr[0]["k"] == 0 and tr[1]["to"] == "base")


def fixed_checks(a: dict, tol: dict) -> dict:
    """Entries, handoffs and declared recoveries of one fixed-H arm episode, recomputed from its record."""
    entries, handoffs, audited = [], [], []
    recs = [g for g in a["recoveries"] if g["entry"]["feasible"]]
    for t in a["transitions"]:
        if t["to"] == "recover":
            g = next((g for g in recs if g["k"] == t["k"]), None)
            entries.append(bool(g and t["D"] > tol["delta_out"] and g["entry"]["predicted_envelope"]
                                and g["entry"]["predicted_miss"] <= tol["delta_in"]))  # delta_in = epsilon
    for g in recs:
        if g["handoff"]:
            flags = [(d <= tol["delta_in"] and abs(b - u) / tol["sigma_u"] <= tol["alpha"])
                     for d, b, u in zip(g["D"], g["base_actions"], g["actions"])]
            handoffs.append(bool(len(flags) >= F.WINDOW and all(flags[-F.WINDOW:])
                                 and g["endpoint_miss"] <= tol["delta_in"]))
            audited.append(bool(g["audited"]))
    return {"entries": entries, "handoffs": handoffs, "audited": audited}


def summarize(recs: list, tol: dict) -> dict:
    starts = [r for r in recs if r["kind"] in ("fresh", "start")]
    controls = [r for r in recs if r["kind"] in ("fresh_control", "control")]
    out = {"starts": len(starts), "controls": len(controls), "arms": {}}
    arms = sorted({a for r in recs for a in r["arms"]}, key=F.ARMS.index)
    for arm in arms:
        rs = [r for r in starts if arm in r["arms"]]
        cs = [r for r in controls if arm in r["arms"]]
        if arm == "any_horizon":
            hits = [r for r in rs if r["arms"][arm]["recovered_M"] is not None]
            out["arms"][arm] = {"gate": gate(rs, hits),
                                "recovered_M": count(r["arms"][arm]["recovered_M"] for r in hits)}
            continue
        if arm == "shortest_feasible":
            hits = [r for r in rs if sf_recovered(r["arms"][arm])]
            segs = [g for r in rs + cs for g in r["arms"][arm]["segments"] if g["M"] is not None]
            ent = [v for r in rs + cs for v in R79.entry_valid(r["arms"][arm], tol)]
            hand = [v for r in rs + cs for v in R79.handoff_valid(r["arms"][arm], tol)]
            aud = [bool(g["audited"]) for g in segs if g["declared"]]
        else:
            hits = [r for r in rs if r["arms"][arm]["recovered"]]
            ch = [fixed_checks(r["arms"][arm], tol) for r in rs + cs]
            ent, hand, aud = ([v for c in ch for v in c[k]] for k in ("entries", "handoffs", "audited"))
        eps = rs + cs
        tr = [(r, r["arms"][arm]["transitions"]) for r in eps]
        hand_t = [b["k"] - a_["k"] for _, t in tr for a_, b in zip(t, t[1:]) if a_["to"] == "recover"]
        out["arms"][arm] = {
            "gate": gate(rs, hits),
            "outcome_starts": count(r["arms"][arm]["outcome"] for r in rs),
            "outcome_controls": count(r["arms"][arm]["outcome"] for r in cs),
            "base_reproduced": [sum(r["arms"][arm]["base_reproduced"] for r in eps), len(eps)],
            "entries_valid": [sum(ent), len(ent)], "handoffs_valid": [sum(hand), len(hand)],
            "declared_audited": [sum(aud), len(aud)],
            "handoff_steps": q(hand_t),
            "reentries": sum(max(0, sum(x["to"] == "recover" for x in t) - 1) for _, t in tr),
            "controls_departed": sum(r["arms"][arm]["bc_first_departure"] is not None for r in cs),
            "controls_entered": sum(bool(r["arms"][arm]["transitions"]) for r in cs),
            "controls_handed_back": sum(any(x["to"] == "base" for x in r["arms"][arm]["transitions"]) for r in cs),
            "upright_minus_bc_complete": q([r["arms"][arm]["upright"] - r["arms"][arm]["bc_upright"] for r in eps
                                            if r["arms"][arm]["upright"] is not None])}
    if {"any_horizon", "fixed_learned"} <= set(arms):
        anyh = {r["unit"] for r in starts if r["arms"]["any_horizon"]["recovered_M"] is not None}
        fl = {r["unit"] for r in starts if r["arms"]["fixed_learned"]["recovered"]}
        out["retention"] = [len(anyh & fl), len(anyh)]
    if {"shortest_feasible", "fixed_learned"} <= set(arms):
        pair = count(f"fixed_{int(r['arms']['fixed_learned']['recovered'])}"
                     f"_shortest_{int(sf_recovered(r['arms']['shortest_feasible']))}" for r in starts)
        out["paired_fixed_vs_shortest"] = pair
    return out


def accounted(recs: list) -> bool:
    ok = set(F.SPEC["abstention"]) | set(F.W.SPEC["abstention"])
    for r in recs:
        for arm, a in r["arms"].items():
            if arm == "any_horizon":
                continue
            if not ((a["outcome"] == "complete" and len(a["actions"]) == a["n"]) or
                    (a["outcome"] in ok and len(a["actions"]) <= a["n"])):
                return False
            if not len(a["modes"]) == len(a["actions"]) == len(a["states"]) - 1:
                return False
    return True


def decide(s: dict, recs: list, set_name: str) -> dict:
    A = s["arms"]
    wrappers = [a for a in ("shortest_feasible", "fixed_learned", "fixed_exact") if a in A]
    out = {"exact_gate": A.get("fixed_exact", {}).get("gate", {}).get("passes"),
           "learned_gate": A.get("fixed_learned", {}).get("gate", {}).get("passes"),
           "retention": s.get("retention"),
           "base_reproduced": all(A[a]["base_reproduced"][0] == A[a]["base_reproduced"][1] for a in wrappers),
           "audits": all(A[a][k][0] == A[a][k][1] for a in wrappers
                         for k in ("entries_valid", "handoffs_valid", "declared_audited")),
           "accounted": accounted(recs)}
    ret = s.get("retention")
    out["retention_fraction"] = ret[0] / ret[1] if ret and ret[1] else None
    out["passes"] = bool(set_name != "development" and out["exact_gate"] and out["learned_gate"]
                         and out["retention_fraction"] is not None
                         and out["retention_fraction"] >= F.SPEC["decision"]["min_retention"]
                         and out["base_reproduced"] and out["audits"] and out["accounted"])
    return out


def record(recs: list, identity: dict, set_name: str, tol: dict) -> dict:
    s = summarize(recs, tol)
    notes = {"smoke": "smoke: Stage 2's key-99 starts and #79's smoke controls; not a study outcome",
             "development": "development replay on #76's starts: descriptive, authorizes no change",
             "study": "study: the frozen fresh inventory"}
    return {"schema": "ilsb.recovery_fixed_horizon 1", "set": set_name, "note": notes[set_name],
            "code": C.code_identity(), "identity": identity, "summary": s, "decision": decide(s, recs, set_name),
            "units_sha256": {name((r["kind"], r["unit"]))[:-5]: hashlib.sha256(C.dumps(r).encode()).hexdigest()
                             for r in recs}}


def build_identity(st, args) -> dict:
    s = "smoke" if args.set == "smoke" else "study"
    return {"system": "pendulum", "set": args.set, "ccil_commit": F.S3.ccil_commit(), "epsilon": st.sy.eps,
            "pendulum_sha256": F.S3.sha256(args.pendulum), "policy_sha256": F.S3.sha256(args.policy),
            "inventory_sha256": F.S3.sha256(F.HERE / "inventory.json"),
            "selection_sha256": F.S3.sha256(F.HERE / "selection.json"),
            "tolerances_sha256": F.SPEC["tolerances"]["sha256"],
            "stage2_record_sha256": F.W.R.SPEC["stage2"]["record_sha256"][s],
            "stage3_record_sha256": F.W.R.SPEC["stage3"]["record_sha256"][s],
            "stage4_record_sha256": F.W.SPEC["stage4"]["record_sha256"][s],
            "model0_weights_sha256": F.W.R.SPEC["stage3"]["model0_weights_sha256"], "arms": list(st.arms)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=("smoke", "development", "study"), required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    for a in ("pendulum", "policy", "stage2", "stage3", "stage4"):
        p.add_argument(f"--{a}", type=Path, required=True)
    p.add_argument("--study-go", action="store_true")
    p.add_argument("--limit", type=int, help="development only: the first N units; no record is written")
    args = p.parse_args()
    if args.set == "study" and not args.study_go:
        sys.exit("refused: the study run needs its own go (--study-go)")
    if args.set != "smoke" and C.code_identity()["dirty"]:
        sys.exit("refused: the development replay and the study run from a clean commit")
    if args.set != "smoke" and F.S3.ccil_commit() != F.S3.SPEC["models"]["ccil_commit"]:
        sys.exit(f"refused: this set needs CCIL {F.S3.SPEC['models']['ccil_commit']}, not {F.S3.ccil_commit()}")
    run = args.run_dir
    if (run / "EXIT").exists():
        sys.exit(f"refused: {run} is a finished run; never overwrite one")
    (run / "units").mkdir(parents=True, exist_ok=True)
    (run / "timing").mkdir(exist_ok=True)
    log = C.Log(run / "run.log")
    C.header(log, {"set": args.set, "workers": args.workers, **{a: str(getattr(args, a)) for a in (
        "pendulum", "policy", "stage2", "stage3", "stage4")}})
    inputs = (args.set, args.pendulum, args.policy, args.stage3, args.stage2, args.stage4)
    init(*inputs)  # refuses unless every input is the protocol's
    identity = build_identity(STUDY, args)
    ident_path = run / "identity.json"
    if ident_path.exists():
        if json.loads(ident_path.read_text())["identity"] != identity:
            sys.exit(f"refused: {run} was started with other inputs")
    else:
        C.write_atomic(ident_path, C.dumps({"set": args.set, "code": C.code_identity(), "identity": identity}))
    units = STUDY.units[:args.limit]
    todo = [u for u in units if not (run / "units" / name(u)).exists()]
    log(f"stage units: {len(units)} units x arms {list(STUDY.arms)}, {len(units) - len(todo)} already written")

    def done(u, res):
        rec, secs = res
        C.write_atomic(run / "units" / name(u), C.dumps(rec))
        C.write_atomic(run / "timing" / name(u), C.dumps({"unit": list(u), "seconds": secs}))

    t0 = time.time()
    try:
        C.schedule(work, todo, args.workers, log, f"pendulum fixed-H {args.set}", init, inputs, on_done=done)
    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        (run / "EXIT").write_text("EXIT=1\n")
        raise
    recs = [json.loads((run / "units" / name(u)).read_text()) for u in units]
    if args.limit is None:
        rec = record(recs, identity, args.set, STUDY.tol)
        C.write_atomic(run / "record.json", C.dumps(rec))
        log(f"record {F.S3.sha256(run / 'record.json')} decision " + json.dumps(rec["decision"], sort_keys=True))
        (run / "EXIT").write_text("EXIT=0\n")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
