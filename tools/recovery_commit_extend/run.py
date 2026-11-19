"""#85: the stage's arms on every unit (recovery-commit-extend.md, commit.json, inventory.json).

    PY tools/recovery_commit_extend/run.py --set smoke|a|b --run-dir DIR --workers N \
        --pendulum pendulum_cont_corrected.npz --policy s3321/policy.npz --stage2 S2 --stage3 S3 --stage4 S4

S2, S3 and S4 are the Stage 2 Pendulum, #73 and #76 run directories: the smokes' for the smoke set, the studies'
(from their releases) for the stages. Stage a (commit-and-extend exact) and stage b (the learned arms) run the frozen
fresh inventory and each needs its own --stage-go. Each unit is written atomically to DIR/units/, so a rerun resumes
and two runs compare byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import commit as E  # noqa: E402

F, C = E.F, E.C
STUDY = None
FAILURES = set(E.SPEC["target_failure"]) - {"expired"}


def name(unit) -> str:
    return f"{unit[0]}-{unit[1]:04d}.json"


def init(set_name, pendulum, policy, stage3, stage2, stage4) -> None:
    global STUDY
    STUDY = E.Study(set_name, pendulum, policy, stage3, stage2, stage4)


def work(unit):
    t0 = time.time()
    rec = STUDY.run_unit(unit)
    return rec, time.time() - t0


# --- the record ---------------------------------------------------------------------------------------------------

R82 = None


def r82():
    global R82
    if R82 is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("run82", F.HERE / "run.py")
        R82 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(R82)
    return R82


def gate(recs: list, hits: list) -> dict:
    radii = sorted({r["radius"] for r in recs if r["radius"] is not None})
    n = {str(x): sum(1 for r in recs if r["radius"] == x) for x in radii}
    by = {str(x): sum(1 for r in hits if r["radius"] == x) for x in radii}
    frac = {x: by[x] / n[x] for x in n if n[x]}
    d = E.SPEC["decision"]
    return {"recovered": len(hits), "by_radius": by, "fraction_by_radius": frac,
            "passes": len(hits) >= d["min_recovered"] and bool(frac) and max(frac.values()) >= d[
                "min_fraction_at_some_radius"]}


def authority(c: dict, eps: float) -> bool:
    return bool(c["plan"] is not None and c["predicted_envelope"] and c["predicted_miss"] <= eps)


def ce_checks(a: dict, tol: dict, eps: float) -> dict:
    """Entries, planner calls, target bookkeeping, handoffs and declared recoveries of one commit-and-extend
    episode, recomputed from its record."""
    entries, calls, targets, handoffs, audited = [], [], [], [], []
    for t in a["transitions"]:
        if t["to"] == "recover":
            g = next((g for g in a["recoveries"] if g["k"] == t["k"]), None)
            ok = bool(g and t["D"] > tol["delta_out"] and g["targets"])
            if ok:
                c = [c for c in g["calls"] if c["k"] == t["k"]]
                ok = bool(c and c[-1]["kind"] == "commit" and c[-1]["passed"] and c[-1]["M"] == g["targets"][0]["M"])
            entries.append(ok)
    for g in a["recoveries"]:
        calls += [c["passed"] == authority(c, eps) and c["failure"] in (None, *FAILURES) and
                  (c["failure"] is None) == c["passed"] for c in g["calls"]]
        # every executed step's plan passed: one passing call at each recovery step
        passing = [c["k"] for c in g["calls"] if c["passed"]]
        steps = list(range(g["k"], g["k"] + len(g["actions"])))
        targets.append(passing == steps and all(
            b["M"] > a_["M"] and a_["end"] in E.SPEC["target_failure"] for a_, b in zip(g["targets"], g["targets"][1:])))
        if g["handoff"]:
            flags = [(d <= tol["delta_in"] and abs(b - u) / tol["sigma_u"] <= tol["alpha"])
                     for d, b, u in zip(g["D"], g["base_actions"], g["actions"])]
            handoffs.append(bool(len(flags) >= E.WINDOW and all(flags[-E.WINDOW:])
                                 and g["endpoint_miss"] <= tol["delta_in"] and g["targets"][-1]["end"] == "handoff"))
            audited.append(bool(g["audited"]))
    return {"entries": entries, "calls": calls, "targets": targets, "handoffs": handoffs, "audited": audited}


def recovered(r: dict, arm: str) -> bool:
    a = r["arms"][arm]
    return a["recovered_M"] is not None if arm == "any_horizon" else bool(a["recovered"])


def paths(recs: list) -> dict:
    gs = [g for r in recs for arm in ("ce_exact", "ce_learned") if arm in r["arms"]
          for g in r["arms"][arm]["recoveries"]]
    ends = [t["end"] for g in gs for t in g["targets"]]
    return {"commitment at M > 1": sum(bool(g["targets"]) and g["targets"][0]["M"] > 1 for g in gs),
            "countdown": sum(c["kind"] == "replan" and c["passed"] for g in gs for c in g["calls"]),
            "target failure by the authority test": sum(e in FAILURES for e in ends),
            "target failure by expiry": ends.count("expired"),
            "extension": sum(len(g["targets"]) > 1 for g in gs),
            "handoff": sum(g["handoff"] for g in gs),
            "abstention": sum(r["arms"][arm]["outcome"] in E.SPEC["abstention"] for r in recs
                              for arm in ("ce_exact", "ce_learned") if arm in r["arms"])}


def summarize(recs: list, tol: dict, eps: float) -> dict:
    count, q = r82().count, r82().q
    starts = [r for r in recs if r["kind"] in ("fresh", "start")]
    controls = [r for r in recs if r["kind"] in ("fresh_control", "control")]
    out = {"starts": len(starts), "controls": len(controls), "arms": {}}
    arms = sorted({a for r in recs for a in r["arms"]}, key=E.ARMS.index)
    for arm in arms:
        rs = [r for r in starts if arm in r["arms"]]
        cs = [r for r in controls if arm in r["arms"]]
        hits = [r for r in rs if recovered(r, arm)]
        if arm == "any_horizon":
            out["arms"][arm] = {"gate": gate(rs, hits),
                                "recovered_M": count(r["arms"][arm]["recovered_M"] for r in hits)}
            continue
        eps_ = rs + cs
        if arm == "fixed_learned":
            ch = [r82().fixed_checks(r["arms"][arm], tol) for r in eps_]
            keys = ("entries", "handoffs", "audited")
        else:
            ch = [ce_checks(r["arms"][arm], tol, eps) for r in eps_]
            keys = ("entries", "calls", "targets", "handoffs", "audited")
        checks = {k: [sum(v for c in ch for v in c[k]), sum(len(c[k]) for c in ch)] for k in keys}
        tr = [r["arms"][arm]["transitions"] for r in eps_]
        hand_t = [b["k"] - a_["k"] for t in tr for a_, b in zip(t, t[1:]) if a_["to"] == "recover"]
        adv = [(recovered(r, arm), r["arms"][arm]["advantage"]) for r in rs]
        s = {"gate": gate(rs, hits),
             "outcome_starts": count(r["arms"][arm]["outcome"] for r in rs),
             "outcome_controls": count(r["arms"][arm]["outcome"] for r in cs),
             "base_reproduced": [sum(r["arms"][arm]["base_reproduced"] for r in eps_), len(eps_)],
             "checks": checks, "handoff_steps": q(hand_t),
             "reentries": sum(max(0, sum(x["to"] == "recover" for x in t) - 1) for t in tr),
             "controls_departed": sum(r["arms"][arm]["bc_first_departure"] is not None for r in cs),
             "controls_entered": sum(bool(r["arms"][arm]["transitions"]) for r in cs),
             "controls_handed_back": sum(any(x["to"] == "base" for x in r["arms"][arm]["transitions"]) for r in cs),
             "advantage": {"recovered": [sum(a["delta"] >= 0 for h, a in adv if h), sum(1 for h, _ in adv if h)],
                           "not_recovered": [sum(a["delta"] >= 0 for h, a in adv if not h),
                                             sum(1 for h, _ in adv if not h)],
                           "delta": q([a["delta"] for _, a in adv]),
                           "hold": [sum(a["hold"] for _, a in adv), sum(a["bc_hold"] for _, a in adv), len(adv)]}}
        if arm != "fixed_learned":
            gs = [g for r in rs for g in r["arms"][arm]["recoveries"]]
            s["committed_M_at_entry"] = count(g["targets"][0]["M"] for g in gs if g["targets"])
            s["committed_M_at_handoff"] = count(g["targets"][-1]["M"] for g in gs if g["handoff"])
            s["target_ends"] = count(t["end"] for g in gs for t in g["targets"])
            s["targets_per_recovery"] = count(len(g["targets"]) for g in gs)
        out["arms"][arm] = s
    if "any_horizon" in arms and "ce_learned" in arms:
        anyh = {r["unit"] for r in starts if recovered(r, "any_horizon")}
        ce = {r["unit"] for r in starts if recovered(r, "ce_learned")}
        out["retention"] = [len(anyh & ce), len(anyh)]
    present = [a for a in E.ARMS if a in arms and all(a in r["arms"] for r in starts)]
    if len(present) > 1:
        out["paired"] = {"arms": present, "counts": count(
            "".join(str(int(recovered(r, a))) for a in present) for r in starts)}
    out["paths"] = paths(recs)
    return out


def accounted(recs: list) -> bool:
    ok = set(E.SPEC["abstention"]) | set(F.SPEC["abstention"])
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
    wrappers = [a for a in ("fixed_learned", "ce_exact", "ce_learned") if a in A]
    out = {"base_reproduced": all(A[a]["base_reproduced"][0] == A[a]["base_reproduced"][1] for a in wrappers),
           "audits": all(v[0] == v[1] for a in wrappers for v in A[a]["checks"].values()),
           "accounted": accounted(recs)}
    mech = out["base_reproduced"] and out["audits"] and out["accounted"]
    if "ce_exact" in A:
        out["stage_a"] = {"exact_gate": A["ce_exact"]["gate"]["passes"]}
        out["stage_a"]["passes"] = bool(set_name in ("a", "smoke") and out["stage_a"]["exact_gate"] and mech)
    if "ce_learned" in A:
        ret = s.get("retention")
        frac = ret[0] / ret[1] if ret and ret[1] else None
        fl = A.get("fixed_learned", {}).get("gate", {}).get("recovered")
        b = {"learned_gate": A["ce_learned"]["gate"]["passes"], "retention": ret, "retention_fraction": frac,
             "beats_fixed": [A["ce_learned"]["gate"]["recovered"], fl]}
        b["passes"] = bool(set_name in ("b", "smoke") and b["learned_gate"] and frac is not None
                           and frac >= E.SPEC["decision"]["min_retention"] and fl is not None
                           and A["ce_learned"]["gate"]["recovered"] > fl and mech)
        out["stage_b"] = b
    return out


def record(recs: list, identity: dict, set_name: str, tol: dict, eps: float) -> dict:
    s = summarize(recs, tol, eps)
    notes = {"smoke": "smoke: Stage 2's key-99 starts and #79's smoke controls; not a study outcome",
             "a": "stage a: commit-and-extend exact on the frozen fresh inventory",
             "b": "stage b: the learned arms on the frozen fresh inventory"}
    return {"schema": "ilsb.recovery_commit_extend 1", "set": set_name, "note": notes[set_name],
            "code": C.code_identity(), "identity": identity, "summary": s, "decision": decide(s, recs, set_name),
            "units_sha256": {name((r["kind"], r["unit"]))[:-5]: hashlib.sha256(C.dumps(r).encode()).hexdigest()
                             for r in recs}}


def build_identity(st, args) -> dict:
    s = "smoke" if args.set == "smoke" else "study"
    return {"system": "pendulum", "set": args.set, "ccil_commit": F.S3.ccil_commit(), "epsilon": st.sy.eps,
            "pendulum_sha256": F.S3.sha256(args.pendulum), "policy_sha256": F.S3.sha256(args.policy),
            "inventory_sha256": F.S3.sha256(E.HERE / "inventory.json"),
            "selection_sha256": F.S3.sha256(E.HERE / "selection.json"),
            "tolerances_sha256": E.SPEC["tolerances"]["sha256"],
            "stage2_record_sha256": F.W.R.SPEC["stage2"]["record_sha256"][s],
            "stage3_record_sha256": F.W.R.SPEC["stage3"]["record_sha256"][s],
            "stage4_record_sha256": F.W.SPEC["stage4"]["record_sha256"][s],
            "model0_weights_sha256": F.W.R.SPEC["stage3"]["model0_weights_sha256"],
            "arms": {k: list(v) for k, v in st.arms.items()}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=("smoke", "a", "b"), required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    for a in ("pendulum", "policy", "stage2", "stage3", "stage4"):
        p.add_argument(f"--{a}", type=Path, required=True)
    p.add_argument("--stage-go", action="store_true")
    args = p.parse_args()
    if args.set != "smoke" and not args.stage_go:
        sys.exit("refused: each stage run needs its own go (--stage-go)")
    if args.set != "smoke" and C.code_identity()["dirty"]:
        sys.exit("refused: the stages run from a clean commit")
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
    units = STUDY.units
    todo = [u for u in units if not (run / "units" / name(u)).exists()]
    log(f"stage units: {len(units)} units x arms {identity['arms']}, {len(units) - len(todo)} already written")

    def done(u, res):
        rec, secs = res
        C.write_atomic(run / "units" / name(u), C.dumps(rec))
        C.write_atomic(run / "timing" / name(u), C.dumps({"unit": list(u), "seconds": secs}))

    t0 = time.time()
    try:
        C.schedule(work, todo, args.workers, log, f"pendulum commit-extend {args.set}", init, inputs, on_done=done)
    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        (run / "EXIT").write_text("EXIT=1\n")
        raise
    recs = [json.loads((run / "units" / name(u)).read_text()) for u in units]
    rec = record(recs, identity, args.set, STUDY.tol, STUDY.sy.eps)
    C.write_atomic(run / "record.json", C.dumps(rec))
    log(f"record {F.S3.sha256(run / 'record.json')} decision " + json.dumps(rec["decision"], sort_keys=True))
    (run / "EXIT").write_text("EXIT=0\n")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
