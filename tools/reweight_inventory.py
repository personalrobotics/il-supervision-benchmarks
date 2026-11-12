"""Inventory of the historical artifacts for #40's Stage A, by recorded identity and hash (read-only).

    python tools/reweight_inventory.py probe --kind KIND --root RUN_ROOT > part.json      (on the host holding the run)
    python tools/reweight_inventory.py combine part.json ... [--out docs/studies/evidence/reweight/inventory.json]

probe   reads one run root and writes nothing. KIND is one of
          pendulum-control  a Pendulum relabel control (positive- or capture-control run directory)
          pusht-v22         the PushT v2.2 run root
          pusht-surface     a PushT surface stage's run root
        It reports the run's own record hashes, every policy checkpoint the study selected (file SHA-256), and the
        label evidence per pool: the CCIL files present, their hashes, and the per-row fields they carry.
combine checks each part against the committed records of this repository and classifies every label source:
          full          CCIL selection (mask, weight, finite error) and certificates (per-row classification)
          partial       some of those fields: e.g. a selection mask and weight without certificates or error
          oracle        demonstrator relabels: trusted for the retrospective check only, never CCIL-certified
          unverifiable  a file or hash the committed record names is missing or differs: excluded
A missing field is reported as missing. Nothing is inferred from configuration, regenerated, or defaulted.
"""
import argparse
import glob
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EV = REPO / "docs/studies/evidence"


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def one(pattern: str):
    hits = sorted(glob.glob(pattern))
    return hits[0] if len(hits) == 1 else None


def ccil_evidence(runs: Path) -> dict:
    """The CCIL files of one generate/curate run directory and the per-row fields they carry, read as written."""
    import numpy as np
    out = {}
    for name, pat in (("candidates", "labels-generate-*/labels/candidates.npz"),
                      ("selection", "labels-curate-*/labels/selection.npz"),
                      ("certificates", "labels-*/labels/certificates.npz"),
                      ("calibration", "labels-*/labels/calibration.*"),
                      ("diagnosis", "labels-diagnose-*/diagnosis/diagnosis.json"),
                      ("transition_model", "transition-fit-*/checkpoints/transition_model.npz")):
        p = one(str(runs / pat))
        out[name] = {"path": os.path.relpath(p, runs), "sha256": sha(p)} if p else None
    if out["selection"]:
        s = np.load(runs / out["selection"]["path"], allow_pickle=False)
        meta = json.loads(str(s["meta"]))
        w = s["weight"]
        out["selection"].update({
            "schema": f"{meta['schema']} {meta['version']}", "policy": meta["policy"],
            "evidence_mode": meta["evidence_mode"], "evidence_allowed": meta["evidence_allowed"],
            "certificates_sha256": meta["certificates_sha256"], "calibration_sha256": meta.get("calibration_sha256"),
            "candidates_sha256": meta["candidates_sha256"], "error_spec": meta["error"], "caveats": meta["caveats"],
            "rows": int(len(w)), "mask_rows": int(s["mask"].sum()),
            "weight_values_on_mask": sorted({float(x) for x in np.unique(w[s["mask"]])}),
            "finite_error_rows": int(np.isfinite(s["error"]).sum())})
    if out["diagnosis"]:
        out["diagnosis"]["evidence_class"] = json.loads((runs / out["diagnosis"]["path"]).read_text()).get(
            "evidence_class")
    return out


def probe_pendulum_control(root: Path) -> dict:
    import numpy as np
    rec = one(str(root / "*-control.json"))
    policies = {f"{Path(p).parts[-3]}/{Path(p).parts[-2]}": sha(p)
                for p in sorted(glob.glob(str(root / "policies/*/s*/policy.npz")))}
    labels = {}
    for p in sorted(glob.glob(str(root / "select/s*.npz"))):
        z = np.load(p, allow_pickle=False)
        labels[Path(p).stem] = {"sha256": sha(p), "fields": sorted(z.files),
                                "rows": {k.split("/")[0]: int(len(z[k])) for k in z.files if k.endswith("/a")}}
    data = {Path(p).parent.name: {"subset_sha256": sha(p), "split_sha256": sha(Path(p).parent / "split.json")}
            for p in sorted(glob.glob(str(root / "data/s*/subset.npz")))}
    return {"record": {"path": Path(rec).name, "sha256": sha(rec)} if rec else None, "policies": policies,
            "relabels": labels, "data": data}


def probe_pusht_v22(root: Path) -> dict:
    units = {}
    for u in sorted(p.name for p in (root / "units").iterdir()):
        d = root / "units" / u
        scores = json.loads((d / "select-scores.json").read_text())
        step = min(scores["infonce"], key=lambda k: (scores["infonce"][k], int(k)))
        model = d / "train/checkpoints" / f"{int(step):06d}" / "pretrained_model"
        units[u] = {"select_scores_sha256": sha(d / "select-scores.json"), "selected_step": int(step),
                    "checkpoint": {f.name: sha(f) for f in sorted(model.iterdir()) if f.is_file()} if model.is_dir()
                    else None,
                    "checkpoint_steps": sorted(int(p.name) for p in (d / "train/checkpoints").iterdir()
                                               if p.name.isdigit())}
    return {"records": {f: sha(root / f) for f in ("root.json", "labels.json", "arms.json", "report.json")},
            "units": units, "pools": {"all": ccil_evidence(root / "labels/runs")}}


def probe_pusht_surface(root: Path) -> dict:
    man = json.loads((root / "manifest.json").read_text())
    stages = {s: {"sha256": sha(root / "stages" / f"{s}.json"), "manifest": h} for s, h in man["stages"].items()}
    train = json.loads((root / "stages/train.json").read_text())
    units = {}
    for k, v in train["units"].items():
        e = root / "store/policy" / v["entry"]
        files = json.loads((e / "manifest.json").read_text()) if e.is_dir() else None
        units[k] = {"entry": v["entry"], "selected_step": v["selected_step"],
                    "checkpoint": None if files is None else
                    {f: {"sha256": sha(e / f), "manifest": files[f]["sha256"]} for f in sorted(files)}}
    pools = {}
    pre = json.loads((root / "stages/preflight.json").read_text())
    for e in sorted((root / "store/pool").iterdir()):
        gen = sha(e / "generate-result.json")
        work = [w for w in sorted((root / "work").glob("pool-*")) if (w / "runs/generate-result.json").is_file()
                and sha(w / "runs/generate-result.json") == gen]
        labels = json.loads((e / "labels.json").read_text())
        r = next((k for k, v in pre["pools"].items() if v["episodes"] == labels["episodes"]), None)
        pools[r or e.name] = {"entry": e.name, "labels_npz_sha256": sha(e / "labels.npz"),
                              "generate_result_sha256": gen, "work": work[0].name if len(work) == 1 else None,
                              "ccil": ccil_evidence(work[0] / "runs") if len(work) == 1 else None}
    return {"protocol_sha256": man["protocol_sha256"], "stages": stages, "units": units, "pools": pools}


PROBES = {"pendulum-control": probe_pendulum_control, "pusht-v22": probe_pusht_v22,
          "pusht-surface": probe_pusht_surface}


def classify_ccil(ev: dict) -> tuple:
    if ev is None or ev.get("selection") is None:
        return "unverifiable", ["no CCIL selection file"]
    s, missing = ev["selection"], []
    if ev["certificates"] is None or s["certificates_sha256"] is None:
        missing.append("certificates (per-row classification)")
    if s["finite_error_rows"] == 0:
        missing.append("error (no finite per-row error)")
    if ev["candidates"] is None or ev["candidates"]["sha256"] != s["candidates_sha256"]:
        return "unverifiable", ["candidates file missing or not the selection's"]
    return ("full" if not missing else "partial"), missing


def combine(parts: list) -> dict:
    out, problems = {}, []
    for part in parts:
        name, body = part["name"], part["probe"]
        entry = {"kind": part["kind"], "host": part["host"], "root": part["root"]}
        if part["kind"] == "pendulum-control":
            committed = EV / f"pendulum-{part['study']}" / body["record"]["path"]
            rec = json.loads(committed.read_text())
            ok_record = sha(committed) == body["record"]["sha256"]
            bad = sorted(k for k, h in rec["policies"].items() if body["policies"].get(k) != h)
            entry.update({"record": {"committed": str(committed.relative_to(REPO)), "matches": ok_record},
                          "policies": {"count": len(body["policies"]), "mismatched_or_missing": bad,
                                       "sha256": body["policies"]},
                          "labels": {"class": "oracle" if ok_record and not bad else "unverifiable",
                                     "note": "demonstrator relabels at BC-visited states; no CCIL selection, weight, "
                                             "certificate or error exists for them",
                                     "relabels": body["relabels"], "data": body["data"]}})
            if not ok_record or bad:
                problems.append(f"{name}: record or policies differ from the committed record")
        elif part["kind"] == "pusht-v22":
            src = json.loads((EV / "pusht/labels.json").read_text())["sources"]["sha256"]
            want = {"labels.json": src["docs/studies/evidence/pusht-v2/run/labels.json"],
                    "root.json": src["docs/studies/evidence/pusht-v2/run/root.json"],
                    "arms.json": src["docs/studies/evidence/pusht-v2/run/arms.json"]}
            rec_ok = all(body["records"][f] == h for f, h in want.items())
            res = json.loads((EV / "pusht/results.json").read_text())["sources"]["sha256"]
            sel_bad = sorted(u for u, v in body["units"].items()
                             if res.get(f"docs/studies/evidence/pusht-v2/run/units/{u}/select-scores.json")
                             not in (None, v["select_scores_sha256"]) or v["checkpoint"] is None)
            cls, missing = classify_ccil(body["pools"]["all"])
            if not rec_ok:
                cls = "unverifiable"
                problems.append(f"{name}: run records differ from the archive's committed hashes")
            entry.update({"records": {"matches_committed": rec_ok, "sha256": body["records"]},
                          "units": {"count": len(body["units"]), "unverified": sel_bad, "detail": body["units"]},
                          "labels": {"all": {"class": cls, "missing": missing, "ccil": body["pools"]["all"]}}})
        else:
            stage = part["study"]
            rec = json.loads((EV / "pusht-surface" / f"{stage}.json").read_text())
            st_ok = all(v["sha256"] == v["manifest"] == rec["stage_sha256"][s] for s, v in body["stages"].items())
            bad = sorted(k for k, v in body["units"].items()
                         if v["checkpoint"] is None or v["entry"] != rec["units"][k]["train_entry"]
                         or any(f["sha256"] != f["manifest"] for f in v["checkpoint"].values()))
            labels = {}
            for r, p in body["pools"].items():
                cls, missing = classify_ccil(p["ccil"])
                labels[r] = {"class": cls if st_ok else "unverifiable", "missing": missing, **p}
            if not st_ok or bad:
                problems.append(f"{name}: stage hashes or policy entries differ from the committed record")
            entry.update({"stages_match_committed": st_ok, "protocol_sha256": body["protocol_sha256"],
                          "units": {"count": len(body["units"]), "unverified": bad, "detail": body["units"]},
                          "labels": labels})
        out[name] = entry
    return {"schema": "ilsb.reweight.inventory 1", "sources": out, "problems": problems}


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("probe")
    a.add_argument("--kind", choices=sorted(PROBES), required=True)
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--name", required=True)
    a.add_argument("--study", required=True, help="positive-control, capture-control, v22, stage1 or stage2")
    b = sub.add_parser("combine")
    b.add_argument("parts", type=Path, nargs="+")
    b.add_argument("--out", type=Path, default=EV / "reweight/inventory.json")
    args = p.parse_args()
    if args.cmd == "probe":
        root = args.root.resolve()
        print(json.dumps({"name": args.name, "kind": args.kind, "study": args.study, "host": socket.gethostname(),
                          "root": str(root), "probe": PROBES[args.kind](root)}, indent=1, sort_keys=True))
        return
    inv = combine([json.loads(f.read_text()) for f in args.parts])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(inv, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "problems": inv["problems"],
                      "classes": {n: {k: v["class"] for k, v in s["labels"].items()} if "class" not in s["labels"]
                                  else s["labels"]["class"] for n, s in inv["sources"].items()}}))
    sys.exit(1 if inv["problems"] else 0)


if __name__ == "__main__":
    main()
