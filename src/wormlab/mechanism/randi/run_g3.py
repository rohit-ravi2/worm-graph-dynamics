"""g3 run (Deep Mechanism Campaign): signal-provenance / unmixing families judged on CONFIRMATION then RESERVE under
policy 1.3 with every frozen test run on each partition (no adaptive stopping).

    python -m wormlab.mechanism.randi.run_g3 --g0 RUN --g1 RUN --g2 RUN [--run-id ID]

Refuses to start unless: the g3 preregistration file exists and its recorded hashes match the current code, test and
hypothesis specs; every CONFIRMATION/RESERVE animal has sealed provenance and event_crops payloads; parent capsules verify.
Exploratory descriptives on CONFIRMATION/RESERVE are computed AFTER the adjudication from the provider's in-memory tables
(no additional payload reads) and written as clearly separated extra outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wormlab.evidence.capsule import verify
from wormlab.mechanism import engine, provenance_contrasts, provenance_descriptives as PD, provenance_verdict as PV, signal_provenance as SP, unmixing as U
from wormlab.mechanism.capsule_build import build
from wormlab.mechanism.randi import benchmark_g3 as B3, partitions
from wormlab.mechanism.randi.provider_g3 import CROPS_ADDENDUM, PROV_ADDENDUM, RandiProvenanceProvider, _base, _seals
from wormlab.mechanism.randi.run import tamper_test

REPO = partitions.REPO
PREREG_ORIGINAL = REPO / "results/mechanism/deep_campaign/g3_preregistration.json"
PREREG_CHAIN = [PREREG_ORIGINAL, REPO / "results/mechanism/deep_campaign/g3_preregistration_a1.json",
                REPO / "results/mechanism/deep_campaign/g3_preregistration_a2.json"]     # amendments (all before any sealed access)
PREREG = PREREG_CHAIN[-1]
PHENOMENON_FAMILY_TESTS = ("T17_unmixed_proximity", "T13_optical_copy_subtracted_proximity")


def prereg_hashes() -> dict:
    mods = ["src/wormlab/mechanism/signal_provenance.py", "src/wormlab/mechanism/unmixing.py", "src/wormlab/mechanism/provenance_contrasts.py",
            "src/wormlab/mechanism/provenance_verdict.py", "src/wormlab/mechanism/randi/provider_g3.py", "src/wormlab/mechanism/randi/benchmark_g3.py",
            "src/wormlab/mechanism/adjudicator.py", "src/wormlab/mechanism/engine.py", "src/wormlab/mechanism/evidence_graph.py",
            "src/wormlab/mechanism/spec.py", "src/wormlab/mechanism/randi/run_g3.py", "src/wormlab/mechanism/waveform_provenance.py",
            "src/wormlab/mechanism/provenance_descriptives.py", "src/wormlab/mechanism/roi_extract.py"]
    return {m: hashlib.sha256((REPO / m).read_bytes()).hexdigest() for m in mods}


def check_prereg(hyps, tests) -> dict:
    if not PREREG.exists():
        raise SystemExit("g3 preregistration missing: run scripts/deep_campaign/freeze_g3_prereg.py first")
    pr = json.loads(PREREG.read_text())
    cur = prereg_hashes()
    bad = {m: (pr["code_sha256"].get(m), h) for m, h in cur.items() if pr["code_sha256"].get(m) != h}
    if bad:
        raise SystemExit(f"code changed since preregistration: {sorted(bad)}")
    th = {t.test_id: t.spec_hash for t in tests}
    if th != pr["tests"]:
        raise SystemExit("test specs changed since preregistration")
    hh = {h.hypothesis_id: h.spec_hash for h in hyps}
    if hh != pr["hypotheses"]:
        raise SystemExit("hypothesis specs changed since preregistration")
    return pr


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("randi_proximity_g3_%Y%m%dT%H%M%SZ"))
    ap.add_argument("--g0", required=True); ap.add_argument("--g1", required=True); ap.add_argument("--g2", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    manifest = partitions.load()
    runs = partitions.BENCH_DIR / "runs"
    outs = {g: json.loads((runs / r / "outputs/outcomes.json").read_text()) for g, r in (("g0", a.g0), ("g1", a.g1), ("g2", a.g2))}
    parent = REPO / "results/_capsules" / f"mechanism_{a.g2}"
    interp = REPO / "results/_capsules" / f"mechanism_{a.g2}__interpretation__policy1.2__catalogue1.1"
    for c in (parent, interp):
        if not bool(verify(c)):
            raise SystemExit(f"{c} does not verify")
    status = {}
    for part in ("DISCOVERY", "CONFIRMATION", "RESERVE"):
        b = _base(part)
        status[part] = dict(expected=len(manifest["partitions"][part]), provenance=len(_seals(b / "provenance")), event_crops=len(_seals(b / "event_crops")))
        if min(status[part]["provenance"], status[part]["event_crops"]) < status[part]["expected"]:
            failed = [json.loads(l) for d in ("provenance", "event_crops") if (b / d / "SEAL.jsonl").exists() for l in (b / d / "SEAL.jsonl").read_text().splitlines()]
            if not any(r["status"] == "failed" for r in failed):
                raise SystemExit(f"extraction incomplete: {status}")
    motivating = ["g2_capsule:" + (parent / "ROOT.sha256").read_text().strip()[:16],
                  "synthetic_battery_v2:" + hashlib.sha256((REPO / "results/mechanism/deep_campaign/synthetic_battery/battery_report_v2.json").read_bytes()).hexdigest()[:16],
                  "literature:BIOPHYSICAL_PARAMETERS.json:" + hashlib.sha256((REPO / "docs/mechanism_engine/deep_campaign/BIOPHYSICAL_PARAMETERS.json").read_bytes()).hexdigest()[:16]]
    hyps = B3.hypotheses_g3(motivating)
    tests = B3.tests_g3()
    pr = check_prereg(hyps, tests)
    provenance_contrasts.register()
    prior = [dict(o, details=dict(o["details"], imported_from_generation=g)) for g in ("g0", "g1", "g2") for o in outs[g]
             if o["stage"] != "HISTORICAL" and o["status"] != "NOT_RUN_ACCESS_DENIED"]
    provider = RandiProvenanceProvider(manifest, prior, workers=a.workers)
    cfg = engine.RunConfig(max_judge_steps=len(tests) + 2, judge_partitions=("CONFIRMATION", "RESERVE"), policy=pr["run_config"]["policy"], release_reserve_to_all=True,
                           judge_all=True, phenomenon_test="T0_proximity_effect", invalid_predictions=tuple(tuple(x) for x in pr["invalid_predictions"]),
                           phenomenon_family_tests=PHENOMENON_FAMILY_TESTS, mixture_share_tests=("T17_unmixed_proximity",))
    run_dir = partitions.BENCH_DIR / "runs" / a.run_id
    run = engine.MechanismRun(run_dir, hyps, tests, B3.measurement_model_g3(), manifest, B3.historical_contamination_g3(outs["g0"], outs["g1"], outs["g2"]),
                              provider, cfg)
    result = run.run()
    # ---- exploratory descriptives on the partitions already read (in-memory tables; no new payload reads) ----
    expl = {}
    for part, c in provider._cache.items():
        fam = {tid: fn(c, seed=cfg.seed) for tid, fn in provenance_contrasts.G3_REGISTRY.items()}
        fam["T15_kappa_profile_calibration"] = SP.t15_roi_profile(c["rows"], cfg.seed)
        expl[part] = dict(grade="EXPLORATORY (post-adjudication descriptives)", families=fam, calls={k: PV.call(k, v) for k, v in fam.items() if k in PV.MIN_EFFECT},
                          classification=PV.classify(fam, c["halo"]), halo=c["halo"], descriptives=PD.all_descriptives(c), measurement=c["measurement"])
        run.journal.append(action="EXPLORATORY_DESCRIPTIVES", partition=part, note="computed after adjudication from in-memory tables")
    run.extra_outputs["exploratory_descriptives.json"] = json.loads(json.dumps(expl, default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o))))
    run.extra_outputs["extraction_status.json"] = status
    run.extra_outputs["g3_preregistration.json"] = pr
    capsule_dir = REPO / "results" / "_capsules" / f"mechanism_{a.run_id}"
    extra = [partitions.BENCH_DIR / partitions.MANIFEST_NAME, PROV_ADDENDUM, CROPS_ADDENDUM, *PREREG_CHAIN,
             REPO / "results/mechanism/deep_campaign/prediction_audit_v1.json",
             REPO / "results/mechanism/deep_campaign/prediction_audit.json", REPO / "results/mechanism/deep_campaign/synthetic_battery/battery_report_v2.json",
             REPO / "results/mechanism/deep_campaign/synthetic_battery/v3/battery_report_v3.json", REPO / "results/mechanism/deep_campaign/contamination_map.json"]
    extra += [_base(p) / d / "SEAL.jsonl" for p in ("DISCOVERY", "CONFIRMATION", "RESERVE") for d in ("provenance", "event_crops")]
    extra += sorted((REPO / "results/mechanism/deep_campaign/scout_g3").glob("scout_*.json")) + [REPO / "results/mechanism/deep_campaign/scout_g3/scout_ledger.jsonl"]
    cap = build(run, capsule_dir, capsule_id=f"wormlab-mechanism-{a.run_id}", extra_inputs=[p for p in extra if p.exists()], payload_dirs=[], parent=parent,
                description="Mechanism Adjudication Engine g3: signal provenance + evoked unmixing families (Deep Mechanism Campaign), policy per preregistration amendment")
    tam = tamper_test(capsule_dir)
    summary = dict(run_id=a.run_id, verdict=result["verdict"]["verdict"], surviving=result["verdict"]["surviving"], capsule=str(capsule_dir),
                   capsule_root=cap["root"], capsule_verified=cap["verified"], tamper=tam,
                   calls={o.partition + ":" + o.test_id: (o.call, o.estimate, o.ci) for o in run.outcomes if o.stage == "CONFIRMATION"})
    (run_dir / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=1, default=str))
    (partitions.BENCH_DIR / "logs" / "current_run_id_g3").write_text(a.run_id)
    print(json.dumps(summary, indent=1, default=str))
    return summary


if __name__ == "__main__":
    main()
