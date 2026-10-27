"""End-to-end Mechanism Adjudication run on the Randi proximity benchmark -> verified immutable capsule.

    wormlab mechanism run [--run-id ID]   (or python scripts/mechanism_run_randi.py)

Refuses to run unless every DISCOVERY and CONFIRMATION animal in the frozen partition manifest has a sealed
payload (or a recorded extraction failure), so the confirmation set is fixed before any JUDGE access.
"""
import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wormlab.evidence.capsule import verify
from wormlab.mechanism import engine, nulls
from wormlab.mechanism.capsule_build import build
from wormlab.mechanism.randi import benchmark as B
from wormlab.mechanism.randi import partitions
from wormlab.mechanism.randi.provider import ANALYSES, RandiProvider

REPO = partitions.REPO


def extraction_complete(provider, manifest) -> dict:
    out = {}
    for part in ("DISCOVERY", "CONFIRMATION"):
        seal = provider.dirs[part] / "SEAL.jsonl"
        rows = [json.loads(l) for l in seal.read_text().splitlines()] if seal.exists() else []
        done = {r["sub"] for r in rows}
        out[part] = dict(expected=len(manifest["partitions"][part]), sealed=len({r["sub"] for r in rows if r["status"] in ("ok", "exists")}),
                         failed=sorted(r["sub"] for r in rows if r["status"] == "failed"),
                         missing=sorted(set(manifest["partitions"][part]) - done))
    return out


def extraction_complete_part(provider, manifest, part) -> dict:
    seal = provider.dirs[part] / "SEAL.jsonl"
    rows = [json.loads(l) for l in seal.read_text().splitlines()] if seal.exists() else []
    return dict(expected=len(manifest["partitions"][part]), sealed=len({r["sub"] for r in rows if r["status"] in ("ok", "exists")}),
                failed=sorted(r["sub"] for r in rows if r["status"] == "failed"), missing=sorted(set(manifest["partitions"][part]) - {r["sub"] for r in rows}))


def tamper_test(capsule_dir: Path) -> dict:
    with tempfile.TemporaryDirectory() as td:
        cp = Path(td) / "cap"
        shutil.copytree(capsule_dir, cp)
        blob = sorted((cp / "store").iterdir())[0]
        blob.chmod(0o644)
        b = bytearray(blob.read_bytes())
        b[len(b) // 2] ^= 0x01
        blob.write_bytes(bytes(b))
        rep = verify(cp)
        return dict(one_byte_tamper_detected=not bool(rep), tampered_blob=blob.name)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("randi_proximity_%Y%m%dT%H%M%SZ"))
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--generation", choices=["g0", "g1", "g2"], default="g0")
    ap.add_argument("--parent-run", default=None, help="parent run id (g0 run for g1; g1 run for g2)")
    ap.add_argument("--grandparent-run", default=None, help="g0 run id (required for g2)")
    a = ap.parse_args(argv)
    manifest = partitions.load()
    provider = RandiProvider(manifest)
    status = extraction_complete(provider, manifest)
    if not a.allow_incomplete and any(v["missing"] for v in status.values()):
        raise SystemExit(f"extraction incomplete: {status}")
    run_dir = partitions.BENCH_DIR / "runs" / a.run_id
    parent_capsule = None
    if a.generation == "g0":
        run = engine.MechanismRun(run_dir, B.hypotheses(), B.tests(), B.measurement_model(), manifest, B.historical_contamination(), provider,
                                  engine.RunConfig(max_judge_steps=20))
        payload_dirs = [provider.dirs["DISCOVERY"], provider.dirs["CONFIRMATION"]]
    elif a.generation == "g2":
        if not (a.parent_run and a.grandparent_run):
            raise SystemExit("--parent-run (g1) and --grandparent-run (g0) are required for g2")
        from .provider_g2 import ADDENDUM, RandiVoxelProvider
        g1d, g0d = partitions.BENCH_DIR / "runs" / a.parent_run, partitions.BENCH_DIR / "runs" / a.grandparent_run
        parent_capsule = REPO / "results" / "_capsules" / f"mechanism_{a.parent_run}"
        interp = REPO / "results" / "_capsules" / f"mechanism_{a.parent_run}__interpretation_policy1.2"
        for c in (parent_capsule, interp):
            if not bool(verify(c)):
                raise SystemExit(f"{c} does not verify")
        addendum = json.loads(ADDENDUM.read_text())
        g0_out = json.loads((g0d / "outputs/outcomes.json").read_text())
        g1_out = json.loads((g1d / "outputs/outcomes.json").read_text())
        prior = [dict(o, details=dict(o["details"], imported_from_generation=gen)) for gen, outs in (("g0", g0_out), ("g1", g1_out))
                 for o in outs if o["stage"] != "HISTORICAL" and o["status"] != "NOT_RUN_ACCESS_DENIED"]
        motivating = ["g1_capsule:" + (parent_capsule / "ROOT.sha256").read_text().strip()[:16],
                      "g1_interpretation_policy1.2:" + (interp / "ROOT.sha256").read_text().strip()[:16]]
        provider = RandiVoxelProvider(manifest, addendum, prior)
        run = engine.MechanismRun(run_dir, B.hypotheses_g2(motivating), B.tests_g2(), B.measurement_model_g1(), manifest,
                                  B.historical_contamination_g2(g0_out, g1_out), provider,
                                  engine.RunConfig(max_judge_steps=4, judge_partitions=("RESERVE", "CONFIRMATION"), policy="1.2",
                                                   release_reserve_to_all=True, phenomenon_test="T0_proximity_effect"))
        payload_dirs = [provider.dirs["RESERVE"], provider.dirs["CONFIRMATION"]]
        status["VOXEL"] = {p_: dict(sealed=len(provider.seals(p_)), expected=len(manifest["partitions"][p_])) for p_ in ("RESERVE", "CONFIRMATION")}
        if not a.allow_incomplete and any(v["sealed"] < v["expected"] for v in status["VOXEL"].values()):
            raise SystemExit(f"voxel extraction incomplete: {status['VOXEL']}")
    else:
        if not a.parent_run:
            raise SystemExit("--parent-run is required for g1")
        prun = partitions.BENCH_DIR / "runs" / a.parent_run
        parent_capsule = REPO / "results" / "_capsules" / f"mechanism_{a.parent_run}"
        if not bool(verify(parent_capsule)):
            raise SystemExit("parent capsule does not verify")
        g0_out = json.loads((prun / "outputs/outcomes.json").read_text())
        scout = partitions.BENCH_DIR / "scout_g1" / "scout_g1_diagnostics.json"
        import hashlib
        motivating = [o["outcome_id"] for o in g0_out if o["test_id"] in ("T11_baseline_covariance", "T5_control_channel") and o["partition"] == "CONFIRMATION"]
        motivating.append("scout_g1_diagnostics:" + hashlib.sha256(scout.read_bytes()).hexdigest()[:16])
        run = engine.MechanismRun(run_dir, B.hypotheses_g1(motivating), B.tests_g1(), B.measurement_model_g1(), manifest,
                                  B.historical_contamination_g1(g0_out), provider,
                                  engine.RunConfig(max_judge_steps=20, judge_partition="RESERVE", policy="1.1", release_reserve_to_all=True))
        payload_dirs = [provider.dirs["DISCOVERY"], provider.dirs["RESERVE"]]
        status["RESERVE"] = extraction_complete_part(provider, manifest, "RESERVE")
        if not a.allow_incomplete and status["RESERVE"]["missing"]:
            raise SystemExit(f"RESERVE extraction incomplete: {status['RESERVE']}")

    def after_scout(r):
        disc = [o for o in r.outcomes if o.partition == "DISCOVERY"]
        tab = provider._cache.get("DISCOVERY")
        if tab is None or "target_resp" not in tab:
            r.log("SYNTHETIC_NULLS_SKIPPED", reason="no discovery event table for this generation's data stream")
            return
        ev = {(a_, e) for a_, e, k in zip(tab["animal"], tab["event"], tab["kind"]) if k == "stim"}
        tr = {(a_, e): t for a_, e, t, k in zip(tab["animal"], tab["event"], tab["target_resp"], tab["kind"]) if k == "stim"}
        p_target = float(np.mean([tr[k] for k in ev]))
        r.extra_outputs["synthetic_nulls.json"] = nulls.compare(disc, [t for t in r.tests if t.test_id in nulls.contrasts.REGISTRY],
                                                                 n_animals=len(provider.units("DISCOVERY")), p_target=p_target)
        r.log("SYNTHETIC_NULLS", ranking=r.extra_outputs["synthetic_nulls.json"]["ranking"])

    run.after_scout = after_scout
    result = run.run()
    run.extra_outputs["extraction_status.json"] = status
    run.extra_outputs["unc31_contamination_ledger.json"] = B.UNC31_LEDGER
    capsule_dir = REPO / "results" / "_capsules" / f"mechanism_{a.run_id}"
    extra = [partitions.BENCH_DIR / partitions.MANIFEST_NAME, partitions.BENCH_DIR / "g2_addendum_manifest.json"] + [provider.dirs[p_] / "SEAL.jsonl" for p_ in provider.dirs] + [
             partitions.BENCH_DIR / "scout_g1" / "scout_g1_diagnostics.json", partitions.BENCH_DIR / "scout_g1" / "scout_ledger.jsonl",
             ANALYSES / "reextract/out/haze_vs_discrete.json", ANALYSES / "out/t6b_within_pair.json", ANALYSES / "reextract/LINKAGE_NOTES.md"]
    cap = build(run, capsule_dir, capsule_id=f"wormlab-mechanism-{a.run_id}", extra_inputs=[p for p in extra if p.exists()],
                payload_dirs=payload_dirs, parent=parent_capsule,
                description="Mechanism Adjudication Engine: Randi 2023 proximity benchmark (WT, DANDI:001075 raw volumes)")
    tam = tamper_test(capsule_dir)
    summary = dict(run_id=a.run_id, verdict=result["verdict"]["verdict"], surviving=result["verdict"]["surviving"],
                   capsule=str(capsule_dir), capsule_root=cap["root"], capsule_verified=cap["verified"], tamper=tam)
    (run_dir / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
