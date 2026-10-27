"""`wormlab mechanism ...` — narrow CLI for the Mechanism Adjudication Engine."""
from __future__ import annotations

import json
from pathlib import Path


def _emit(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def _bench(name: str):
    if name != "randi":
        raise SystemExit(f"unknown benchmark {name!r} (available: randi)")
    from .randi import benchmark as B
    return B


def _load_hypotheses(run_dir: Path):
    from .spec import HypothesisSpec, Prediction
    out = []
    for d in json.loads((run_dir / "outputs/hypotheses.json").read_text()):
        d = {k: v for k, v in d.items() if k != "spec_hash"}
        d["predictions"] = [Prediction(**p) for p in d["predictions"]]
        out.append(HypothesisSpec(**d))
    return out


def _load_outcomes(run_dir: Path):
    from .spec import TestOutcome
    return [TestOutcome(**o) for o in json.loads((run_dir / "outputs/outcomes.json").read_text())]


def _rebuild_graph(run_dir: Path, outcomes, hypotheses):
    from .evidence_graph import EvidenceGraph, relation_for
    g = EvidenceGraph()
    stored = json.loads((run_dir / "outputs/evidence_graph.json").read_text())
    for o in outcomes:
        node = stored["nodes"].get(o.outcome_id, {})
        g.add_outcome(o, data_stream=node.get("data_stream", "unknown"))
        for h in hypotheses:
            rel = relation_for(h.prediction_for(o.test_id), o.call) if o.status == "DECIDED" else None
            if rel:
                g.relate_hypothesis(o.outcome_id, h.hypothesis_id, rel, kind=h.prediction_for(o.test_id).kind, grade=o.contamination_grade)
    return g


def cmd_hypotheses_validate(a) -> int:
    B = _bench(a.benchmark)
    rows = []
    for h in B.hypotheses():
        h.validate()
        rows.append({"hypothesis_id": h.hypothesis_id, "class": h.mechanism_class, "spec_hash": h.spec_hash,
                     "falsifying_tests": [p.test_id for p in h._preds() if p.counts_against_when()], "discriminating_tests": h.discriminating_tests})
    _emit({"command": "mechanism hypotheses validate", "ok": True, "hypotheses": rows})
    return 0


def cmd_compile(a) -> int:
    from .spec import SpecError, compile_test_against_hypotheses
    B = _bench(a.benchmark)
    H = B.hypotheses()
    rows, ok = [], True
    for t in B.tests():
        try:
            c = compile_test_against_hypotheses(t, H)
            rows.append({"test_id": t.test_id, "spec_hash": t.spec_hash, "disagreeing_pairs": len(c["disagreeing_pairs"]), "predictions": c["predictions"]})
        except SpecError as e:
            ok = False
            rows.append({"test_id": t.test_id, "rejected": str(e)})
    m = B.measurement_model()
    m.validate()
    _emit({"command": "mechanism compile", "ok": ok, "measurement_model_hash": m.spec_hash, "tests": rows})
    return 0 if ok else 4


def cmd_rank_tests(a) -> int:
    from . import discriminator
    from .firewall import AccessBroker, AccessLedger, FreezeRegistry
    from .randi import partitions
    from .randi.provider import RandiProvider
    B = _bench(a.benchmark)
    m = partitions.load()
    prov = RandiProvider(m)
    broker = AccessBroker(m, AccessLedger(), FreezeRegistry(AccessLedger()), B.historical_contamination())
    H, T = B.hypotheses(), B.tests()
    units = prov.units(a.partition)
    avail = {t.test_id: {"partition": a.partition, "grade": broker.grade(a.partition, t.statistic_family, [h.hypothesis_id for h in H]),
                         "expected_units": len(units)} for t in T} if units else {}
    rep = discriminator.rank_tests(H, {}, T, availability=avail, measurement=B.measurement_model())
    _emit({"command": "mechanism rank-tests", "partition": a.partition, "units": len(units), **rep})
    return 0


def cmd_run(a) -> int:
    from .randi.run import main
    argv = ["--run-id", a.run_id] if a.run_id else []
    if a.allow_incomplete:
        argv.append("--allow-incomplete")
    s = main(argv)
    return 0 if s["capsule_verified"] else 5


def cmd_status(a) -> int:
    from .firewall import AccessLedger
    rd = Path(a.run_dir)
    j, l = AccessLedger(rd / "journal.jsonl"), AccessLedger(rd / "access_ledger.jsonl")
    verdict = json.loads((rd / "outputs/verdict.json").read_text()) if (rd / "outputs/verdict.json").exists() else None
    _emit({"command": "mechanism status", "journal_entries": len(j.entries), "journal_head": j.head, "ledger_entries": len(l.entries),
           "ledger_head": l.head, "denied_requests": sum(1 for e in l.entries if e.get("granted") is False),
           "last_events": [e["action"] for e in j.entries[-5:]], "verdict": verdict and verdict["verdict"]})
    return 0


def cmd_evidence(a) -> int:
    rd = Path(a.run_dir)
    fam = json.loads((rd / "outputs/evidence_families.json").read_text())
    _emit({"command": "mechanism evidence", "n_outcomes": len(fam["families"]), "n_families": len(set(fam["families"].values())),
           "support_summary": fam["support_summary"]})
    return 0


def cmd_adjudicate(a) -> int:
    from .adjudicator import adjudicate
    rd = Path(a.run_dir)
    H, O = _load_hypotheses(rd), _load_outcomes(rd)
    g = _rebuild_graph(rd, O, H)
    stored = json.loads((rd / "outputs/verdict.json").read_text())
    rep = {"status": "NO_AVAILABLE_TEST_HAS_DISCRIMINATING_POWER", "candidates": []}
    mm = json.loads((rd / "outputs/measurement_model.json").read_text())
    tests = json.loads((rd / "outputs/tests.json").read_text())
    blocked = {t["test_id"] for t in tests
               if any(mm["fields"].get(f, {}).get("status", "UNKNOWN") == "UNKNOWN" for f in t["required_measurement_fields"])}
    res = adjudicate(H, O, g, phenomenon_test="T0_proximity_effect", discrimination_report=rep, measurement_blocked_tests=blocked,
                     policy=stored.get("policy_version", "1.0"))
    same = res["verdict"] == stored["verdict"] and {h: x["state"] for h, x in res["hypotheses"].items()} == {h: x["state"] for h, x in stored["hypotheses"].items()}
    _emit({"command": "mechanism adjudicate", "replayed_verdict": res["verdict"], "stored_verdict": stored["verdict"], "deterministic_match": same})
    return 0 if same else 6


def cmd_propose(a) -> int:
    rd = Path(a.run_dir)
    _emit({"command": "mechanism propose-experiment", **json.loads((rd / "outputs/experiment_proposal.json").read_text())})
    return 0


def cmd_verify(a) -> int:
    from ..evidence.capsule import EvidenceCapsule, verify
    from .firewall import AccessLedger
    cdir = Path(a.capsule)
    rep = verify(cdir)
    cap = EvidenceCapsule(cdir)
    chains = {}
    for art in cap.artifacts:
        if art["original_path"] in ("access_ledger.jsonl", "journal.jsonl"):
            led = AccessLedger()
            led.entries = [json.loads(l) for l in cap.blob_path(art).read_text().splitlines()]
            try:
                led.verify()
                chains[art["original_path"]] = "ok"
            except ValueError as e:
                chains[art["original_path"]] = str(e)
    ok = bool(rep) and all(v == "ok" for v in chains.values())
    _emit({"command": "mechanism verify", "capsule_ok": bool(rep), "root": cap.root_hash, "hash_chains": chains, "ok": ok})
    return 0 if ok else 7


def add_parser(sub) -> None:
    mech = sub.add_parser("mechanism", help="mechanism adjudication engine")
    ms = mech.add_subparsers(dest="cmd", required=True)
    hy = ms.add_parser("hypotheses", help="hypothesis specs")
    hys = hy.add_subparsers(dest="subcmd", required=True)
    hv = hys.add_parser("validate", help="validate benchmark hypotheses")
    hv.add_argument("--benchmark", default="randi")
    hv.set_defaults(func=cmd_hypotheses_validate)
    c = ms.add_parser("compile", help="compile tests against hypotheses (rejects non-discriminating tests)")
    c.add_argument("--benchmark", default="randi")
    c.set_defaults(func=cmd_compile)
    r = ms.add_parser("rank-tests", help="discriminator ranking from untested state (reads no outcome data)")
    r.add_argument("--benchmark", default="randi")
    r.add_argument("--partition", default="DISCOVERY")
    r.set_defaults(func=cmd_rank_tests)
    ru = ms.add_parser("run", help="end-to-end run -> verified capsule")
    ru.add_argument("--run-id", default=None)
    ru.add_argument("--allow-incomplete", action="store_true")
    ru.set_defaults(func=cmd_run)
    for name, fn, h in (("status", cmd_status, "journal/ledger status of a run"), ("evidence", cmd_evidence, "evidence families + support"),
                        ("adjudicate", cmd_adjudicate, "deterministically replay adjudication from stored outputs"),
                        ("propose-experiment", cmd_propose, "experiment proposal of a run")):
        p = ms.add_parser(name, help=h)
        p.add_argument("run_dir")
        p.set_defaults(func=fn)
    v = ms.add_parser("verify", help="verify a mechanism capsule + its hash-chained ledgers")
    v.add_argument("capsule")
    v.set_defaults(func=cmd_verify)
