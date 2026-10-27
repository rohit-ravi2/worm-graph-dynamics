"""Mechanism adjudication run loop.

    AUDITOR   verify partition manifest + seals; load historical contamination
    COMPILER  validate hypotheses/tests, compile discrimination, FREEZE (ledger)
    SCOUT     discriminator-ranked tests on DISCOVERY (+ imported historical outcomes, synthetic nulls)
    JUDGE     repeatedly: rank remaining tests for CONFIRMATION, take the best permitted one, single look,
              update evidence graph, adjudicate; stop at a terminal verdict or when no test discriminates
    DESIGN    experiment proposal if hypotheses remain unresolved
    CAPSULE   (caller) freeze everything into an EvidenceCapsule

Termination is scientific: the loop ends when the adjudicator returns a terminal verdict, which it only does
when no permitted, discriminating analysis remains (or a mechanism is supported).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import adjudicator, contrasts, design, discriminator
from .evidence_graph import EvidenceGraph, relation_for
from .firewall import AccessBroker, AccessLedger, ContaminationViolation, FreezeRegistry, HistoricalContamination, Role
from .measurement import UNKNOWN, MeasurementModelSpec
from .spec import (HypothesisSpec, MechanismTestSpec, OutcomeStatus, Stage, TestOutcome, canonical_sha,
                   compile_test_against_hypotheses, mixture_of)


class DataProvider:
    """Interface. Implementations must read payloads only inside methods that receive a broker grant."""

    def units(self, partition: str) -> list[str]:
        raise NotImplementedError

    def table(self, partition: str, grant: dict) -> dict:
        raise NotImplementedError

    def data_stream(self) -> str:
        raise NotImplementedError

    def imported_outcomes(self) -> list[TestOutcome]:
        return []


@dataclass
class RunConfig:
    phenomenon_test: str | None = "T0_proximity_effect"
    synthetic_is_confirmation: bool = False
    max_judge_steps: int = 20
    scout_partition: str = "DISCOVERY"
    judge_partition: str = "CONFIRMATION"
    judge_partitions: tuple = ()             # if set, judge each partition in turn (independent replications)
    seed: int = 20260916
    allow_mixtures: bool = True
    preregister_mixtures: bool = True
    policy: str = "1.0"
    release_reserve_to_all: bool = False     # AUDITOR releases RESERVE to every frozen hypothesis (new generation judged on RESERVE)
    judge_all: bool = False                  # run EVERY frozen test on each judge partition in fixed order (no adaptive stopping)
    invalid_predictions: tuple = ()          # (family, test_id) NECESSARY predictions failed by the synthetic prediction audit (policy 1.3)
    phenomenon_family_tests: tuple = ()      # tests that re-measure the phenomenon (policy 1.3 explanatory relevance)
    mixture_share_tests: tuple = ()          # tests whose explained_fraction comes from the FULL contamination model (policy 1.3 overlay)


@dataclass
class MechanismRun:
    run_dir: Path
    hypotheses: list
    tests: list
    measurement: MeasurementModelSpec
    partition_manifest: dict
    historical: list
    provider: DataProvider
    config: RunConfig = field(default_factory=RunConfig)

    def __post_init__(self):
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = AccessLedger(self.run_dir / "access_ledger.jsonl")
        self.journal = AccessLedger(self.run_dir / "journal.jsonl")
        self.registry = FreezeRegistry(self.ledger)
        self.broker = AccessBroker(self.partition_manifest, self.ledger, self.registry, self.historical)
        self.graph = EvidenceGraph()
        self.outcomes: list[TestOutcome] = []
        self.discrimination_reports: list[dict] = []
        self.adjudications: list[dict] = []
        self.deviations: list[dict] = []
        self.compiled: list[dict] = []
        self.resource = {"started": time.time(), "steps": []}
        self._tests = {t.test_id: t for t in self.tests}
        self.after_scout = None          # optional callable(run) executed after SCOUT adjudication (e.g. synthetic nulls)
        self.extra_outputs: dict = {}    # name -> JSON-serialisable object written with the outputs

    # -------------------------------------------------------------------------------------------------
    def log(self, event: str, **meta):
        self.journal.append(action=event, **meta)

    def _states(self) -> dict:
        if not self.adjudications:
            return {}
        return {h: a["state"] for h, a in self.adjudications[-1]["hypotheses"].items()}

    def _usable(self, t: MechanismTestSpec) -> tuple[bool, str]:
        for f in t.required_measurement_fields:
            if self.measurement.get(f).status == UNKNOWN:
                return False, f"measurement field {f} UNKNOWN"
        return True, ""

    def _availability(self, partition: str) -> dict:
        out = {}
        units = self.provider.units(partition)
        run = {(o.partition, o.feature_family) for o in self.outcomes}
        for t in self.tests:
            ok, _ = self._usable(t)
            if not ok or not units or t.test_id not in contrasts.REGISTRY:   # no implementation for this stream -> unavailable
                continue
            grade = self.broker.grade(partition, t.statistic_family, [h.hypothesis_id for h in self.hypotheses])
            out[t.test_id] = {"partition": partition, "grade": grade, "expected_units": len(units)}
        return out

    def _record(self, o: TestOutcome):
        self.outcomes.append(o)
        oid = self.graph.add_outcome(o, data_stream=o.details.get("data_stream") or (self.provider.data_stream() if o.stage != Stage.HISTORICAL else "historical"))
        for h in self.hypotheses:
            rel = relation_for(h.prediction_for(o.test_id), o.call) if o.status == OutcomeStatus.DECIDED else None
            if rel:
                self.graph.relate_hypothesis(oid, h.hypothesis_id, rel, kind=h.prediction_for(o.test_id).kind, grade=o.contamination_grade)

    def _adjudicate(self, report: dict | None, stage: str):
        blocked = {t.test_id for t in self.tests if not self._usable(t)[0]}
        res = adjudicator.adjudicate(self.hypotheses, self.outcomes, self.graph, phenomenon_test=self.config.phenomenon_test,
                                     discrimination_report=report, measurement_blocked_tests=blocked,
                                     synthetic_is_confirmation=self.config.synthetic_is_confirmation, allow_mixtures=self.config.allow_mixtures,
                                     policy=self.config.policy,
                                     **({"invalid_predictions": frozenset(tuple(x) for x in self.config.invalid_predictions),
                                         "phenomenon_family_tests": tuple(self.config.phenomenon_family_tests),
                                         "mixture_share_tests": tuple(self.config.mixture_share_tests)} if self.config.policy in ("1.3", "1.4") else {}))
        res["stage"] = stage
        res["n_outcomes"] = len(self.outcomes)
        res["evidence_support_summary"] = self.graph.support_summary()
        self.adjudications.append(res)
        self.log("ADJUDICATE", stage=stage, verdict=res["verdict"])
        return res

    def _run_test(self, t: MechanismTestSpec, partition: str, role: Role, stage: str) -> TestOutcome:
        hids = [h.hypothesis_id for h in self.hypotheses]
        t0 = time.time()
        try:
            grant = self.broker.request(role=role, actor=f"{role.value.lower()}:{t.test_id}", partition=partition, test=t, hypothesis_ids=hids)
        except ContaminationViolation as e:
            self.deviations.append({"test_id": t.test_id, "partition": partition, "reason": str(e)})
            o = TestOutcome(t.test_id, t.spec_hash, stage, partition, OutcomeStatus.NOT_RUN_ACCESS_DENIED, "INCONCLUSIVE", None, None,
                            t.min_effect, 0, 0, [], t.statistic_family, "CONTAMINATED", details={"denied": str(e)}).finalize()
            self._record(o)
            return o
        tab = self.provider.table(partition, grant)
        fn = contrasts.REGISTRY.get(t.test_id)
        if fn is None or not len(tab.get("animal", [])):
            o = TestOutcome(t.test_id, t.spec_hash, stage, partition, OutcomeStatus.NOT_RUN_DATA_MISSING, "INCONCLUSIVE", None, None,
                            t.min_effect, 0, 0, [], t.statistic_family, grant.get("grade", "UNTOUCHED"),
                            details={"reason": "no contrast implementation for this data stream" if fn is None else "empty table"}).finalize()
        else:
            raw = fn(tab, seed=self.config.seed)
            units = sorted(set(tab["animal"].tolist()))
            o = contrasts.to_outcome(t, raw, stage=stage, partition=partition, units=units, grade=grant.get("grade", "UNTOUCHED"),
                                     resolution_s=raw.get("resolution_s"))
        self._record(o)
        self.resource["steps"].append({"test_id": t.test_id, "partition": partition, "seconds": round(time.time() - t0, 2)})
        self.log("TEST", test_id=t.test_id, partition=partition, stage=stage, status=o.status, call=o.call, outcome_id=o.outcome_id)
        return o

    # -------------------------------------------------------------------------------------------------
    def run(self) -> dict:
        self.log("START", partition_manifest_sha256=self.partition_manifest.get("manifest_sha256"), measurement_sha=self.measurement.spec_hash)
        # COMPILER
        self.measurement.validate()
        for h in self.hypotheses:
            h.validate()
        if self.config.preregister_mixtures:
            singles = [h for h in self.hypotheses if h.mechanism_class != "MIXTURE_MODEL"]
            for i, a in enumerate(singles):
                for b in singles[i + 1:]:
                    gen = max(a.generation, b.generation)
                    m = mixture_of(f"H8[{a.family}+{b.family}].g{gen}", [a, b], claim=f"pre-registered mixture of {a.hypothesis_id} and {b.hypothesis_id}")
                    if gen > 0:
                        m.parent_hypothesis_ids = [f"H8[{a.family}+{b.family}].g{gen - 1}"]
                        m.created_from_evidence_ids = sorted(set(a.created_from_evidence_ids) | set(b.created_from_evidence_ids))
                    m.validate()
                    self.hypotheses.append(m)
        for t in self.tests:
            try:
                self.compiled.append(compile_test_against_hypotheses(t, self.hypotheses))
            except Exception as e:  # a non-discriminating test is removed, not silently kept (phenomenon gate exempt)
                if t.test_id == self.config.phenomenon_test:
                    t.validate()
                    self.compiled.append({"test_id": t.test_id, "role": "phenomenon_gate", "note": str(e), "spec_hash": t.spec_hash})
                else:
                    self.deviations.append({"test_id": t.test_id, "reason": f"rejected by compiler: {e}"})
        rejected = {d["test_id"] for d in self.deviations}
        self.tests = [t for t in self.tests if t.test_id not in rejected]
        for h in self.hypotheses:
            self.registry.freeze_hypothesis(h, by="compiler")
        for t in self.tests:
            self.registry.freeze_test(t, by="compiler")
        self.log("FROZEN", hypotheses={h.hypothesis_id: h.spec_hash for h in self.hypotheses}, tests={t.test_id: t.spec_hash for t in self.tests})
        if self.config.release_reserve_to_all and ("RESERVE" in self.config.judge_partitions or self.config.judge_partition == "RESERVE"):
            self.broker.release_reserve(actor="auditor", hypothesis_ids=[h.hypothesis_id for h in self.hypotheses],
                                        reason="new hypothesis generation: CONFIRMATION contaminated for this lineage; judge on untouched RESERVE")

        # HISTORICAL imports (exploratory grade)
        for o in self.provider.imported_outcomes():
            self.broker.request(role=Role.SCOUT, actor="scout:historical", partition="HISTORICAL", resource=o.test_id)
            self._record(o)

        # SCOUT on discovery: run every usable test in discriminator order (discovery is cheap to inspect)
        rep = discriminator.rank_tests(self.hypotheses, self._states(), self.tests, availability=self._availability(self.config.scout_partition),
                                       measurement=self.measurement)
        self.discrimination_reports.append(dict(stage="SCOUT", **rep))
        for c in rep["candidates"]:
            if c["partition"] == "-" or c["eig"] <= 0:
                continue
            self._run_test(self._tests[c["test_id"]], self.config.scout_partition, Role.SCOUT, Stage.DISCOVERY)
        self._adjudicate(None, "SCOUT")
        if self.after_scout is not None:
            self.after_scout(self)

        # JUDGE: phenomenon gate first, then one best test at a time, single look (per judge partition)
        verdict = None
        for jp in (self.config.judge_partitions or (self.config.judge_partition,)):
            if self.config.phenomenon_test in self._tests:
                self._run_test(self._tests[self.config.phenomenon_test], jp, Role.JUDGE, Stage.CONFIRMATION)
            verdict = None
            if self.config.judge_all:
                for t in sorted(self.tests, key=lambda x: x.test_id):
                    if t.test_id == self.config.phenomenon_test or t.test_id not in contrasts.REGISTRY:
                        continue
                    if (jp, t.statistic_family) in {(o.partition, o.feature_family) for o in self.outcomes if o.partition == jp}:
                        continue
                    self._run_test(t, jp, Role.JUDGE, Stage.CONFIRMATION)
                verdict = self._adjudicate({"status": discriminator.NO_TEST, "candidates": []}, f"JUDGE_{jp}_ALL")
                continue
            for step in range(self.config.max_judge_steps):
                run_already = {(o.partition, o.feature_family) for o in self.outcomes if o.partition == jp}
                rep = discriminator.rank_tests(self.hypotheses, self._states(), self.tests, availability=self._availability(jp),
                                               measurement=self.measurement, already_run=run_already)
                self.discrimination_reports.append(dict(stage=f"JUDGE_{jp}_{step}", **rep))
                if rep["status"] != "RANKED":
                    verdict = self._adjudicate(rep, f"JUDGE_{jp}_{step}_TERMINAL")
                    break
                best = rep["candidates"][0]
                self._run_test(self._tests[best["test_id"]], jp, Role.JUDGE, Stage.CONFIRMATION)
                after = discriminator.rank_tests(self.hypotheses, self._states(), self.tests, availability=self._availability(jp),
                                                 measurement=self.measurement,
                                                 already_run={(o.partition, o.feature_family) for o in self.outcomes if o.partition == jp})
                verdict = self._adjudicate(after, f"JUDGE_{jp}_{step}")
                if verdict["terminal"]:
                    break
        if verdict is None or not verdict["terminal"]:
            final_rep = {"status": discriminator.NO_TEST, "candidates": []}
            verdict = self._adjudicate(final_rep, "BUDGET_EXHAUSTED")
            self.deviations.append({"reason": "judge step budget exhausted before a terminal verdict; adjudicated with no remaining tests"})

        # DESIGN
        # design operates on single mechanisms still in play (mixtures are compositions, not separate interventions)
        survivors = [h for h in self.hypotheses if h.mechanism_class != "MIXTURE_MODEL" and h.hypothesis_id in verdict["surviving"]]
        needs = ("NON_IDENTIFIABLE_FROM_AVAILABLE_DATA", "NEW_EXPERIMENT_REQUIRED", "MEASUREMENT_MODEL_INADEQUATE", "MULTIPLE_MECHANISMS_REQUIRED")
        if self.config.policy in ("1.1", "1.2", "1.3", "1.4"):
            needs = needs + ("MECHANISM_DISFAVORED", "MECHANISM_FALSIFIED_WITHIN_TESTED_MODEL")
            survivors = [h for h in self.hypotheses if h.mechanism_class != "MIXTURE_MODEL"
                         and verdict["hypotheses"][h.hypothesis_id]["state"] not in ("FALSIFIED_WITHIN_MODEL", "STRONGLY_DISFAVORED")]
        proposal = design.propose(survivors) if verdict["verdict"] in needs else {"status": "NOT_REQUIRED", "proposals": []}
        self.resource["seconds"] = round(time.time() - self.resource["started"], 1)
        self.log("END", verdict=verdict["verdict"])
        self.result = dict(verdict=verdict, proposal=proposal)
        return self.result

    # -------------------------------------------------------------------------------------------------
    def write_outputs(self) -> dict:
        d = self.run_dir / "outputs"
        d.mkdir(exist_ok=True)
        files = {
            "hypotheses.json": [h.content() | {"spec_hash": h.spec_hash} for h in self.hypotheses],
            "hypothesis_generations.json": {hid: v for hid, v in self.registry.hypotheses.items()},
            "tests.json": [t.content() | {"spec_hash": t.spec_hash} for t in self.tests],
            "compiled_tests.json": self.compiled,
            "measurement_model.json": self.measurement.content() | {"spec_hash": self.measurement.spec_hash},
            "partition_manifest.json": self.partition_manifest,
            "contamination_ledger.json": self.broker.contamination_report(),
            "outcomes.json": [o.__dict__ for o in self.outcomes],
            "discrimination_reports.json": self.discrimination_reports,
            "evidence_graph.json": self.graph.content(),
            "evidence_families.json": {"families": self.graph.families(), "support_summary": self.graph.support_summary()},
            "adjudications.json": self.adjudications,
            "verdict.json": self.result["verdict"],
            "experiment_proposal.json": self.result["proposal"],
            "deviations.json": self.deviations,
            "resource_record.json": self.resource,
        }
        files.update(self.extra_outputs)
        paths = {}
        for name, obj in files.items():
            p = d / name
            p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
            paths[name] = p
        return paths
