"""Discovery/confirmation firewall: roles, freezing, generations, contamination, access ledger.

Roles
-----
SCOUT     reads DISCOVERY (and SYNTHETIC/HISTORICAL) outcomes; proposes hypotheses.
COMPILER  reads no outcome data; turns hypotheses into predictions/tests (metadata only).
JUDGE     evaluates FROZEN predictions on CONFIRMATION data it was not allowed to expose earlier.
AUDITOR   verifies hashes, ledgers, partitions; reads no outcome payloads.

Rules enforced by ``AccessBroker.request`` (every data read in the engine goes through it)
------------------------------------------------------------------------------------------
1. SCOUT and COMPILER can never read CONFIRMATION or RESERVE payloads.
2. JUDGE can read CONFIRMATION only for a test whose spec is frozen AND whose lineage hypotheses are all
   frozen BEFORE the read, AND only if the (partition, statistic family) pair is not contaminated for
   that lineage.
3. Reading CONFIRMATION outcomes marks (partition, family) as inspected by the lineage roots of the
   hypotheses judged. A later generation that descends from any inspected lineage finds those pairs
   CONTAMINATED and is refused: it needs RESERVE (explicitly released by the AUDITOR) or new data.
4. Re-freezing a hypothesis id with different content is refused: a changed hypothesis is a new
   generation.
5. RESERVE requires an AUDITOR release naming the generation it is released to.
6. Historical contamination (analyses run before this engine existed) is loaded into the ledger and
   applies to every lineage.

Every request (granted or refused) is appended to a hash-chained ledger.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from .spec import HypothesisSpec, MechanismTestSpec, canonical_sha


class Role(str, Enum):
    SCOUT = "SCOUT"
    COMPILER = "COMPILER"
    JUDGE = "JUDGE"
    AUDITOR = "AUDITOR"


class AccessViolation(PermissionError):
    pass


class ContaminationViolation(PermissionError):
    pass


class FreezeViolation(RuntimeError):
    pass


GRADES = ("UNTOUCHED", "PARTIAL", "CONTAMINATED")
GENESIS = "0" * 64


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AccessLedger:
    """Append-only, hash-chained JSONL ledger (in memory, optionally mirrored to a file)."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.entries: list[dict] = []
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                self.entries.append(json.loads(line))
            self.verify()

    def append(self, **payload) -> dict:
        prev = self.entries[-1]["entry_hash"] if self.entries else GENESIS
        body = dict(seq=len(self.entries), at=payload.pop("at", _now()), prev_hash=prev, **payload)
        body["entry_hash"] = canonical_sha(body)
        self.entries.append(body)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(body, sort_keys=True) + "\n")
        return body

    def verify(self) -> None:
        prev = GENESIS
        for i, e in enumerate(self.entries):
            body = {k: v for k, v in e.items() if k != "entry_hash"}
            if e["seq"] != i or e["prev_hash"] != prev or canonical_sha(body) != e["entry_hash"]:
                raise ValueError(f"access ledger chain broken at seq {i}")
            prev = e["entry_hash"]

    @property
    def head(self) -> str:
        return self.entries[-1]["entry_hash"] if self.entries else GENESIS


@dataclass
class HistoricalContamination:
    partition: str            # DISCOVERY / CONFIRMATION / RESERVE / "ALL_WT"
    statistic_family: str
    grade: str                # PARTIAL or CONTAMINATED
    evidence: str             # where it was inspected
    units: str = "all animals in partition"


class FreezeRegistry:
    def __init__(self, ledger: AccessLedger):
        self.ledger = ledger
        self.hypotheses: dict[str, dict] = {}
        self.tests: dict[str, dict] = {}

    def freeze_hypothesis(self, h: HypothesisSpec, *, by: str) -> dict:
        h.validate()
        prev = self.hypotheses.get(h.hypothesis_id)
        if prev is not None:
            if prev["spec_hash"] != h.spec_hash:
                raise FreezeViolation(f"{h.hypothesis_id} already frozen with different content; create generation g{h.generation + 1}")
            return prev
        rec = self.ledger.append(action="FREEZE_HYPOTHESIS", role=Role.COMPILER, actor=by, resource=h.hypothesis_id,
                                 spec_hash=h.spec_hash, lineage_root=self.lineage_root(h), granted=True)
        self.hypotheses[h.hypothesis_id] = dict(spec_hash=h.spec_hash, seq=rec["seq"], parents=list(h.parent_hypothesis_ids),
                                                lineage_root=self.lineage_root(h))
        return self.hypotheses[h.hypothesis_id]

    def freeze_test(self, t: MechanismTestSpec, *, by: str) -> dict:
        t.validate()
        prev = self.tests.get(t.test_id)
        if prev is not None:
            if prev["spec_hash"] != t.spec_hash:
                raise FreezeViolation(f"test {t.test_id} already frozen with different content; bump version")
            return prev
        rec = self.ledger.append(action="FREEZE_TEST", role=Role.COMPILER, actor=by, resource=t.test_id,
                                 spec_hash=t.spec_hash, granted=True)
        self.tests[t.test_id] = dict(spec_hash=t.spec_hash, seq=rec["seq"], family=t.statistic_family)
        return self.tests[t.test_id]

    def lineage_root(self, h: HypothesisSpec) -> str:
        if not h.parent_hypothesis_ids:
            return h.family
        roots = sorted({self.hypotheses[p]["lineage_root"] if p in self.hypotheses else p.split(".g")[0] for p in h.parent_hypothesis_ids})
        return "+".join(roots)

    def ancestors(self, hid: str) -> set[str]:
        out, stack = set(), [hid]
        while stack:
            x = stack.pop()
            for p in self.hypotheses.get(x, {}).get("parents", []):
                if p not in out:
                    out.add(p)
                    stack.append(p)
        return out


class AccessBroker:
    def __init__(self, partition_manifest: dict, ledger: AccessLedger, registry: FreezeRegistry,
                 historical: list[HistoricalContamination] = ()):
        self.manifest = partition_manifest
        self.ledger = ledger
        self.registry = registry
        self.historical = list(historical)
        # (partition, family) -> set of hypothesis ids whose judgement inspected it
        self.inspected: dict[tuple[str, str], set[str]] = {}
        self.reserve_released: dict[str, str] = {}   # hypothesis id -> release entry hash
        for hc in self.historical:
            self.ledger.append(action="HISTORICAL_CONTAMINATION", role=Role.AUDITOR, actor="contamination_ledger",
                               resource=f"{hc.partition}:{hc.statistic_family}", grade=hc.grade, evidence=hc.evidence, granted=True)

    # ---- contamination -------------------------------------------------------------------------
    def grade(self, partition: str, family: str, hypothesis_ids: list[str] = ()) -> str:
        worst = "UNTOUCHED"
        order = {g: i for i, g in enumerate(GRADES)}
        for hc in self.historical:
            if hc.statistic_family == family and hc.partition in (partition, "ALL_WT"):
                if order[hc.grade] > order[worst]:
                    worst = hc.grade
        seen = self.inspected.get((partition, family), set())
        if seen and hypothesis_ids:
            for hid in hypothesis_ids:
                lineage = self.registry.ancestors(hid) | {hid}   # single look: a lineage judges a family once
                if lineage & seen:
                    return "CONTAMINATED"
        return worst

    # ---- access -----------------------------------------------------------------------------------
    def request(self, *, role: Role | str, actor: str, partition: str, test: MechanismTestSpec | None = None,
                hypothesis_ids: list[str] = (), resource: str = "", payload_sha256: str = "") -> dict:
        role = Role(role)
        family = test.statistic_family if test else ""
        base = dict(action="READ", role=role.value, actor=actor, partition=partition, resource=resource,
                    test_id=test.test_id if test else "", family=family, hypothesis_ids=list(hypothesis_ids),
                    payload_sha256=payload_sha256)

        def deny(reason: str, exc=AccessViolation):
            self.ledger.append(**base, granted=False, reason=reason)
            raise exc(f"{role.value} {actor} denied {partition}/{family or resource}: {reason}")

        if partition in ("SYNTHETIC", "HISTORICAL", "METADATA"):
            if role == Role.COMPILER and partition != "METADATA":
                deny("COMPILER reads no outcome data")
            return self.ledger.append(**base, granted=True, grade="SYNTHETIC" if partition == "SYNTHETIC" else partition)
        if role == Role.AUDITOR:
            deny("AUDITOR verifies hashes/ledgers only; it does not read outcome payloads")
        if role == Role.COMPILER:
            deny("COMPILER reads no outcome data")
        if partition == "DISCOVERY":
            return self.ledger.append(**base, granted=True, grade=self.grade("DISCOVERY", family, hypothesis_ids))
        if partition == "CONFIRMATION":
            if role != Role.JUDGE:
                deny("only JUDGE may read CONFIRMATION")
            self._require_frozen(test, hypothesis_ids, deny)
            g = self.grade("CONFIRMATION", family, hypothesis_ids)
            if g == "CONTAMINATED":
                deny("confirmation data already inspected by this lineage (or historically) for this statistic family",
                     ContaminationViolation)
            entry = self.ledger.append(**base, granted=True, grade=g)
            self.inspected.setdefault(("CONFIRMATION", family), set()).update(hypothesis_ids)
            return entry
        if partition == "RESERVE":
            if role != Role.JUDGE:
                deny("only JUDGE may read RESERVE")
            self._require_frozen(test, hypothesis_ids, deny)
            if not hypothesis_ids or any(h not in self.reserve_released for h in hypothesis_ids):
                deny("RESERVE not released by AUDITOR for these hypothesis generations")
            g = self.grade("RESERVE", family, hypothesis_ids)
            if g == "CONTAMINATED":
                deny("reserve already inspected by this lineage for this family", ContaminationViolation)
            entry = self.ledger.append(**base, granted=True, grade=g)
            self.inspected.setdefault(("RESERVE", family), set()).update(hypothesis_ids)
            return entry
        deny(f"unknown partition {partition!r}")

    def _require_frozen(self, test, hypothesis_ids, deny):
        if test is None:
            deny("confirmation reads must name a frozen test")
        ft = self.registry.tests.get(test.test_id)
        if ft is None or ft["spec_hash"] != test.spec_hash:
            deny("test spec not frozen (or changed since freeze)")
        if not hypothesis_ids:
            deny("confirmation reads must name the frozen hypotheses being judged")
        for hid in hypothesis_ids:
            if hid not in self.registry.hypotheses:
                deny(f"hypothesis {hid} not frozen before confirmation access")

    def release_reserve(self, *, actor: str, hypothesis_ids: list[str], reason: str) -> dict:
        for hid in hypothesis_ids:
            if hid not in self.registry.hypotheses:
                raise FreezeViolation(f"cannot release RESERVE to unfrozen hypothesis {hid}")
        e = self.ledger.append(action="RELEASE_RESERVE", role=Role.AUDITOR.value, actor=actor, hypothesis_ids=list(hypothesis_ids),
                               reason=reason, granted=True)
        for hid in hypothesis_ids:
            self.reserve_released[hid] = e["entry_hash"]
        return e

    def contamination_report(self) -> dict:
        return dict(historical=[hc.__dict__ for hc in self.historical],
                    inspected_in_this_run={f"{p}:{f}": sorted(v) for (p, f), v in self.inspected.items()},
                    reserve_released=dict(self.reserve_released), ledger_head=self.ledger.head,
                    n_ledger_entries=len(self.ledger.entries))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
