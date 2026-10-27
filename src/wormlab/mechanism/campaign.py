"""Campaign bookkeeping: CampaignQuestion ledger (P0-P3) and a machine-readable claim registry.

* Every question has a priority and a status; a campaign may close only when every P0/P1 question is TERMINAL.
* Terminal statuses name what was learned, including "blocked on a named asset" (never a silent drop).
* Claims carry allowed wording AND forbidden wording; ``check_text`` scans a report for forbidden phrasings so a
  retracted claim cannot silently re-enter a document.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum


class Priority(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class QStatus(str, Enum):
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    ANSWERED = "ANSWERED"                          # decided with stated evidence grade
    NON_IDENTIFIABLE = "NON_IDENTIFIABLE"          # existing data cannot separate; resolution spec attached
    BLOCKED_ON_ASSET = "BLOCKED_ON_ASSET"          # named missing asset / experiment
    SUPERSEDED = "SUPERSEDED"                      # replaced by a sharper question (id given)
    DEFERRED_OUT_OF_SCOPE = "DEFERRED_OUT_OF_SCOPE"  # P2/P3 only, with reason

    @property
    def terminal(self) -> bool:
        return self not in (QStatus.OPEN, QStatus.IN_PROGRESS)


@dataclass
class CampaignQuestion:
    qid: str
    section: str
    question: str
    priority: Priority
    status: QStatus = QStatus.OPEN
    answer: str = ""
    evidence: list = field(default_factory=list)       # outcome ids / artifact paths
    evidence_grade: str = ""                           # CONFIRMATION / PARTIAL / EXPLORATORY / SYNTHETIC / LITERATURE
    blocked_on: str = ""
    superseded_by: str = ""
    next_test: str = ""

    def validate(self) -> list[str]:
        err = []
        if self.status == QStatus.ANSWERED and (not self.answer or not self.evidence or not self.evidence_grade):
            err.append(f"{self.qid}: ANSWERED needs answer, evidence and evidence_grade")
        if self.status == QStatus.BLOCKED_ON_ASSET and not self.blocked_on:
            err.append(f"{self.qid}: BLOCKED_ON_ASSET must name the asset")
        if self.status == QStatus.NON_IDENTIFIABLE and not self.next_test:
            err.append(f"{self.qid}: NON_IDENTIFIABLE must name the resolving test/experiment")
        if self.status == QStatus.SUPERSEDED and not self.superseded_by:
            err.append(f"{self.qid}: SUPERSEDED must name the replacing question")
        if self.status == QStatus.DEFERRED_OUT_OF_SCOPE and self.priority in (Priority.P0, Priority.P1):
            err.append(f"{self.qid}: P0/P1 questions cannot be deferred")
        return err


class QuestionLedger:
    def __init__(self, questions: list[CampaignQuestion] | None = None):
        self.q: dict[str, CampaignQuestion] = {}
        for x in questions or []:
            self.add(x)

    def add(self, x: CampaignQuestion):
        if x.qid in self.q:
            raise KeyError(f"duplicate question {x.qid}")
        self.q[x.qid] = x

    def update(self, qid: str, **kw):
        from copy import deepcopy
        x = deepcopy(self.q[qid])
        for k, v in kw.items():
            if k == "status":
                v = QStatus(v)
            setattr(x, k, v)
        errs = x.validate()
        if errs:
            raise ValueError("; ".join(errs))     # the stored question is left unchanged
        self.q[qid] = x
        return x

    def closable(self) -> tuple[bool, list[str]]:
        blockers = [f"{x.qid} ({x.priority.value}) {x.status.value}" for x in self.q.values()
                    if x.priority in (Priority.P0, Priority.P1) and not x.status.terminal]
        errs = [e for x in self.q.values() for e in x.validate()]
        return (not blockers and not errs), blockers + errs

    def to_json(self) -> str:
        return json.dumps([asdict(x) | {"priority": x.priority.value, "status": x.status.value} for x in self.q.values()], indent=1)

    @classmethod
    def from_json(cls, text: str) -> "QuestionLedger":
        out = []
        for d in json.loads(text):
            d["priority"], d["status"] = Priority(d["priority"]), QStatus(d["status"])
            out.append(CampaignQuestion(**d))
        return cls(out)


@dataclass
class Claim:
    claim_id: str
    domain: str                       # randi / worm / fish / software
    status: str                       # SUPPORTED / RETRACTED / OPEN / NON_IDENTIFIABLE / FALSIFIED / DISFAVORED
    allowed_wording: list
    forbidden_wording: list           # regex patterns (case-insensitive)
    evidence: list = field(default_factory=list)
    evidence_grade: str = ""
    scope_note: str = ""
    superseded_claims: list = field(default_factory=list)

    @property
    def claim_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


class ClaimRegistry:
    def __init__(self, claims: list[Claim] | None = None):
        self.c = {x.claim_id: x for x in claims or []}

    def check_text(self, text: str) -> list[dict]:
        """Return every forbidden phrasing found in ``text`` (line numbers), so reports can be gated."""
        hits = []
        lines = text.splitlines()
        for cl in self.c.values():
            for pat in cl.forbidden_wording:
                rx = re.compile(pat, re.I)
                for i, line in enumerate(lines, 1):
                    if rx.search(line) and not re.search(r"(forbidden|not allowed|retract|NOT:|✗|does not survive|not supported|withdrawn|superseded|do not claim)", line, re.I):
                        hits.append(dict(claim_id=cl.claim_id, pattern=pat, line=i, text=line.strip()[:200]))
        return hits

    def to_json(self) -> str:
        return json.dumps([asdict(x) | {"claim_hash": x.claim_hash} for x in self.c.values()], indent=1)
