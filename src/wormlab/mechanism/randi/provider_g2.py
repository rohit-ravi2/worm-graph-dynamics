"""g2 provider: voxel_x1/1.0 outputs (sealed per partition) + imported prior outcomes (downgraded to PARTIAL)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..engine import DataProvider
from ..spec import TestOutcome
from . import partitions
from .provider import RandiProvider, SealViolation

ADDENDUM = partitions.BENCH_DIR / "g2_addendum_manifest.json"


class RandiVoxelProvider(DataProvider):
    def __init__(self, manifest: dict, addendum: dict, prior_outcomes: list[dict]):
        self.manifest, self.addendum, self.prior = manifest, addendum, prior_outcomes
        self.base = RandiProvider(manifest)
        self.dirs = {p: partitions.BENCH_DIR / ("discovery" if p == "DISCOVERY" else f"sealed/{p.lower()}") / "voxel_x1"
                     for p in ("DISCOVERY", "CONFIRMATION", "RESERVE")}
        self._cache = {}

    def seals(self, partition):
        f = self.dirs[partition] / "SEAL.jsonl"
        return {r["sub"]: r["sha256"] for r in (json.loads(l) for l in f.read_text().splitlines()) if r["status"] in ("ok", "exists")} if f.exists() else {}

    def units(self, partition):
        if partition not in self.dirs:
            return []
        s = self.seals(partition)
        return [u for u in self.manifest["partitions"][partition] if u in s and (self.dirs[partition] / f"{u}.json").exists()]

    def data_stream(self):
        return f"voxel_x1/1.0:{self.addendum['voxel_extractor_sha256'][:12]}"

    def table(self, partition, grant):
        if not grant.get("granted") or grant.get("partition") != partition:
            raise PermissionError("table() requires a granted broker entry for this partition")
        if partition in self._cache:
            return self._cache[partition]
        seals = self.seals(partition)
        cols = {k: [] for k in ("animal", "event", "cls", "shell_lo", "nucleus_median", "background_median", "difference")}
        for u in self.units(partition):
            path = self.dirs[partition] / f"{u}.json"
            if hashlib.sha256(path.read_bytes()).hexdigest() != seals[u]:
                raise SealViolation(f"{path} does not match its seal")
            d = json.loads(path.read_text())
            if d["extractor_sha256"] != self.addendum["voxel_extractor_sha256"]:
                raise SealViolation(f"{path} produced by a different extractor")
            for r in d["rows"]:
                cols["animal"].append(u); cols["event"].append(r["event"]); cols["cls"].append(r["cls"]); cols["shell_lo"].append(r["shell"][0])
                for k in ("nucleus_median", "background_median", "difference"):
                    cols[k].append(r[k])
        self._cache[partition] = {k: np.array(v) for k, v in cols.items()}
        return self._cache[partition]

    def imported_outcomes(self):
        out = [o for o in self.base.imported_outcomes()]
        for o in self.prior:
            d = dict(o)
            d["details"] = dict(d.get("details", {}), data_stream=d.get("details", {}).get("data_stream") or "raw_traces/1.0",
                                imported_from_generation=d.get("details", {}).get("imported_from_generation", "prior"),
                                regraded="inspected before this generation existed: PARTIAL at most")
            if d["partition"] in ("CONFIRMATION", "RESERVE") and d["contamination_grade"] == "UNTOUCHED":
                d["contamination_grade"] = "PARTIAL"
            d.pop("outcome_id", None)
            out.append(TestOutcome(**d).finalize())
        return out
