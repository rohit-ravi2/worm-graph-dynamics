"""Real-data provider for the Randi proximity benchmark (raw_traces/1.0 payloads).

* DISCOVERY payloads: results/mechanism/randi_proximity/discovery/raw_traces/
* CONFIRMATION payloads: results/mechanism/randi_proximity/sealed/confirmation/raw_traces/ — each file's sha256
  is checked against SEAL.jsonl (written at extraction time) before use; a mismatch aborts the read.
Payloads are only opened inside ``table()``, which requires a granted broker entry for that partition.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .. import observations as O
from ..engine import DataProvider
from ..spec import Call, OutcomeStatus, Stage, TestOutcome
from . import partitions, raw_traces

REPO = partitions.REPO
ANALYSES = REPO / "analyses" / "randi_distance_map"


class SealViolation(RuntimeError):
    pass


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RandiProvider(DataProvider):
    def __init__(self, manifest: dict):
        self.manifest = manifest
        self.dirs = {"DISCOVERY": partitions.BENCH_DIR / "discovery" / "raw_traces",
                     "CONFIRMATION": partitions.BENCH_DIR / "sealed" / "confirmation" / "raw_traces",
                     "RESERVE": partitions.BENCH_DIR / "sealed" / "reserve" / "raw_traces"}
        self._conn = None
        self._cache: dict[str, dict] = {}

    def seals(self, partition: str) -> dict:
        p = self.dirs[partition] / "SEAL.jsonl"
        out = {}
        if p.exists():
            for line in p.read_text().splitlines():
                r = json.loads(line)
                if r.get("status") in ("ok", "exists"):
                    out[r["sub"]] = r["sha256"]
        return out

    def units(self, partition: str) -> list[str]:
        if partition not in self.dirs:
            return []
        seals = self.seals(partition)
        return [s for s in self.manifest["partitions"][partition] if s in seals and (self.dirs[partition] / f"{s}.npz").exists()]

    def data_stream(self) -> str:
        return f"raw_traces/1.0:{self.manifest['extractor']['source_sha256'][:12]}"

    def connectome(self):
        if self._conn is None:
            self._conn = O.Connectome.load_wormneuroatlas()
        return self._conn

    def table(self, partition: str, grant: dict) -> dict:
        if not grant.get("granted") or grant.get("partition") != partition:
            raise PermissionError("table() requires a granted broker entry for this partition")
        if partition in self._cache:
            return self._cache[partition]
        seals = self.seals(partition)
        tabs = []
        for sub in self.units(partition):
            path = self.dirs[partition] / f"{sub}.npz"
            if _sha(path) != seals[sub]:
                raise SealViolation(f"{path} does not match its extraction seal")
            tabs.append(O.build_table(O.load_payload(path), sub, self.connectome()))
        self._cache[partition] = O.concat(tabs)
        return self._cache[partition]

    # ---- historical (exploratory) evidence imported from runs 1-2 --------------------------------------------
    def imported_outcomes(self) -> list[TestOutcome]:
        out = []
        disc = self.manifest["partitions"]["DISCOVERY"]
        haze = json.loads((ANALYSES / "reextract/out/haze_vs_discrete.json").read_text())
        shell = next(r for r in haze["up"] if list(r["shell_um"]) == [3.5, 5])
        n_est, n_lo, n_hi = shell["nucleus_centre_dF"]
        b_est, b_lo, b_hi = shell["background_dF"]
        est, ci = n_est - b_est, [n_lo - b_hi, n_hi - b_lo]      # conservative interval difference
        out.append(TestOutcome("X1_nucleus_vs_background", "historical", Stage.HISTORICAL, "HISTORICAL", OutcomeStatus.DECIDED,
                               Call.POS if ci[0] > 0.03 else Call.INCONCLUSIVE if ci[1] > -0.03 else Call.NEG, est, ci, 0.03, 24, shell["n_events"],
                               disc, "haze_vs_discrete", "CONTAMINATED",
                               details={"source": "analyses/randi_distance_map/reextract/out/haze_vs_discrete.json", "shell_um": [3.5, 5],
                                        "data_stream": "reextract_indep_pipeline", "line_profiles": {k: v for k, v in haze.get("line_profiles", {}).items() if k != "mean_profile"}}).finalize())
        t6b = json.loads((ANALYSES / "out/t6b_within_pair.json").read_text())["within_pair"]["funatlas_sig"]
        rd = t6b["near_minus_far"]
        wt = json.loads((ANALYSES / "out/cache/index.json").read_text())
        out.append(TestOutcome("T0_proximity_effect", "historical", Stage.HISTORICAL, "HISTORICAL", OutcomeStatus.DECIDED, Call.POS,
                               float(rd[0]), [float(rd[1]), float(rd[2])], 0.03, len(wt), t6b["n_near_events"] + t6b["n_far_events"],
                               sorted(r["sub"] for r in wt), "proximity_rate", "PARTIAL",
                               details={"source": "analyses/randi_distance_map/out/t6b_within_pair.json", "operationalization":
                                        "within labeled pair, near (inflated d_box<2.5) vs far events, funatlas-significant pairs, DANDI Base signals",
                                        "data_stream": "dandi_base_signals"}).finalize())
        return [o for o in out if o is not None]
