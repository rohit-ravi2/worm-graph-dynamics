"""Frozen animal-level partitions for the Randi proximity mechanism benchmark.

Independence unit: the ANIMAL (recording session). Events within an animal share optics, registration,
segmentation and physiological state; stimulated sources recur within animals; so animals, not
target-receiver pairs, are the unit that partitions data.

* DISCOVERY: animals whose raw-volume near-field statistics were already inspected in
  analyses/randi_distance_map/reextract (sub-0 + batch1_subjects.txt). Scout may use them.
* CONFIRMATION: a seeded random sample from the remaining eligible animals. Extracted into a sealed
  store by the frozen extractor; readable only through the access broker by the JUDGE after freeze.
* RESERVE: every other eligible animal. Never extracted in this generation; available to a later
  hypothesis generation as untouched data.

Eligibility: DANDI segmentation file, raw imaging file and Leifer-export cache all present.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import raw_traces

REPO = Path(__file__).resolve().parents[4]
BENCH_DIR = REPO / "results" / "mechanism" / "randi_proximity"
MANIFEST_NAME = "partition_manifest.json"
DISCOVERY_SOURCE = REPO / "analyses" / "randi_distance_map" / "reextract" / "batch1_subjects.txt"
N_CONFIRMATION = 40
SEED = 20260916


def canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def eligible_animals() -> list[str]:
    subs = sorted((p.name.replace("_seg.nwb", "") for p in raw_traces.SEG_DIR.glob("sub-*_seg.nwb")),
                  key=lambda s: int(s.split("-")[1]))
    return [s for s in subs if raw_traces.raw_path(s) is not None and (raw_traces.EXPORT_CACHE / f"{s}.npz").exists()]


def build_manifest() -> dict:
    elig = eligible_animals()
    discovery = sorted(set(["sub-0"] + DISCOVERY_SOURCE.read_text().split()) & set(elig), key=lambda s: int(s.split("-")[1]))
    rest = [s for s in elig if s not in discovery]
    rng = np.random.default_rng(SEED)
    conf = sorted(rng.choice(rest, size=min(N_CONFIRMATION, len(rest)), replace=False).tolist(), key=lambda s: int(s.split("-")[1]))
    reserve = [s for s in rest if s not in conf]
    body = dict(
        benchmark="randi_proximity",
        dataset="DANDI:001075 (Randi et al. 2023, WT)",
        independence_unit="animal",
        seed=SEED,
        eligible=elig,
        partitions={"DISCOVERY": discovery, "CONFIRMATION": conf, "RESERVE": reserve},
        extractor=dict(module="wormlab.mechanism.randi.raw_traces", version=raw_traces.EXTRACTOR_VERSION,
                       source_sha256=raw_traces.source_sha256()),
        rationale=("DISCOVERY = animals already inspected for raw near-field statistics (reextract batch1); "
                   "CONFIRMATION = seeded sample of untouched-for-raw-statistics animals; RESERVE = never extracted."),
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    body["manifest_sha256"] = canonical_sha({k: v for k, v in body.items() if k != "created_utc"})
    return body


def freeze(path: Path | None = None) -> dict:
    path = path or (BENCH_DIR / MANIFEST_NAME)
    if path.exists():
        raise FileExistsError(f"partition manifest already frozen at {path}; partitions are write-once")
    path.parent.mkdir(parents=True, exist_ok=True)
    m = build_manifest()
    path.write_text(json.dumps(m, indent=1, sort_keys=True))
    path.chmod(0o444)
    return m


def load(path: Path | None = None) -> dict:
    path = path or (BENCH_DIR / MANIFEST_NAME)
    m = json.loads(path.read_text())
    expect = canonical_sha({k: v for k, v in m.items() if k not in ("created_utc", "manifest_sha256")})
    if expect != m["manifest_sha256"]:
        raise ValueError("partition manifest hash mismatch: manifest was modified after freezing")
    return m


if __name__ == "__main__":
    m = freeze()
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in m["partitions"].items()}),
          m["manifest_sha256"], m["extractor"]["source_sha256"])
