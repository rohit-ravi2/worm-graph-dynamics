"""Frozen event-window voxel crops for evoked-volume unmixing (one animal).

Why: a static-image optical kernel cannot see an activity-dependent halo (e.g. cytosolic/neurite indicator that is
dim at rest), which synthetic fixtures show reproduces coupling-like signatures. Unmixing each evoked dF volume into
compact nucleus components plus a free radial halo around the target needs voxels, which provenance_traces does not
store.

Per stim event with a target (from the animal's raw_traces/1.0 payload; nuclei identical):
  pre  = mean depth-sorted green volume over volumes v-8 .. v-1
  post = mean over volumes v+1 .. v+12            (v = event volume from the payload)
  crop = target-centred box: planes c0-8..c0+8, rows/cols c-40..c+40 (clipped; offsets stored)
Stored as float32: pre and post crops, crop origin, plus the number of full volumes averaged in each window.
No statistic is computed here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..observations import load_payload
from . import raw_traces

VERSION = "event_crops/1.0"
PRE = (-8, 0)       # volumes [v-8, v)
POST = (1, 13)      # volumes [v+1, v+13)
HALF = (8, 40, 40)


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def crop_bounds(c, shape):
    lo = [max(int(c[k]) - HALF[k], 0) for k in range(3)]
    hi = [min(int(c[k]) + HALF[k] + 1, shape[k]) for k in range(3)]
    return lo, hi


def crops_from_source(get_volume, n_volumes: int, events: list, nuclei: np.ndarray, shape) -> dict:
    """get_volume(v) -> depth-sorted volume or None. Shared by real and synthetic sources."""
    out = {}
    cache = {}

    def vol(v):
        if v not in cache:
            if len(cache) > 64:
                cache.pop(next(iter(cache)))
            cache[v] = get_volume(v) if 0 <= v < n_volumes else None
        return cache[v]

    for e in events:
        if e.get("kind") != "stim" or e["target_nucleus"] < 0:
            continue
        v = int(e["vol"])
        lo, hi = crop_bounds(nuclei[e["target_nucleus"]], shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        acc = {}
        for name, (a, b) in (("pre", PRE), ("post", POST)):
            s, n = None, 0
            for vv in range(v + a, v + b):
                V = vol(vv)
                if V is None:
                    continue
                s = V[sl].astype(np.float64) if s is None else s + V[sl]
                n += 1
            acc[name] = (s / n).astype(np.float32) if n else None
            acc["n_" + name] = n
        if acc["pre"] is None or acc["post"] is None:
            continue
        k = f"e{e['event']}"
        out[k + "_pre"], out[k + "_post"] = acc["pre"], acc["post"]
        out[k + "_origin"] = np.array(lo, np.int32)
        out[k + "_n"] = np.array([acc["n_pre"], acc["n_post"]], np.int32)
    return out


def extract_animal(sub: str, payload_path: str | Path, out_path: str | Path) -> dict:
    import h5py
    p = load_payload(payload_path)
    nuc = p["nuclei"].astype(int)
    with h5py.File(raw_traces.raw_path(sub), "r") as h:
        D = h["acquisition/PumpProbeImagingGreen/data"]
        z, tf, starts, full, vt = raw_traces._volumes(h)
        get = lambda v: raw_traces._read_volume(D, z, tf, starts, full, v)[0]
        arrays = crops_from_source(get, len(full), p["meta"]["events"], nuc, (25, 512, 512))
    np.savez_compressed(out_path, nuclei=nuc, meta_json=json.dumps(dict(sub=sub, version=VERSION, extractor_sha256=source_sha256(),
                                                                         payload_sha256=hashlib.sha256(Path(payload_path).read_bytes()).hexdigest(),
                                                                         pre=PRE, post=POST, half=HALF, events=p["meta"]["events"])), **arrays)
    return dict(sub=sub, n_events=sum(1 for k in arrays if k.endswith("_pre")))


def load(path) -> dict:
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files if k != "meta_json"}
    out["meta"] = json.loads(str(d["meta_json"]))
    return out
