"""Frozen voxel-level extractor for T12 (prospective nucleus-centre vs inter-nuclear background), one animal.

Protocol (fixed before any T12 outcome on CONFIRMATION/RESERVE was computed; mirrors the DISCOVERY analysis in
analyses/randi_distance_map/reextract/scripts/haze_vs_discrete.py, with the event selection and statistic defined here):

* Inputs: the animal's sealed raw_traces/1.0 payload (nuclei, event list, traces) and its raw green volumes.
* Target response per stim event: the engine's own definition (observations.build_table target_z: per-nucleus robust z of
  box dF/F 0.25-6.5 s vs pre across the animal's events). z > 3 = TARGET-UP; |z| < 1 = TARGET-FLAT (control).
  Up to 12 of each per animal, in event order.
* Event dF volume: mean depth-sorted green over volumes v+1..v+12 minus mean over v-6..v-1 (v = event volume from payload).
* Normalisation: divide by the target's own 3x5x5 box dF (TARGET-UP events with target dF <= 0 are skipped);
  TARGET-FLAT events divide by max(|target box dF|, 1 count).
* Voxel classes within 3D distance shells from the target centre (0.42 um/px, 1.22 um/plane):
  nucleus-centre = within 1 px (y, x) in the same plane as a detected nucleus other than the target;
  background     = farther than 4 px (y, x) or 2 planes from every detected nucleus (target included).
* Per event and shell: median normalised dF of nucleus-centre voxels, median of background voxels, and their difference.
Output JSON per animal: per-event rows (event, class, shell, nucleus_median, background_median, difference, n voxels).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .. import observations as O
from . import raw_traces

VERSION = "voxel_x1/1.0"
SHELLS = [(2.0, 3.5), (3.5, 5.0), (5.0, 7.0), (7.0, 9.0)]
MAX_EVENTS = 12
SC = np.array([raw_traces.UM_PER_PLANE, raw_traces.UM_PER_PX, raw_traces.UM_PER_PX])


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _target_z(payload: dict, sub: str) -> dict:
    """Target robust z per stim event, EXACTLY as the engine defines it (observations.build_table: per-nucleus robust z of box
    dF/F across the animal's events; the event's target_z)."""
    tab = O.build_table(payload, sub, None)
    out = {}
    for e, k, tz in zip(tab["event"], tab["kind"], tab["target_z"]):
        if k == "stim" and int(e) not in out and np.isfinite(tz):
            out[int(e)] = float(tz)
    return out


def extract_animal(sub: str, payload_path: str | Path, out_path: str | Path) -> dict:
    p = O.load_payload(payload_path)
    nuc = p["nuclei"].astype(int)
    z = _target_z(p, sub)
    evs = {e["event"]: e for e in p["meta"]["events"]}
    up = [e for e, v in z.items() if v > 3][:MAX_EVENTS]
    flat = [e for e, v in z.items() if abs(v) < 1][:MAX_EVENTS]
    rows = []
    with h5py.File(raw_traces.raw_path(sub), "r") as h:
        D = h["acquisition/PumpProbeImagingGreen/data"]
        zz, tf, starts, full, vt = raw_traces._volumes(h)

        def mean_vol(vols):
            acc, n = None, 0
            for v in vols:
                V, _ = raw_traces._read_volume(D, zz, tf, starts, full, v)
                if V is not None:
                    acc = V if acc is None else acc + V
                    n += 1
            return acc / n if n else None

        for cls, events in (("TARGET_UP", up), ("TARGET_FLAT", flat)):
            for e in events:
                ev = evs[e]
                v, t = ev["vol"], ev["target_nucleus"]
                pre, post = mean_vol(range(v - 6, v)), mean_vol(range(v + 1, v + 13))
                if pre is None or post is None:
                    continue
                dF = post - pre
                c = nuc[t]
                tdf = float(dF[c[0] - 1:c[0] + 2, c[1] - 2:c[1] + 3, c[2] - 2:c[2] + 3].mean())
                if cls == "TARGET_UP" and tdf <= 0:
                    continue
                norm = tdf if cls == "TARGET_UP" else max(abs(tdf), 1.0)
                z0, z1 = max(c[0] - 8, 0), min(c[0] + 9, 25)
                y0, y1 = max(c[1] - 25, 0), min(c[1] + 26, 512)
                x0, x1 = max(c[2] - 25, 0), min(c[2] + 26, 512)
                gz, gy, gx = np.meshgrid(np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
                vox = np.stack([gz.ravel(), gy.ravel(), gx.ravel()], 1)
                dist = np.linalg.norm((vox - c) * SC, axis=1)
                loc = nuc[(np.abs(nuc[:, 1] - c[1]) < 32) & (np.abs(nuc[:, 2] - c[2]) < 32)]
                nuc_mask = np.zeros(len(vox), bool)
                bg_mask = np.ones(len(vox), bool)
                for o in loc:
                    dz_, dy_, dx_ = np.abs(vox[:, 0] - o[0]), np.abs(vox[:, 1] - o[1]), np.abs(vox[:, 2] - o[2])
                    if not np.array_equal(o, c):
                        nuc_mask |= (dz_ == 0) & (dy_ <= 1) & (dx_ <= 1)
                    bg_mask &= ~((dz_ <= 2) & (dy_ <= 4) & (dx_ <= 4))
                val = dF[gz, gy, gx].ravel() / norm
                for lo, hi in SHELLS:
                    sh = (dist > lo) & (dist <= hi)
                    nv, bv = val[sh & nuc_mask], val[sh & bg_mask]
                    if len(nv) >= 3 and len(bv) >= 3:
                        rows.append(dict(event=int(e), cls=cls, shell=[lo, hi], nucleus_median=float(np.median(nv)),
                                         background_median=float(np.median(bv)), difference=float(np.median(nv) - np.median(bv)),
                                         n_nucleus=int(len(nv)), n_background=int(len(bv))))
    out = dict(sub=sub, version=VERSION, extractor_sha256=source_sha256(), payload_sha256=hashlib.sha256(Path(payload_path).read_bytes()).hexdigest(),
               n_up=len(up), n_flat=len(flat), rows=rows)
    Path(out_path).write_text(json.dumps(out))
    return dict(sub=sub, n_rows=len(rows), n_up=len(up), n_flat=len(flat))
