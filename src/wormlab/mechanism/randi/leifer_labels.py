"""Leifer NeuroPAL labels for raw-detected nuclei, via the OSF per-volume tracked coordinates (g5 label source).

Leifer exports (legacy/.../randi_raw/exported_data/<i>_labels.txt) list one label per reference neuron in the same
column order as OSF `matches.txt`. Each reference neuron's position is the median over volumes of its tracked
coordinates, converted to the depth-sorted raw frame exactly as in ``leifer_osf.load`` (acquisition-order z ->
raw plane 24 - z). Raw nuclei (provenance payload ``nuclei``, plane/y/x) take the label of the nearest reference
neuron within ``max_xy_px`` laterally and ``max_dz`` planes axially, one-to-one (greedy by distance).
No response values are read.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import leifer_osf

REPO = Path(__file__).resolve().parents[4]
EXPORTS = REPO / "legacy/neuromod/latent-dim/randi_raw/exported_data"
VERSION = "leifer_labels/1.0"


@lru_cache(maxsize=1)
def _export_index_by_ts() -> dict:
    out = {}
    for p in EXPORTS.glob("*_ds_name.txt"):
        m = re.search(r"pumpprobe_\d+_\d+", p.read_text())
        if m:
            out[m.group(0)] = int(p.name.split("_")[0])
    return out


def reference(sub: str) -> dict:
    """Reference-neuron labels and median raw-frame positions (plane, y, x) for one WT animal."""
    folder = leifer_osf.folder_for(sub)
    i = _export_index_by_ts()[folder.name]
    labels = [l.strip() for l in open(EXPORTS / f"{i}_labels.txt")]
    labels = [x if (x and not re.fullmatch(r"[-0-9. ]*", x)) else "" for x in labels]
    b = json.loads((folder / "brains.json").read_text())
    n_in = np.array(b["nInVolume"], int)
    coords = np.array(b["coordZYX"], float)
    offs = np.concatenate([[0], np.cumsum(n_in)[:-1]])
    with open(folder / "matches.txt") as g:
        g.readline()
        m = np.loadtxt(g)
    zof = b["zOfFrame"]
    frames = np.array([len(z) for z in zof])
    direction = np.array([np.sign(z[-1] - z[0]) if len(z) > 1 else 0 for z in zof])
    V, N = m.shape
    tr = np.full((V, N, 3), np.nan)
    vi, ni = np.nonzero(m >= 0)
    tr[vi, ni] = coords[offs[vi] + m[vi, ni].astype(int)]
    up = direction > 0
    tz = tr[..., 0].copy()
    tz[up] = (frames[up, None] - 1) - tz[up]
    tr[..., 0] = 24 - tz
    assert len(labels) == N, f"{sub}: {len(labels)} labels vs {N} reference neurons"
    return dict(sub=sub, export_index=i, labels=labels, pos=np.nanmedian(tr, 0))


def label_nuclei(nuclei: np.ndarray, ref: dict, max_xy_px: float = 2.5, max_dz: float = 1.5) -> tuple[list[str], dict]:
    pos = ref["pos"]
    ok = np.all(np.isfinite(pos), 1) & np.array([bool(x) for x in ref["labels"]])
    cand = []
    for j in np.where(ok)[0]:
        dxy = np.hypot(nuclei[:, 1] - pos[j, 1], nuclei[:, 2] - pos[j, 2])
        dz = np.abs(nuclei[:, 0] - pos[j, 0])
        for k in np.where((dxy <= max_xy_px) & (dz <= max_dz))[0]:
            cand.append((float(dxy[k] + 0.42 / 1.22 * dz[k]), int(k), int(j)))
    out = [""] * len(nuclei)
    used_n, used_r = set(), set()
    for _, k, j in sorted(cand):
        if k in used_n or j in used_r:
            continue
        out[k] = ref["labels"][j]
        used_n.add(k)
        used_r.add(j)
    return out, dict(n_ref_labeled=int(ok.sum()), n_mapped=len(used_r), n_nuclei=int(len(nuclei)))
