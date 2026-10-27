"""Frozen continuous whole-recording ROI traces for the signal-provenance campaign (one animal).

Why a second extractor: raw_traces/1.0 stores 24-volume event windows only (~40 % of the recording) and no static
image, so neither an outcome-independent contamination kernel (from the mean image) nor spontaneous/evoked copy
comparisons over long baselines can be computed from it. This extractor adds, without changing any definition
already frozen:

* Nuclei and events: taken from the animal's raw_traces/1.0 payload (same LoG nuclei, same stim-target
  assignment). The mean volume is recomputed exactly as raw_traces does (400 volumes) and its re-segmentation must
  reproduce the payload nuclei; a mismatch aborts.
* Every volume of the recording (index aligned with raw_traces volume indices; partial sweeps are NaN rows):
  ROI means for every nucleus (roi_extract.ROI_NAMES), per-frame background statistics, per-plane acquisition
  times and per-plane depth in um (depth jitter feeds the registration-uncertainty analysis).
* The static mean volume (float32) for PSF / contamination-kernel calibration.
No statistic, contrast or outcome is computed here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .. import roi_extract as RX
from ..observations import load_payload
from . import raw_traces

VERSION = "provenance_traces/1.0"


def source_sha256() -> str:
    h = hashlib.sha256(Path(__file__).read_bytes())
    h.update(Path(RX.__file__).read_bytes())
    return h.hexdigest()


def extract_animal(sub: str, payload_path: str | Path, out_path: str | Path) -> dict:
    p = load_payload(payload_path)
    nuc = p["nuclei"].astype(int)
    with h5py.File(raw_traces.raw_path(sub), "r") as h:
        D = h["acquisition/PumpProbeImagingGreen/data"]
        z, tf, starts, full, vt = raw_traces._volumes(h)
        nv = len(full)
        acc, n = None, 0
        for v in np.linspace(5, nv - 5, 400).astype(int):
            V, _ = raw_traces._read_volume(D, z, tf, starts, full, v)
            if V is not None:
                acc = V if acc is None else acc + V
                n += 1
        M = (acc / n).astype(np.float32)
        nuc2, _ = raw_traces._segment(M)
        if nuc2.shape != nuc.shape or not np.array_equal(nuc2, nuc):
            raise RuntimeError(f"{sub}: re-segmentation does not reproduce raw_traces/1.0 nuclei")
        shape = M.shape
        ops = RX.roi_operators(nuc, shape)
        R = np.full((nv, len(nuc), len(RX.ROI_NAMES)), np.nan, np.float32)
        FS = np.full((nv, shape[0], len(RX.FRAME_STATS)), np.nan, np.float32)
        PT = np.full((nv, shape[0]), np.nan)
        PZ = np.full((nv, shape[0]), np.nan, np.float32)
        for v in range(nv):
            if not full[v]:
                continue
            a, b = starts[v], starts[v + 1]
            order = np.argsort(z[a:b])
            V = D[a:b].astype(np.float32)[order]
            R[v] = RX.extract_volume(V, ops)
            FS[v] = RX.frame_stats(V)
            PT[v] = tf[a:b][order]
            PZ[v] = z[a:b][order]
    np.savez_compressed(out_path, nuclei=nuc, rois=R, frame_stats=FS, plane_times=PT, plane_depth_um=PZ, mean_volume=M,
                        vol_time=vt, meta_json=json.dumps(dict(sub=sub, version=VERSION, extractor_sha256=source_sha256(),
                                                               roi_names=RX.ROI_NAMES, frame_stats=RX.FRAME_STATS,
                                                               payload_sha256=hashlib.sha256(Path(payload_path).read_bytes()).hexdigest(),
                                                               n_volumes=int(nv), n_full=int(full.sum()), n_mean_volumes=int(n),
                                                               events=p["meta"]["events"], um_per_px=raw_traces.UM_PER_PX,
                                                               um_per_plane=raw_traces.UM_PER_PLANE)))
    return dict(sub=sub, n_nuclei=int(len(nuc)), n_volumes=int(nv))


def load(path) -> dict:
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files if k != "meta_json"}
    out["meta"] = json.loads(str(d["meta_json"]))
    return out
