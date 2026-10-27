"""Loader for the public OSF deposit of Randi et al. 2023 (osf.io/e2syt, exported_data_full.tar.gz, sha256 recorded in
data/randi_osf_e2syt/exported_data_full.tar.gz.sha256).

Per pumpprobe recording: brains.json (all per-volume detections coordZYX, nInVolume, zOfFrame), matches.txt (per volume x
reference neuron: detection index within that volume, -1 unmatched), green.txt / red.txt (Leifer box signals at the
tracked coordinates, same volume x reference-neuron layout). Pickles (fconn/recording) are NOT loaded (untrusted code).

Volume alignment to raw PumpProbe volumes: Leifer splits the frame stream into volumes of 24-26 frames; the raw stream
is split at sweep reversals. Each Leifer volume is assigned to the raw volume containing its middle frame, assuming the
Leifer frame stream starts at raw frame ``frame_offset``; the offset is chosen to maximise agreement of sweep direction
(sign of zOfFrame slope vs raw depth slope, which have opposite sign conventions) and is validated downstream by
stimulation-locked target responses. Status of the alignment: INFERRED_FROM_DATA. Preferred: ``align_by_timestamps`` (DANDI seg per-volume timestamps).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OSF_DIR = Path(__file__).resolve().parents[4] / "data" / "randi_osf_e2syt" / "exported_data_full"


def folder_for(sub: str) -> Path | None:
    from . import raw_traces
    p = raw_traces.raw_path(sub)
    if p is None:
        return None
    ses = p.name.split("_ses-")[1].split("_")[0]          # 20210824-104000
    f = OSF_DIR / f"pumpprobe_{ses.replace('-', '_')}"
    return f if f.exists() else None


def load(sub: str) -> dict:
    f = folder_for(sub)
    b = json.loads((f / "brains.json").read_text())
    n_in = np.array(b["nInVolume"], int)
    coords = np.array(b["coordZYX"], float)
    offs = np.concatenate([[0], np.cumsum(n_in)[:-1]])
    matches = np.loadtxt(f / "matches.txt")
    green = np.loadtxt(f / "green.txt")
    red = np.loadtxt(f / "red.txt")
    V, N = matches.shape
    tracked = np.full((V, N, 3), np.nan)
    ok = matches >= 0
    vi, ni = np.nonzero(ok)
    tracked[vi, ni] = coords[offs[vi] + matches[vi, ni].astype(int)]
    zof = b["zOfFrame"]
    frames = np.array([len(z) for z in zof])
    direction = np.array([np.sign(z[-1] - z[0]) if len(z) > 1 else 0 for z in zof])
    targets = [int(x) for x in (f / "targets_manually_located.txt").read_text().split("\n")[1:] if x.strip() and not x.startswith("#")]
    # coordZYX z is the frame index in ACQUISITION order within the volume; convert to the depth-sorted raw plane index
    # (validated on sub-0: z jitter p90 5 -> 1 plane; 68/114 reference neurons within 2.5 px of raw LoG nuclei)
    tz = tracked[..., 0].copy()
    up = direction > 0
    tz[up] = (frames[up, None] - 1) - tz[up]
    tracked_raw = tracked.copy()
    tracked_raw[..., 0] = 24 - tz
    return dict(folder=str(f), tracked=tracked, tracked_raw_zyx=tracked_raw, green=green, red=red, frames_per_volume=frames, direction=direction,
                n_ref=N, n_vol=V, targets_manually_located=targets, labels=b.get("labels"))


def align_by_timestamps(sub: str, raw_vol_time: np.ndarray) -> np.ndarray:
    """Leifer/DANDI volume index -> nearest raw volume index using the DANDI seg NWB per-volume timestamps."""
    import h5py
    from . import raw_traces
    with h5py.File(raw_traces.SEG_DIR / f"{sub}_seg.nwb", "r") as h:
        ts = h["processing/ophys/GreenSignals/BaseGreenSignal/timestamps"][:]
    idx = np.searchsorted(raw_vol_time, ts)
    idx = np.clip(idx, 1, len(raw_vol_time) - 1)
    left = raw_vol_time[idx - 1]
    right = raw_vol_time[idx]
    return np.where(np.abs(ts - left) <= np.abs(ts - right), idx - 1, idx)


def align_to_raw(osf: dict, raw_starts: np.ndarray, raw_depth: np.ndarray, max_offset: int = 60) -> dict:
    """Map Leifer volume index -> raw volume index. raw_starts: frame index of each raw volume start (len V_raw+1)."""
    fpv = osf["frames_per_volume"]
    cum = np.concatenate([[0], np.cumsum(fpv)])
    raw_dir = np.array([np.sign(raw_depth[raw_starts[v + 1] - 1] - raw_depth[raw_starts[v]]) for v in range(len(raw_starts) - 1)])
    best = None
    for off in range(-max_offset, max_offset + 1):
        mid = cum[:-1] + fpv // 2 + off
        rv = np.searchsorted(raw_starts, mid, side="right") - 1
        ok = (rv >= 0) & (rv < len(raw_dir)) & (mid < raw_starts[-1])
        agree = np.mean(osf["direction"][ok] == -raw_dir[rv[ok]]) if ok.any() else 0
        if best is None or agree > best[1]:
            best = (off, agree, rv, ok)
    off, agree, rv, ok = best
    return dict(frame_offset=int(off), direction_agreement=float(agree), leifer_to_raw=np.where(ok, rv, -1))
