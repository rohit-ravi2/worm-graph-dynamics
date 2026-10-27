"""Scan-time-resolved event traces from DANDI:001075 raw PumpProbe volumes (one animal).

Independent of Leifer's deposited ROIs/signals (their raw<->segmentation linkage is not recoverable
from public data; see analyses/randi_distance_map/reextract/LINKAGE_NOTES.md). This module is the
frozen extractor behind the mechanism benchmark's timing / geometry / sham contrasts; its source
hash is pinned in the partition manifest, so confirmation outputs are only valid for the exact code
that produced them.

Per animal
----------
* Volumes: split at depth-direction reversals; only full 25-frame sweeps are used. Each frame keeps
  its own acquisition timestamp (KNOWN_FROM_METADATA) -> plane-resolved observation times.
* Nuclei: LoG blob detection on the depth-sorted mean green volume (400 volumes spread over the
  recording): sigma (0.8 plane, 1.8 px, 1.8 px), local maxima 3x5x5, threshold median + 10 MAD.
* Events: stim start/stop from the stim table; event volume = first volume whose mean timestamp
  exceeds stim start. Window volumes [-8, +16].
* Target: nucleus nearest the stim spot (spot x,y; green transform xy identity) within 4 px, and within
  2 planes of the flipped target-ROI centroid plane when the Leifer-export target ROI is known.
* Camera offset (ASSUMED 100 counts) is NOT subtracted; traces are raw means.
* Recorded nuclei per event: the target, every nucleus within R_NEAR_UM of the target, and a fixed
  seeded sample of up to N_FAR nuclei in (R_NEAR_UM, R_FAR_UM].
* Traces per recorded nucleus: mean over a 3x5x5 box and a 1x3x3 core at the nucleus centre, for each
  window volume, with the timestamp of the frame that imaged the nucleus centre plane.
* Red control: 3x5x5 box in the red channel at the INFERRED red transform (z flip, dy, dx), which was
  estimated from tagRFP brightness at DANDI centroids (probe_global_offset); status INFERRED_FROM_DATA.
* Sham events: up to N_SHAM windows fully inside the pre-first-stimulus or post-last-stimulus (+20 vol)
  periods, pseudo-target = a
  seeded random nucleus; same extraction.
* Labels: DANDI ROI label transferred to a nucleus when a DANDI centroid (x, y, 24 - z) lies within
  2 px xy and 1 plane (INFERRED_FROM_DATA).

Output: one .npz per animal (see ``extract_animal``); nothing is summarised or scored here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage as ndi

SEG_DIR = Path(os.environ.get('RANDI_SEG_DIR', Path(__file__).resolve().parents[4] / 'data' / 'randi2023_dandi001075_seg'))
# Leifer-export stim targets aligned to DANDI subjects (analyses/randi_distance_map/scripts/build_cache.py)
EXPORT_CACHE = Path(__file__).resolve().parents[4] / "analyses" / "randi_distance_map" / "out" / "cache"
CAMERA_OFFSET = 100.0     # ASSUMED: wormdatamodel framePixelOffset (Hamamatsu ORCA); traces stored raw
RAW_DIR = Path(os.environ.get('RANDI_RAW_DIR', Path(__file__).resolve().parents[4] / 'data' / 'randi2023_dandi001075'))
UM_PER_PX = 0.42          # wormdatamodel recording.frameUmPerPixel (Leifer code constant)
UM_PER_PLANE = 1.22       # median |diff(depth_per_frame_in_um)| (KNOWN_FROM_METADATA)
WIN_PRE, WIN_POST = 8, 16
R_NEAR_UM, R_FAR_UM, N_FAR = 15.0, 40.0, 30
N_SHAM = 4
RED_TRANSFORM = {"zflip": True, "dy": 16, "dx": 3}   # INFERRED_FROM_DATA
EXTRACTOR_VERSION = "raw_traces/1.0"


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def raw_path(sub: str) -> Path | None:
    for f in os.listdir(RAW_DIR):
        if f.startswith(sub + "_ses-") and "imaging" in f:
            return RAW_DIR / f
    return None


def _volumes(h):
    z = h["acquisition/PumpProbeImagingGreen/depth_per_frame_in_um"][:]
    tf = h["acquisition/PumpProbeImagingGreen/timestamps"][:]
    d = np.sign(np.diff(z))
    rev = np.where(d[1:] != d[:-1])[0] + 1
    starts = np.concatenate([[0], rev + 1])
    full = np.diff(starts) == 25
    vt = np.array([tf[starts[v]:starts[v + 1]].mean() for v in range(len(starts) - 1)])
    return z, tf, starts, full, vt


def _read_volume(D, z, tf, starts, full, v):
    """Depth-ascending volume and per-plane timestamps, or (None, None)."""
    if v < 0 or v >= len(full) or not full[v]:
        return None, None
    a, b = starts[v], starts[v + 1]
    order = np.argsort(z[a:b])
    return D[a:b].astype(np.float32)[order], tf[a:b][order]


def _segment(M):
    L = -ndi.gaussian_laplace(M - 100.0, sigma=(0.8, 1.8, 1.8))
    mx = ndi.maximum_filter(L, size=(3, 5, 5))
    thr = np.median(L) + 10.0 * 1.4826 * np.median(np.abs(L - np.median(L)))
    p = np.argwhere((L == mx) & (L > thr))
    keep = (p[:, 0] >= 1) & (p[:, 0] <= 23) & (p[:, 1] >= 3) & (p[:, 1] <= 508) & (p[:, 2] >= 3) & (p[:, 2] <= 508)
    return p[keep], float(thr)


def _box(V, c, dz=1, dy=2, dx=2):
    return float(V[c[0] - dz:c[0] + dz + 1, c[1] - dy:c[1] + dy + 1, c[2] - dx:c[2] + dx + 1].mean())


def extract_animal(sub: str, out_path: str | os.PathLike, *, seed: int = 20260916) -> dict:
    rng = np.random.default_rng(seed + int(sub.split("-")[1]))
    with h5py.File(SEG_DIR / f"{sub}_seg.nwb", "r") as h:
        ph = h["processing/ophys"]
        t_on = h["intervals/OptogeneticStimulusTable/start_time"][:]
        t_off = h["intervals/OptogeneticStimulusTable/stop_time"][:]
        power = h["intervals/OptogeneticStimulusTable/power"][:]
        T = ph["TargetedImageSegmentation/TargetPlaneSegmentation"]
        pm, pmi = T["pixel_mask"][:], T["pixel_mask_index"][:]
        C = ph["PumpProbeGreenSegmentations/PumpProbeGreenPlaneSegmentation/centroids"][:].astype(int)
        npid = ph["PumpProbeGreenSegmentations/PumpProbeGreenPlaneSegmentation/neuropal_ids"][:]
        NP = ph["NeuroPALSegmentations/NeuroPALPlaneSegmentation"]
        np_lab = {int(i): l.decode() for i, l in zip(NP["id"][:], NP["labels"][:])}
        tpid = h["intervals/OptogeneticStimulusTable/target_pumpprobe_id"][:]
    roi_label = []
    for x in npid:
        x = x.decode()
        roi_label.append(np_lab.get(int(x), "") if x.strip().isdigit() else "")
    starts_pm = np.concatenate([[0], pmi[:-1]])
    spot = np.array([[pm[s]["x"], pm[s]["y"]] for s in starts_pm], float)
    cache = EXPORT_CACHE / f"{sub}.npz"
    stim_roi = np.load(cache)["stim_target"] if cache.exists() else np.full(len(t_on), -9)

    with h5py.File(raw_path(sub), "r") as h:
        D = h["acquisition/PumpProbeImagingGreen/data"]
        DR = h["acquisition/PumpProbeImagingRed/data"]
        z, tf, starts, full, vt = _volumes(h)
        nv = len(full)
        acc, n = None, 0
        for v in np.linspace(5, nv - 5, 400).astype(int):
            V, _ = _read_volume(D, z, tf, starts, full, v)
            if V is not None:
                acc = V if acc is None else acc + V
                n += 1
        M = acc / n
        nuc, thr = _segment(M)
        P_um = nuc * np.array([UM_PER_PLANE, UM_PER_PX, UM_PER_PX])

        # label transfer (INFERRED): DANDI centroid (x, y, z) -> raw (24 - z, y, x)
        lab = np.array([""] * len(nuc), dtype=object)
        dc = np.stack([24 - C[:, 2], C[:, 1], C[:, 0]], 1)
        for r, c in enumerate(dc):
            dxy = np.hypot(nuc[:, 1] - c[1], nuc[:, 2] - c[2])
            ok = np.where((dxy <= 2) & (np.abs(nuc[:, 0] - c[0]) <= 1))[0]
            if len(ok) and roi_label[r]:
                lab[ok[np.argmin(dxy[ok])]] = roi_label[r]

        def event_block(v, target, spot_xy):
            dist = np.linalg.norm(P_um - P_um[target], axis=1) if target is not None else None
            if target is None:
                return None
            near = np.where(dist <= R_NEAR_UM)[0]
            far_pool = np.where((dist > R_NEAR_UM) & (dist <= R_FAR_UM))[0]
            far = np.sort(rng.choice(far_pool, min(N_FAR, len(far_pool)), replace=False)) if len(far_pool) else far_pool
            idx = np.concatenate([near, far]).astype(int)
            W = WIN_PRE + WIN_POST
            box = np.full((W, len(idx)), np.nan, np.float32)
            core = np.full((W, len(idx)), np.nan, np.float32)
            red = np.full((W, len(idx)), np.nan, np.float32)
            tsamp = np.full((W, len(idx)), np.nan)
            for k, vv in enumerate(range(v - WIN_PRE, v + WIN_POST)):
                V, tp = _read_volume(D, z, tf, starts, full, vv)
                if V is None:
                    continue
                a, b = starts[vv], starts[vv + 1]
                VR = DR[a:b].astype(np.float32)[np.argsort(z[a:b])]
                for j, i in enumerate(idx):
                    c = nuc[i]
                    box[k, j] = _box(V, c)
                    core[k, j] = _box(V, c, dz=0, dy=1, dx=1)
                    tsamp[k, j] = tp[c[0]]
                    rc = (24 - c[0] if RED_TRANSFORM["zflip"] else c[0], c[1] + RED_TRANSFORM["dy"], c[2] + RED_TRANSFORM["dx"])
                    if 1 <= rc[0] <= 23 and 2 <= rc[1] <= 509 and 2 <= rc[2] <= 509:
                        red[k, j] = _box(VR, rc)
            return dict(idx=idx, box=box, core=core, red=red, tsamp=tsamp)

        events = []
        ev_vol = np.searchsorted(vt, t_on)
        for e in range(len(t_on)):
            v = int(ev_vol[e])
            rec = dict(kind="stim", event=e, vol=v, t_on=float(t_on[e]), t_off=float(t_off[e]), power_W=float(power[e]),
                       spot_x=float(spot[e, 0]), spot_y=float(spot[e, 1]), target_nucleus=-1, spot_offset_px=[np.nan, np.nan],
                       target_roi_known=bool(np.isfinite(tpid[e])))
            if v - WIN_PRE < 0 or v + WIN_POST > nv:
                events.append((rec, None)); continue
            sx, sy = spot[e]
            dxy = np.hypot(nuc[:, 2] - sx, nuc[:, 1] - sy)
            zc = (24 - C[stim_roi[e], 2]) if 0 <= stim_roi[e] < len(C) else None
            cand = np.where((dxy <= 4) & ((np.abs(nuc[:, 0] - zc) <= 2) if zc is not None else True))[0]
            if len(cand):
                ti = int(cand[np.argmin(dxy[cand])])
                rec["target_plane_constrained"] = zc is not None
                rec["target_nucleus"] = ti
                rec["spot_offset_px"] = [float(sx - nuc[ti, 2]), float(sy - nuc[ti, 1])]
            events.append((rec, event_block(v, rec["target_nucleus"] if rec["target_nucleus"] >= 0 else None, spot[e])))
        first = int(ev_vol[0]) if len(ev_vol) else nv
        last = int(ev_vol[-1]) + WIN_POST + 20 if len(ev_vol) else 0
        pre_ok = [v for v in range(WIN_PRE + 2, first - WIN_POST - 2, WIN_PRE + WIN_POST)]
        post_ok = [v for v in range(last + WIN_PRE, nv - WIN_POST - 2, WIN_PRE + WIN_POST)]
        sham_starts = (pre_ok + post_ok)[:N_SHAM]
        for k, v in enumerate(sham_starts):
            ti = int(rng.integers(0, len(nuc)))
            rec = dict(kind="sham", event=10000 + k, vol=int(v), t_on=float(vt[v]), t_off=float(vt[v]), power_W=0.0,
                       spot_x=float(nuc[ti, 2]), spot_y=float(nuc[ti, 1]), target_nucleus=ti, spot_offset_px=[0.0, 0.0],
                       target_roi_known=False)
            events.append((rec, event_block(v, ti, None)))

    arrays = {}
    meta_events = []
    for rec, blk in events:
        meta_events.append(rec)
        if blk is None:
            continue
        key = f"e{rec['event']}"
        for k2, arr in blk.items():
            arrays[f"{key}_{k2}"] = arr
    meta = dict(sub=sub, extractor=EXTRACTOR_VERSION, extractor_sha256=source_sha256(), n_nuclei=int(len(nuc)),
                seg_threshold=thr, n_mean_volumes=int(n), um_per_px=UM_PER_PX, um_per_plane=UM_PER_PLANE,
                red_transform=RED_TRANSFORM, window=[WIN_PRE, WIN_POST], r_near_um=R_NEAR_UM, r_far_um=R_FAR_UM,
                events=meta_events, seed=seed)
    np.savez_compressed(out_path, nuclei=nuc, labels=lab.astype(str), meta_json=json.dumps(meta), **arrays)
    return dict(sub=sub, n_nuclei=int(len(nuc)), n_events=len(meta_events),
                n_stim_with_target=sum(1 for r, _ in events if r["kind"] == "stim" and r["target_nucleus"] >= 0),
                n_sham=sum(1 for r, _ in events if r["kind"] == "sham"), n_labeled_nuclei=int((lab != "").sum()))


if __name__ == "__main__":
    import sys
    print(json.dumps(extract_animal(sys.argv[1], sys.argv[2])))
