"""Event-level observation tables built from trace payloads (real raw_traces .npz OR synthetic generators).

The SAME builder is applied to real and synthetic payloads, so synthetic mechanism nulls pass through the
identical measurement/analysis pipeline.

Payload (per animal): ``nuclei`` (N,3 int: plane, y, x), ``labels`` (N str), ``meta`` (dict with
``um_per_px``, ``um_per_plane``, ``window`` [pre, post], ``events`` list), and per event key ``e<k>``:
``_idx`` (recorded nuclei), ``_box`` / ``_core`` / ``_red`` (W x n traces, raw counts), ``_tsamp`` (W x n
acquisition times, s).

Row = (animal, event, receiver) for every recorded non-target nucleus of an event with a target.
"""
from __future__ import annotations
import os

import json
from dataclasses import dataclass

import numpy as np

from .measurement import onset_interval

CAMERA_OFFSET = 100.0
POST_S = (0.25, 6.5)       # post window relative to stimulus onset, by per-sample acquisition time
Z_RESP = 3.0


@dataclass
class Connectome:
    ids: list
    chem: np.ndarray     # chem[i, j] = synapses j -> i  (wormneuroatlas convention)
    gap: np.ndarray

    def __post_init__(self):
        self.index = {n: i for i, n in enumerate(self.ids)}
        A = ((self.chem + self.gap + self.gap.T) > 0).astype(np.int32)   # A[i, j]: j -> i
        self.adj = A
        n = len(self.ids)
        # shortest directed path length j -> i by BFS (cap 6)
        L = np.full((n, n), 99, np.int16)
        for j in range(n):
            L[j, j] = 0
            frontier = [j]
            d = 0
            seen = {j}
            while frontier and d < 6:
                d += 1
                nxt = []
                for u in frontier:
                    for v in np.nonzero(A[:, u])[0]:
                        if v not in seen:
                            seen.add(v)
                            L[v, j] = d
                            nxt.append(v)
                frontier = nxt
        self.path_len = L
        self.two_hop = (A @ A) > 0
        self.common = (A.T @ A)  # shared upstream-of / downstream counts proxy

    @classmethod
    def load_wormneuroatlas(cls) -> "Connectome":
        import h5py
        base = (os.path.join(os.path.dirname(__import__('wormneuroatlas').__file__), 'data') + '/')
        with h5py.File(base + "funatlas.h5") as h:
            ids = [x.decode() for x in h["neuron_ids"][:]]
        with h5py.File(base + "aconnectome_default.h5") as h:
            chem, gap = h["chem"][:].astype(float), h["gap"][:].astype(float)
        return cls(ids, chem, gap)


def load_payload(path) -> dict:
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta_json"]))
    payload = {"nuclei": d["nuclei"], "labels": d["labels"], "meta": meta}
    for k in d.files:
        if k.startswith("e"):
            payload[k] = d[k]
    return payload


def _robust_z(A: np.ndarray) -> np.ndarray:
    med = np.nanmedian(A, 0)
    mad = 1.4826 * np.nanmedian(np.abs(A - med), 0)
    cnt = np.isfinite(A).sum(0)
    g_med = np.nanmedian(A)
    g_mad = 1.4826 * np.nanmedian(np.abs(A - g_med))
    mad = np.where((cnt >= 8) & (mad > 0), mad, g_mad)
    med = np.where(cnt >= 8, med, g_med)
    with np.errstate(all="ignore"):
        return (A - med) / mad


def build_table(payload: dict, animal: str, connectome: Connectome | None = None) -> dict:
    """``base_corr``: Pearson correlation of receiver and target raw traces over the pre-stimulus window
    (contamination copies the target's measured fluctuations; independent activation does not)."""
    meta = payload["meta"]
    nuc = payload["nuclei"].astype(float)
    labels = [str(x) for x in payload["labels"]]
    pre, post = meta["window"]
    scale = np.array([meta["um_per_plane"], meta["um_per_px"], meta["um_per_px"]])
    P = nuc * scale
    N = len(nuc)
    events = [e for e in meta["events"] if e["target_nucleus"] >= 0 and f"e{e['event']}_idx" in payload]
    E = len(events)
    amp = {c: np.full((E, N), np.nan) for c in ("box", "core", "red")}
    onset = np.full((E, N, 2), np.nan)
    bcorr = np.full((E, N), np.nan)
    for i, ev in enumerate(events):
        k = f"e{ev['event']}"
        idx = payload[k + "_idx"]
        ts = payload[k + "_tsamp"]
        t_on = ev["t_on"]
        base_m = ts < t_on                                   # per-sample acquisition time decides baseline membership
        post_m = (ts >= t_on + POST_S[0]) & (ts <= t_on + POST_S[1])
        for c in ("box", "core", "red"):
            X = payload[k + "_" + c].astype(float) - CAMERA_OFFSET
            with np.errstate(all="ignore"):
                base = np.nanmean(np.where(base_m, X, np.nan), 0)
                resp = np.nanmean(np.where(post_m, X, np.nan), 0)
                amp[c][i, idx] = resp / base - 1.0
            if c == "box":
                tj = np.where(idx == ev["target_nucleus"])[0]
                if len(tj):
                    for j, nidx in enumerate(idx):
                        ok = base_m[:, tj[0]] & base_m[:, j] & np.isfinite(X[:, tj[0]]) & np.isfinite(X[:, j])
                        xt, xr = X[ok, tj[0]], X[ok, j]
                        if ok.sum() >= 5 and np.std(xt) > 0 and np.std(xr) > 0:
                            bcorr[i, nidx] = np.corrcoef(xt, xr)[0, 1]
                for j, nidx in enumerate(idx):
                    iv = onset_interval(ts[:, j], X[:, j], t_on, n_base=pre)
                    if iv is not None:
                        onset[i, nidx] = iv
    z = {c: _robust_z(amp[c]) for c in amp}
    rows = {k: [] for k in ("animal", "event", "kind", "target", "receiver", "dist_um", "dxy_um", "dz_um", "resp", "resp_z",
                            "amp", "core_z", "red_z", "target_z", "target_amp", "target_resp", "rt_lo", "rt_hi", "tt_lo", "tt_hi",
                            "offset_px", "cos_offset", "labeled", "chem", "gap", "edge", "two_hop", "path_len", "common", "base_corr")}
    for i, ev in enumerate(events):
        t = ev["target_nucleus"]
        idx = payload[f"e{ev['event']}_idx"]
        off = np.array(ev.get("spot_offset_px", [0.0, 0.0]), float)  # (dx, dy) spot - target, px
        offn = float(np.hypot(*off)) if np.all(np.isfinite(off)) else np.nan
        for r in idx:
            if r == t or not np.isfinite(z["box"][i, r]):
                continue
            v = P[r] - P[t]
            dxy = float(np.hypot(v[1], v[2]))
            cos = np.nan
            if np.isfinite(offn) and offn >= 1 and dxy > 0:
                cos = float((v[2] * off[0] + v[1] * off[1]) / (dxy * offn))  # v in um, off in px: direction only
            lab_t, lab_r = labels[t], labels[r]
            has = bool(connectome is not None and lab_t in getattr(connectome, "index", {}) and lab_r in connectome.index)
            if has:
                it, ir = connectome.index[lab_t], connectome.index[lab_r]
                chem, gap = connectome.chem[ir, it] > 0, (connectome.gap[ir, it] + connectome.gap[it, ir]) > 0
                two, pl, cm = bool(connectome.two_hop[ir, it]), int(connectome.path_len[ir, it]), int(connectome.common[ir, it])
            else:
                chem = gap = two = False
                pl, cm = -1, -1
            vals = dict(animal=animal, event=int(ev["event"]), kind=ev["kind"], target=int(t), receiver=int(r),
                        dist_um=float(np.linalg.norm(v)), dxy_um=dxy, dz_um=float(abs(v[0])), resp=bool(z["box"][i, r] > Z_RESP),
                        resp_z=float(z["box"][i, r]), amp=float(amp["box"][i, r]), core_z=float(z["core"][i, r]),
                        red_z=float(z["red"][i, r]), target_z=float(z["box"][i, t]), target_amp=float(amp["box"][i, t]),
                        target_resp=bool(z["box"][i, t] > Z_RESP), rt_lo=onset[i, r, 0], rt_hi=onset[i, r, 1],
                        tt_lo=onset[i, t, 0], tt_hi=onset[i, t, 1], offset_px=offn, cos_offset=cos, labeled=has,
                        chem=bool(chem), gap=bool(gap), edge=bool(chem or gap), two_hop=two, path_len=pl, common=cm,
                        base_corr=float(bcorr[i, r]))
            for k2, v2 in vals.items():
                rows[k2].append(v2)
    return {k: np.array(v) for k, v in rows.items()}


def concat(tables: list[dict]) -> dict:
    tables = [t for t in tables if len(t.get("animal", []))]
    if not tables:
        return {}
    return {k: np.concatenate([t[k] for t in tables]) for k in tables[0]}
