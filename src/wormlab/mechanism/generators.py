"""Counterfactual mechanism generators -> synthetic trace payloads in the raw_traces format.

Synthetic payloads go through the SAME ``observations.build_table`` and ``contrasts`` code as real data.
Mechanisms are deliberately minimal and interpretable (few parameters, none fitted to the target phenomenon):

* optical      : stimulus light activates any nucleus near the SPOT (anisotropic Gaussian footprint),
                 independently of the target and of wiring; latency ~0.
* contamination: receivers' green measurement mixes in a distance-weighted copy of the TARGET's signal
                 (point-spread / extraction overlap); no neural activation; latency 0.
* stim_artifact: stimulation light produces a brief green+red blip in the stim volume near the spot.
* gap          : target activation drives gap-coupled receivers with prob p_gap, latency 5-50 ms.
* chem         : target activation drives chemical targets with prob p_chem, latency 0.3-1.0 s.
* poly         : target activation drives 2-hop receivers with prob p_poly, latency 0.8-2.0 s.
* peptide      : target activation drives nuclei within a spatial radius (wiring-independent), latency 2-5 s.
* global       : with prob p_global per event, broad activation of all nuclei, latency 0.5-3 s.

Acquisition model: 25 planes per volume, 20 ms per frame, alternating sweep direction, one sample per
nucleus per volume at its plane's frame time (so scan-order artifacts are reproduced).
"""
from __future__ import annotations

import numpy as np

from .observations import CAMERA_OFFSET, Connectome

FRAME_S = 0.02
PLANES = 25
VOL_S = FRAME_S * PLANES
WIN_PRE, WIN_POST = 8, 16


def synthetic_geometry(rng, n=180, planes=25, extent_px=(240, 120)):
    """Nuclei on a jittered lattice (min spacing ~ nucleus size), plane/y/x integer coords."""
    pts = []
    while len(pts) < n:
        c = np.array([rng.integers(2, planes - 2), rng.integers(200, 200 + extent_px[1]), rng.integers(60, 60 + extent_px[0])])
        if all(np.linalg.norm((c - p) * np.array([1.22, 0.42, 0.42])) > 2.6 for p in pts):
            pts.append(c)
    return np.array(pts)


def spatial_connectome(nuc, rng, *, um=(1.22, 0.42, 0.42), p_chem=0.35, lam_chem=8.0, p_gap=0.35, lam_gap=4.0,
                       gap_nearest_only=False):
    """Random wiring whose probability decays with distance (so topology correlates with proximity)."""
    P = nuc * np.array(um)
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    n = len(nuc)
    chem = (rng.random((n, n)) < p_chem * np.exp(-D / lam_chem)).astype(float)
    np.fill_diagonal(chem, 0)
    if gap_nearest_only:
        gap = np.zeros((n, n))
        for i in range(n):
            j = np.argsort(D[i])[1:3]
            gap[i, j] = gap[j, i] = 1
    else:
        g = (rng.random((n, n)) < p_gap * np.exp(-D / lam_gap)).astype(float)
        gap = np.triu(g, 1)
        gap = gap + gap.T
    ids = [f"n{i}" for i in range(n)]
    return Connectome(ids, chem, gap)


def _calcium(t, a, amp, tau_r=0.25, tau_d=2.0):
    x = np.clip(t - a, 0, None)
    return np.where(t >= a, amp * (1 - np.exp(-x / tau_r)) * np.exp(-x / tau_d), 0.0)


DEFAULTS = dict(p_target=0.7, target_amp=1.5, p_opt=0.0, sigma_opt_um=3.0, axial_factor=2.0, contam=0.0, contam_sigma_um=3.0,
                stim_artifact=0.0, p_gap=0.0, p_chem=0.0, p_poly=0.0, p_pep=0.0, pep_radius_um=10.0, p_global=0.0,
                p_global_node=0.3, p_spont=0.02, noise=0.06, n_events=60, n_sham=6, recv_amp=1.0, frame_noise=0.0)


def generate_animal(rng, nuc, conn: Connectome, **params) -> dict:
    p = dict(DEFAULTS, **params)
    N = len(nuc)
    um = np.array([1.22, 0.42, 0.42])
    P = nuc * um
    F0 = rng.uniform(150, 400, N)
    R0 = rng.uniform(300, 600, N)
    events_meta, payload = [], {}
    labels = np.array(conn.ids[:N])
    W = WIN_PRE + WIN_POST

    def plane_time(v, plane):
        up = (v % 2 == 0)
        k = plane if up else PLANES - 1 - plane
        return v * VOL_S + k * FRAME_S + FRAME_S / 2

    specs = [("stim", k) for k in range(p["n_events"])] + [("sham", 10000 + k) for k in range(p["n_sham"])]
    for kind, eid in specs:
        v0 = 40 + 60 * (len(events_meta))
        target = int(rng.integers(0, N))
        t_on = v0 * VOL_S + float(rng.uniform(0, VOL_S))
        vol_first = int(np.floor(t_on / VOL_S)) + 1
        spot = nuc[target, [2, 1]].astype(float) + rng.normal(0, 1.0, 2)
        spot_off = (spot - nuc[target, [2, 1]]).tolist()
        # ---- activation times ----
        act = np.full(N, np.inf)
        amp = np.zeros(N)
        if kind == "stim" and rng.random() < p["p_target"]:
            act[target] = t_on + rng.uniform(0, 0.1)
            amp[target] = p["target_amp"]
        if kind == "sham" and rng.random() < 0.6:           # spontaneous pseudo-target activity
            act[target] = t_on + rng.uniform(0, 0.3)
            amp[target] = p["target_amp"]
        if kind == "stim" and p["p_opt"] > 0:
            d = P.copy()
            spot_um = np.array([P[target, 0], spot[1] * um[1], spot[0] * um[2]])
            dv = (d - spot_um) / np.array([p["axial_factor"], 1, 1])
            pr = p["p_opt"] * np.exp(-np.sum(dv ** 2, 1) / (2 * p["sigma_opt_um"] ** 2))
            hit = (rng.random(N) < pr)
            hit[target] = False
            act[hit] = np.minimum(act[hit], t_on + rng.uniform(0, 0.1, hit.sum()))
            amp[hit] = np.maximum(amp[hit], p["recv_amp"])
        if np.isfinite(act[target]):
            it = target
            a0 = act[target]
            for prob, rel, lat in ((p["p_gap"], "gap", (0.005, 0.05)), (p["p_chem"], "chem", (0.3, 1.0)), (p["p_poly"], "poly", (0.8, 2.0))):
                if prob <= 0:
                    continue
                if rel == "gap":
                    rec = np.nonzero(conn.gap[:, it] + conn.gap[it, :])[0]
                elif rel == "chem":
                    rec = np.nonzero(conn.chem[:, it])[0]
                else:
                    rec = np.nonzero(conn.two_hop[:, it] & (conn.adj[:, it] == 0))[0]
                rec = rec[rec != it]
                go = rec[rng.random(len(rec)) < prob]
                act[go] = np.minimum(act[go], a0 + rng.uniform(*lat, len(go)))
                amp[go] = np.maximum(amp[go], p["recv_amp"])
            if p["p_pep"] > 0:
                d = np.linalg.norm(P - P[target], axis=1)
                go = np.where((d > 0) & (d < p["pep_radius_um"]) & (rng.random(N) < p["p_pep"] * np.exp(-d / p["pep_radius_um"])))[0]
                act[go] = np.minimum(act[go], a0 + rng.uniform(2, 5, len(go)))
                amp[go] = np.maximum(amp[go], p["recv_amp"])
        if kind == "stim" and rng.random() < p["p_global"]:
            go = np.where(rng.random(N) < p["p_global_node"])[0]
            act[go] = np.minimum(act[go], t_on + rng.uniform(0.5, 3.0, len(go)))
            amp[go] = np.maximum(amp[go], p["recv_amp"])
        spont = (rng.random(N) < p["p_spont"])
        spont[target] = False
        act[spont] = np.minimum(act[spont], t_on + rng.uniform(-3, 6, spont.sum()))
        amp[spont] = np.maximum(amp[spont], p["recv_amp"])
        # ---- recorded nuclei (same rule as the real extractor) ----
        dist = np.linalg.norm(P - P[target], axis=1)
        near = np.where(dist <= 15.0)[0]
        pool = np.where((dist > 15.0) & (dist <= 40.0))[0]
        far = np.sort(rng.choice(pool, min(30, len(pool)), replace=False)) if len(pool) else pool
        idx = np.concatenate([near, far]).astype(int)
        box = np.zeros((W, len(idx)), np.float32)
        red = np.zeros((W, len(idx)), np.float32)
        ts = np.zeros((W, len(idx)))
        vols = np.arange(vol_first - 1 - WIN_PRE + 1, vol_first + WIN_POST)[:W] if False else np.arange(vol_first - WIN_PRE, vol_first + WIN_POST)
        tgt_sig = None
        for j, i in enumerate(idx):
            tt = np.array([plane_time(v, nuc[i, 0]) for v in vols])
            sig = F0[i] * (1 + _calcium(tt, act[i], amp[i]))
            ts[:, j] = tt
            if i == target:
                tgt_sig = (tt, sig)
            box[:, j] = sig
            red[:, j] = R0[i]
        if kind == "stim" and p["stim_artifact"] > 0:
            k = WIN_PRE
            for j, i in enumerate(idx):
                w = p["stim_artifact"] * np.exp(-dist[i] ** 2 / (2 * 4.0 ** 2))
                box[k, j] += w * F0[i]
                red[k, j] += w * R0[i]
        if p["frame_noise"] > 0:                          # acquisition noise shared by all nuclei imaged in the same frame
            fn = rng.normal(0, p["frame_noise"], (W, PLANES))
            planes = nuc[idx, 0]
            box = box * (1 + fn[:, planes])
        box = box * (1 + rng.normal(0, p["noise"], box.shape))
        red = red * (1 + rng.normal(0, p["noise"], red.shape))
        if p["contam"] > 0:
            tj = int(np.where(idx == target)[0][0])
            tgt_measured = box[:, tj] - F0[target]          # target's measured signal incl. its noise
            for j, i in enumerate(idx):
                if i == target:
                    continue
                w = p["contam"] * np.exp(-dist[i] ** 2 / (2 * p["contam_sigma_um"] ** 2))
                box[:, j] += w * tgt_measured
        box = box + CAMERA_OFFSET
        red = red + CAMERA_OFFSET
        key = f"e{eid}"
        payload[key + "_idx"] = idx
        payload[key + "_box"] = box.astype(np.float32)
        payload[key + "_core"] = box.astype(np.float32)
        payload[key + "_red"] = red.astype(np.float32)
        payload[key + "_tsamp"] = ts
        events_meta.append(dict(kind=kind, event=eid, vol=int(vol_first), t_on=float(t_on), t_off=float(t_on + 0.5), power_W=0.0012 if kind == "stim" else 0.0,
                                spot_x=float(spot[0]), spot_y=float(spot[1]), target_nucleus=target, spot_offset_px=spot_off, target_roi_known=True))
    payload.update(nuclei=nuc, labels=labels, meta=dict(um_per_px=0.42, um_per_plane=1.22, window=[WIN_PRE, WIN_POST], events=events_meta,
                                                         synthetic_params=p))
    return payload


def generate_cohort(seed: int, n_animals: int, *, n_nuclei=150, conn_kw=None, **params):
    rng = np.random.default_rng(seed)
    out = []
    for a in range(n_animals):
        nuc = synthetic_geometry(rng, n=n_nuclei)
        conn = spatial_connectome(nuc, rng, **(conn_kw or {}))
        out.append((f"syn-{a}", generate_animal(rng, nuc, conn, **params), conn))
    return out
