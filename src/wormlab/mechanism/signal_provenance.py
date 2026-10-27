"""Signal provenance: where does a neighbour's evoked/spontaneous signal come from?

Operates on provenance payloads (real: randi.provenance_traces; synthetic: provenance_synth) with identical code.

Core idea (dimensional analysis in raw counts)
----------------------------------------------
Optical/extraction contamination (H1) adds to receiver n's ROI a copy of the target's photons:
    dC_n = kappa[t->n] * dC_t                (counts; kappa from optics + ROI geometry; independent of n's brightness)
Any mechanism acting through the receiver's OWN indicator (H2 off-target light, H3 electrical coupling, H10 local field,
chemical) produces
    dC_n = F0_n * dFF_n                      (scales with the receiver's own baseline brightness)
kappa is computed from the STATIC mean volume only (anisotropic Gaussian light model per detected nucleus, shared
sigma per animal, non-negative amplitudes) - it never sees a stimulation outcome.

Decomposition of each trace into components (``decompose``): frame-common (per-plane background deviation), slow
(running median, 60 s), event band (0.02-0.2 Hz), high band (0.2-1 Hz), and the residual; ``SignalProvenanceReport``
collects per-family statistics with provenance.

Statistic families (docs/mechanism_engine/deep_campaign/SIGNAL_PROVENANCE.md):
  T13 optical_copy_subtracted_proximity   (family psf_kernel_sufficiency)
  T14 brightness_elasticity                (family brightness_scaling_copy_ratio)
  T15 kappa_profile_calibration            (family evoked_roi_profile; validates the optical kernel, not a discriminator)
  T16 spontaneous_excess                   (family spontaneous_evoked_slope_identity)
  W-descriptives: band transfer, lag, amplitude linearity, alpha stability, spatial decay, reciprocal asymmetry, shared-voxel noise.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.optimize import lsq_linear

from . import roi_extract as RX

PRE_VOLS, POST_VOLS = 8, 16
POST_S = (0.25, 6.5)
NEAR_UM = (0.0, 6.0)
FAR_UM = (15.0, 40.0)
Z_RESP = 3.0
CAMERA_OFFSET = 100.0
N_BOOT = 400
BOX_IDX = [0, 1, 2]
ROI = {n: i for i, n in enumerate(RX.ROI_NAMES)}
VERSION = "signal_provenance/1.0"


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# =====================================================================================================================
# static light model (PSF / contamination kernel) - STATIC IMAGE ONLY
# =====================================================================================================================
def _gauss_col(c, shape, sxy_px, sz_pl, trunc=3.5):
    P, Y, X = shape
    rz, rxy = int(np.ceil(trunc * sz_pl)), int(np.ceil(trunc * sxy_px))
    zz = np.arange(max(int(np.floor(c[0])) - rz, 0), min(int(np.floor(c[0])) + rz + 2, P))
    yy = np.arange(max(int(np.floor(c[1])) - rxy, 0), min(int(np.floor(c[1])) + rxy + 2, Y))
    xx = np.arange(max(int(np.floor(c[2])) - rxy, 0), min(int(np.floor(c[2])) + rxy + 2, X))
    G = (np.exp(-(zz - c[0]) ** 2 / (2 * sz_pl ** 2))[:, None, None] * np.exp(-(yy - c[1]) ** 2 / (2 * sxy_px ** 2))[None, :, None]
         * np.exp(-(xx - c[2]) ** 2 / (2 * sxy_px ** 2))[None, None, :])
    keep = G > 1e-3
    Z, Yi, Xi = np.nonzero(keep)
    return np.ravel_multi_index((zz[Z], yy[Yi], xx[Xi]), shape), G[keep]


def _refine_centres(M, nuc, bg):
    out = nuc.astype(float).copy()
    P, Y, X = M.shape
    for i, c in enumerate(nuc):
        z0, z1 = max(c[0] - 1, 0), min(c[0] + 2, P)
        y0, y1 = max(c[1] - 2, 0), min(c[1] + 3, Y)
        x0, x1 = max(c[2] - 2, 0), min(c[2] + 3, X)
        w = np.clip(M[z0:z1, y0:y1, x0:x1] - bg[z0:z1, None, None], 0, None)
        if w.sum() <= 0:
            continue
        zz, yy, xx = np.meshgrid(np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
        out[i] = [np.sum(w * zz) / w.sum(), np.sum(w * yy) / w.sum(), np.sum(w * xx) / w.sum()]
    return out


def _isolated(nuc, min_px=8, min_pl=3):
    d_xy = np.hypot(nuc[:, None, 1] - nuc[None, :, 1], nuc[:, None, 2] - nuc[None, :, 2])
    d_z = np.abs(nuc[:, None, 0] - nuc[None, :, 0])
    close = (d_xy < min_px) & (d_z < min_pl)
    np.fill_diagonal(close, False)
    return ~close.any(1)


def fit_light_model(M: np.ndarray, nuclei: np.ndarray, *, sigma_grid_xy=None, sigma_grid_z=None, max_isolated=60) -> dict:
    """Shared anisotropic Gaussian width (grid search on isolated nuclei, residual SSE), sub-voxel centres, and
    non-negative amplitudes for ALL detected nuclei jointly with a per-plane background."""
    M = M.astype(np.float64)
    P, Y, X = M.shape
    bg = np.percentile(M.reshape(P, -1), 10, axis=1)
    cen = _refine_centres(M, nuclei, bg)
    iso = np.where(_isolated(nuclei))[0]
    rng = np.random.default_rng(0)
    if len(iso) > max_isolated:
        iso = np.sort(rng.choice(iso, max_isolated, replace=False))
    sigma_grid_xy = sigma_grid_xy if sigma_grid_xy is not None else np.arange(1.2, 3.61, 0.2)
    sigma_grid_z = sigma_grid_z if sigma_grid_z is not None else np.arange(0.5, 2.21, 0.1)
    # crops around isolated nuclei
    crops = []
    for i in iso:
        c = nuclei[i]
        sl = (slice(max(c[0] - 3, 0), min(c[0] + 4, P)), slice(max(c[1] - 6, 0), min(c[1] + 7, Y)), slice(max(c[2] - 6, 0), min(c[2] + 7, X)))
        zz, yy, xx = np.meshgrid(np.arange(sl[0].start, sl[0].stop), np.arange(sl[1].start, sl[1].stop), np.arange(sl[2].start, sl[2].stop), indexing="ij")
        y = (M[sl] - bg[sl[0], None, None]).ravel()
        crops.append((zz.ravel() - cen[i, 0], yy.ravel() - cen[i, 1], xx.ravel() - cen[i, 2], y))
    best, grid = None, []
    for sxy in sigma_grid_xy:
        for sz in sigma_grid_z:
            sse = 0.0
            for dz, dy, dx, y in crops:
                g = np.exp(-dz ** 2 / (2 * sz ** 2) - (dy ** 2 + dx ** 2) / (2 * sxy ** 2))
                A = np.stack([g, np.ones_like(g)], 1)
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                sse += float(np.sum((y - A @ coef) ** 2))
            grid.append((float(sxy), float(sz), sse))
            if best is None or sse < best[2]:
                best = (float(sxy), float(sz), sse)
    sxy, sz = best[0], best[1]
    cols = [_gauss_col(c, M.shape, sxy, sz) for c in cen]
    rows = np.concatenate([r for r, _ in cols])
    G = sparse.csr_matrix((np.concatenate([v for _, v in cols]), (rows, np.concatenate([np.full(len(r), i) for i, (r, _) in enumerate(cols)]))),
                          shape=(M.size, len(cen)))
    support = np.unique(rows)
    y = (M - bg[:, None, None]).ravel()[support]
    Gs = G[support]
    sol = lsq_linear(Gs, y, bounds=(0, np.inf), lsmr_tol="auto", max_iter=200)
    amp = sol.x
    resid = y - Gs @ amp
    r2 = 1 - float(np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2))
    return dict(sigma_xy_px=sxy, sigma_z_planes=sz, centres=cen, amplitudes=amp, background=bg, G=G, r2_support=r2,
                n_isolated=int(len(iso)), grid=grid)


def kappa_matrices(light: dict, nuclei: np.ndarray, shape, *, sigma_scale: float = 1.0, max_um: float = 40.0,
                   um=(1.22, 0.42, 0.42)) -> dict:
    """kappa[r][n, t] = (W_r[n] @ G_t) / (W_box[t] @ G_t)  (unit-amplitude target light; optics + ROI geometry)
    own[r][n]        = (W_r[n] @ A_n G_n) / (W_box[n] @ A_n G_n)
    static_frac[n,t] = light of t in n's box / light of n in n's box."""
    if sigma_scale != 1.0:
        cols = [_gauss_col(c, shape, light["sigma_xy_px"] * sigma_scale, light["sigma_z_planes"] * sigma_scale) for c in light["centres"]]
        rows = np.concatenate([r for r, _ in cols])
        G = sparse.csr_matrix((np.concatenate([v for _, v in cols]), (rows, np.concatenate([np.full(len(r), i) for i, (r, _) in enumerate(cols)]))),
                              shape=(int(np.prod(shape)), len(cols)))
    else:
        G = light["G"]
    ops = RX.roi_operators(nuclei, shape)
    Wbox = (ops["plane_m1"] + ops["plane_0"] + ops["plane_p1"]) / 3.0
    self_box = np.asarray((Wbox @ G).diagonal()).ravel()
    K = {}
    for name in list(RX.ROI_NAMES) + ["box"]:
        W = Wbox if name == "box" else ops[name]
        WG = (W @ G).toarray()
        K[name] = WG / np.maximum(self_box[None, :], 1e-9)
    A = light["amplitudes"]
    own = {name: K[name].diagonal().copy() for name in K}
    static_frac = K["box"] * A[None, :] / np.maximum((K["box"].diagonal() * A)[:, None], 1e-9)
    np.fill_diagonal(static_frac, 1.0)
    P_um = nuclei * np.array(um)
    dist = np.linalg.norm(P_um[:, None] - P_um[None], axis=-1)
    own_light = A * self_box        # receiver's OWN modelled light in its own box (counts above background)
    return dict(kappa=K, own=own, static_frac=static_frac, dist=dist, sigma_scale=sigma_scale, own_light=own_light)


# =====================================================================================================================
# traces and event statistics
# =====================================================================================================================
def box_counts(prov: dict, offset: float = CAMERA_OFFSET) -> np.ndarray:
    return prov["rois"][..., BOX_IDX].mean(-1) - offset


def nucleus_times(prov: dict) -> np.ndarray:
    """(V, N) acquisition time of each nucleus' centre plane."""
    return prov["plane_times"][:, prov["nuclei"][:, 0]]


def frame_common(prov: dict, window_vols: int = 121) -> np.ndarray:
    """(V, N) additive frame-common deviation for each nucleus' 3 planes: per-plane p10 minus its running median."""
    p10 = prov["frame_stats"][..., 0].astype(float)
    ok = np.isfinite(p10)
    fill = np.where(ok, p10, np.nanmedian(p10, 0))
    base = ndi.median_filter(fill, size=(window_vols, 1), mode="nearest")
    dev = np.where(ok, fill - base, np.nan)
    z = prov["nuclei"][:, 0]
    P = dev.shape[1]
    return np.nanmean(np.stack([dev[:, np.clip(z + k, 0, P - 1)] for k in (-1, 0, 1)], 0), 0)


def event_windows(prov: dict) -> list[dict]:
    out = []
    V = prov["rois"].shape[0]
    for e in prov["meta"]["events"]:
        if e.get("kind") != "stim" or e["target_nucleus"] < 0:
            continue
        v = int(e["vol"])
        if v - PRE_VOLS < 0 or v + POST_VOLS > V:
            continue
        out.append(e)
    return out


def event_amplitudes(prov: dict, offset: float = CAMERA_OFFSET, rois: tuple = ("box",) + tuple(RX.ROI_NAMES)) -> dict:
    """Per stim event (with target) and nucleus: base (pre-onset samples of the [v-8, v+16) window by per-sample time),
    post (samples in [t_on+0.25, t_on+6.5]), for each ROI, in counts above offset."""
    evs = event_windows(prov)
    NT = nucleus_times(prov)
    R = prov["rois"]
    N = R.shape[1]
    out = {r: dict(base=np.full((len(evs), N), np.nan), post=np.full((len(evs), N), np.nan)) for r in rois}
    for i, e in enumerate(evs):
        v = int(e["vol"])
        sl = slice(v - PRE_VOLS, v + POST_VOLS)
        ts = NT[sl]
        bm = ts < e["t_on"]
        pm = (ts >= e["t_on"] + POST_S[0]) & (ts <= e["t_on"] + POST_S[1])
        for r in rois:
            X = (R[sl][..., BOX_IDX].mean(-1) if r == "box" else R[sl][..., ROI[r]]) - offset
            with np.errstate(all="ignore"):
                out[r]["base"][i] = np.nanmean(np.where(bm, X, np.nan), 0)
                out[r]["post"][i] = np.nanmean(np.where(pm, X, np.nan), 0)
    return dict(events=evs, rois=out)


def robust_z(A: np.ndarray) -> np.ndarray:
    med = np.nanmedian(A, 0)
    mad = 1.4826 * np.nanmedian(np.abs(A - med), 0)
    cnt = np.isfinite(A).sum(0)
    g_med = np.nanmedian(A)
    g_mad = 1.4826 * np.nanmedian(np.abs(A - g_med))
    mad = np.where((cnt >= 8) & (mad > 0), mad, g_mad)
    med = np.where(cnt >= 8, med, g_med)
    with np.errstate(all="ignore"):
        return (A - med) / mad


@dataclass
class AnimalProvenance:
    animal: str
    prov: dict
    light: dict
    kap: dict
    amps: dict
    F0: np.ndarray
    noise_dC: np.ndarray = None


def prepare_animal(animal: str, prov: dict, *, sigma_scale: float = 1.0, offset: float = CAMERA_OFFSET, light: dict | None = None) -> AnimalProvenance:
    light = light or fit_light_model(prov["mean_volume"], prov["nuclei"])
    kap = kappa_matrices(light, prov["nuclei"], prov["mean_volume"].shape, sigma_scale=sigma_scale,
                         um=(prov["meta"]["um_per_plane"], prov["meta"]["um_per_px"], prov["meta"]["um_per_px"]))
    amps = event_amplitudes(prov, offset)
    F0 = np.nanmedian(box_counts(prov, offset), 0)
    return AnimalProvenance(animal, prov, light, kap, amps, F0)


def pair_rows(ap: AnimalProvenance, *, max_um: float = FAR_UM[1]) -> dict:
    """One row per (stim event, receiver n != target) within max_um, with counts deltas for every ROI, kappa, F0."""
    evs = ap.amps["events"]
    box = ap.amps["rois"]["box"]
    dC = box["post"] - box["base"]
    with np.errstate(all="ignore"):
        amp = dC / box["base"]
    z = robust_z(amp)
    rows = {k: [] for k in ("animal", "event", "target", "receiver", "dist_um", "dC_n", "dC_t", "base_n", "base_t", "F0_n", "F0_t", "own_n",
                            "kappa", "static_frac", "target_z", "resp_z", "resp", "dz_planes", "dxy_px")}
    roi_dC = {r: [] for r in RX.ROI_NAMES}
    kap_r = {r: [] for r in RX.ROI_NAMES}
    own_r = {r: [] for r in RX.ROI_NAMES}
    nuc = ap.prov["nuclei"]
    for i, e in enumerate(evs):
        t = int(e["target_nucleus"])
        d = ap.kap["dist"][:, t]
        cand = np.where((d <= max_um) & (np.arange(len(d)) != t) & np.isfinite(z[i]))[0]
        for n in cand:
            rows["animal"].append(ap.animal); rows["event"].append(int(e["event"])); rows["target"].append(t); rows["receiver"].append(int(n))
            rows["dist_um"].append(float(d[n])); rows["dC_n"].append(float(dC[i, n])); rows["dC_t"].append(float(dC[i, t]))
            rows["base_n"].append(float(box["base"][i, n])); rows["base_t"].append(float(box["base"][i, t]))
            rows["F0_n"].append(float(ap.F0[n])); rows["F0_t"].append(float(ap.F0[t])); rows["own_n"].append(float(ap.kap["own_light"][n]))
            rows["kappa"].append(float(ap.kap["kappa"]["box"][n, t])); rows["static_frac"].append(float(ap.kap["static_frac"][n, t]))
            rows["target_z"].append(float(z[i, t])); rows["resp_z"].append(float(z[i, n])); rows["resp"].append(bool(z[i, n] > Z_RESP))
            rows["dz_planes"].append(int(abs(nuc[n, 0] - nuc[t, 0]))); rows["dxy_px"].append(float(np.hypot(*(nuc[n, 1:] - nuc[t, 1:]))))
            for r in RX.ROI_NAMES:
                roi_dC[r].append(float(ap.amps["rois"][r]["post"][i, n] - ap.amps["rois"][r]["base"][i, n]))
                kap_r[r].append(float(ap.kap["kappa"][r][n, t]))
                own_r[r].append(float(ap.kap["own"][r][n]))
    out = {k: np.array(v) for k, v in rows.items()}
    out["roi_dC"] = np.array([roi_dC[r] for r in RX.ROI_NAMES]).T if rows["animal"] else np.zeros((0, len(RX.ROI_NAMES)))
    out["roi_kappa"] = np.array([kap_r[r] for r in RX.ROI_NAMES]).T if rows["animal"] else np.zeros((0, len(RX.ROI_NAMES)))
    out["roi_own"] = np.array([own_r[r] for r in RX.ROI_NAMES]).T if rows["animal"] else np.zeros((0, len(RX.ROI_NAMES)))
    # residualised response after subtracting the optical copy (same robust-z construction, per receiver across events)
    res = np.full_like(amp, np.nan)
    for i, e in enumerate(evs):
        t = int(e["target_nucleus"])
        with np.errstate(all="ignore"):
            res[i] = (dC[i] - ap.kap["kappa"]["box"][:, t] * dC[i, t]) / box["base"][i]
        res[i, t] = amp[i, t]
    zr = robust_z(res)
    ev_index = {int(e["event"]): i for i, e in enumerate(evs)}
    out["resp_z_resid"] = np.array([zr[ev_index[ev], n] for ev, n in zip(out["event"], out["receiver"])]) if rows["animal"] else np.zeros(0)
    out["resp_resid"] = out["resp_z_resid"] > Z_RESP
    # receiver noise of the event statistic: robust spread of dC over events where it is far (> 15 um) from the target
    far_sd = np.full(dC.shape[1], np.nan)
    tgt = np.array([int(e["target_nucleus"]) for e in evs])
    farmask = ap.kap["dist"][:, tgt].T > FAR_UM[0] if len(evs) else np.zeros((0, dC.shape[1]), bool)
    for n in range(dC.shape[1]):
        x = dC[farmask[:, n], n]
        x = x[np.isfinite(x)]
        if len(x) >= 6:
            far_sd[n] = 1.4826 * np.median(np.abs(x - np.median(x)))
    out["noise_n"] = far_sd[out["receiver"]] if rows["animal"] else np.zeros(0)
    return out


def concat_rows(tabs: list[dict]) -> dict:
    tabs = [t for t in tabs if len(t["animal"])]
    return {k: np.concatenate([t[k] for t in tabs]) for k in tabs[0]} if tabs else {}


def _sel(t, m):
    return {k: v[m] for k, v in t.items()}


def _mh_rd(y, x, strata):
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        a, b = y[m & x], y[m & ~x]
        if len(a) == 0 or len(b) == 0:
            continue
        w = len(a) * len(b) / (len(a) + len(b))
        num += w * (a.mean() - b.mean())
        den += w
    return num / den if den > 0 else np.nan


def _boot(t, stat, seed, n=N_BOOT):
    A = np.unique(t["animal"])
    est = stat(t)
    if len(A) < 2:
        return est, [np.nan, np.nan], []
    groups = {a: np.where(t["animal"] == a)[0] for a in A}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[a] for a in rng.choice(A, len(A))])
        v = stat(_sel(t, idx))
        if np.all(np.isfinite(v)):
            vals.append(v)
    if len(vals) < n // 2:
        return est, [np.nan, np.nan], vals
    return est, [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], vals


# ---------------------------------------------------------------------------------------------------------------------
# T13 optical-copy-subtracted proximity effect
# ---------------------------------------------------------------------------------------------------------------------
def t13_optical_copy_subtracted(rows: dict, seed: int = 0) -> dict:
    m = (rows["target_z"] > Z_RESP) & (((rows["dist_um"] >= NEAR_UM[0]) & (rows["dist_um"] < NEAR_UM[1])) | ((rows["dist_um"] >= FAR_UM[0]) & (rows["dist_um"] < FAR_UM[1])))
    t = _sel(rows, m)
    if not len(t["animal"]):
        return dict(estimate=np.nan, ci=[np.nan, np.nan], n_rows=0)

    def stat(s):
        near = s["dist_um"] < NEAR_UM[1]
        a = _mh_rd(s["resp"], near, s["animal"])
        b = _mh_rd(s["resp_resid"], near, s["animal"])
        return np.array([a, b, 1 - b / a if a and np.isfinite(a) and abs(a) > 1e-9 else np.nan])

    est, _, vals = _boot(t, stat, seed)
    vals = np.array(vals) if len(vals) else np.full((1, 3), np.nan)
    ci = lambda k: [float(np.nanpercentile(vals[:, k], 2.5)), float(np.nanpercentile(vals[:, k], 97.5))]
    near = t["dist_um"] < NEAR_UM[1]
    return dict(estimate=float(est[1]), ci=ci(1), rd_raw=float(est[0]), rd_raw_ci=ci(0), explained_fraction=float(est[2]),
                explained_fraction_ci=ci(2), n_rows=int(len(t["animal"])), n_near=int(near.sum()), n_animals=int(len(np.unique(t["animal"]))),
                near_rate_raw=float(t["resp"][near].mean()) if near.any() else np.nan,
                near_rate_resid=float(t["resp_resid"][near].mean()) if near.any() else np.nan,
                far_rate_raw=float(t["resp"][~near].mean()) if (~near).any() else np.nan)


# ---------------------------------------------------------------------------------------------------------------------
# T14 brightness elasticity of the residual copy (own-indicator signature)
# ---------------------------------------------------------------------------------------------------------------------
def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return np.nan
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _own_fit(s, bins):
    """Per distance bin: r = g1 * kappa*dC_t + g2 * dC_t + beta * own_n * dFF_t + c, inverse-noise-variance weighted.
    g1 absorbs an optical-kernel SCALE error, g2 a flat (halo-like) copy of the target's counts; own_n is the receiver's
    own modelled light (static light model amplitude, so the target's tail light in the receiver box is excluded);
    beta is the receiver's own-indicator response per unit target dF/F. Returns the row-weighted mean beta."""
    r = s["dC_n"] - s["kappa"] * s["dC_t"]
    dff_t = s["dC_t"] / s["base_t"]
    xo = s["own_n"] * dff_t
    w = 1.0 / np.maximum(s["noise_n"], 1.0) ** 2
    b = np.digitize(s["dist_um"], bins)
    num = den = 0.0
    for k in np.unique(b):
        m = (b == k) & np.isfinite(r) & np.isfinite(xo) & np.isfinite(w)
        if m.sum() < 10:
            continue
        X = np.stack([s["kappa"][m] * s["dC_t"][m], s["dC_t"][m], xo[m], np.ones(m.sum())], 1)
        sw = np.sqrt(w[m])
        coef, *_ = np.linalg.lstsq(X * sw[:, None], r[m] * sw, rcond=None)
        num += m.sum() * coef[2]
        den += m.sum()
    return num / den if den else np.nan


def t14_brightness_elasticity(rows: dict, seed: int = 0, near_bins=(0.0, 2.0, 3.0, 4.0, 5.0, 6.0), far_bins=(15.0, 20.0, 25.0, 30.0, 40.0)) -> dict:
    """Own-indicator response beta (receiver dF/F per unit target dF/F beyond the optical copy), identified by the
    receiver-brightness dimension while a free per-bin term proportional to the target's counts absorbs any optical
    kernel error; near (0-6 um) minus far (15-40 um) control, TARGET-UP events, animal-cluster bootstrap. Optical
    contamination with ANY kernel predicts beta_near = beta_far; own-indicator mechanisms predict beta_near > beta_far."""
    base = (rows["target_z"] > Z_RESP) & (rows["dC_t"] > 0) & np.isfinite(rows["own_n"]) & np.isfinite(rows["noise_n"])
    near = base & (rows["dist_um"] < NEAR_UM[1])
    far = base & (rows["dist_um"] >= FAR_UM[0]) & (rows["dist_um"] < FAR_UM[1])
    t = _sel(rows, near | far)
    if near.sum() < 20:
        return dict(estimate=np.nan, ci=[np.nan, np.nan], n_near=int(near.sum()))

    def stat(s):
        nm = s["dist_um"] < NEAR_UM[1]
        a = _own_fit(_sel(s, nm), np.array(near_bins))
        b = _own_fit(_sel(s, ~nm), np.array(far_bins))
        return np.array([a - b, a, b])

    est, _, vals = _boot(t, stat, seed)
    vals = np.array(vals)
    pc = lambda k: [float(np.nanpercentile(vals[:, k], 2.5)), float(np.nanpercentile(vals[:, k], 97.5))]
    return dict(estimate=float(est[0]), ci=pc(0), beta_near=float(est[1]), beta_near_ci=pc(1), beta_far=float(est[2]), beta_far_ci=pc(2),
                n_near=int(near.sum()), n_far=int(far.sum()), n_animals=int(len(np.unique(t["animal"]))))


# ---------------------------------------------------------------------------------------------------------------------
# T15 evoked ROI-profile decomposition (contamination profile vs receiver's own static profile)
# ---------------------------------------------------------------------------------------------------------------------
def t15_roi_profile(rows: dict, seed: int = 0) -> dict:
    m = (rows["target_z"] > Z_RESP) & (rows["dist_um"] < NEAR_UM[1]) & (rows["dC_t"] > 0) & np.isfinite(rows["noise_n"])
    t = _sel(rows, m)
    if len(t["animal"]) < 20:
        return dict(estimate=np.nan, ci=[np.nan, np.nan], n_rows=int(len(t["animal"])))

    def stat(s):
        # stacked regression: y_{row,r} = a * kappa_r*dC_t + b_row * own_r ; b_row profiled out per row
        num = den = 0.0
        for k in range(len(s["animal"])):
            y = s["roi_dC"][k]
            pc = s["roi_kappa"][k] * s["dC_t"][k]
            po = s["roi_own"][k]
            if not (np.all(np.isfinite(y)) and np.all(np.isfinite(pc)) and np.all(np.isfinite(po))) or np.dot(po, po) <= 0:
                continue
            # residualise on own profile (and constant), then accumulate the slope on the contamination profile
            Z = np.stack([po, np.ones_like(po)], 1)
            H = Z @ np.linalg.pinv(Z)
            ry, rc = y - H @ y, pc - H @ pc
            w = 1.0 / max(s["noise_n"][k], 1.0) ** 2
            num += w * float(rc @ ry)
            den += w * float(rc @ rc)
        return num / den if den > 0 else np.nan

    est, ci, _ = _boot(t, stat, seed)
    return dict(estimate=float(est), ci=ci, n_rows=int(len(t["animal"])), n_animals=int(len(np.unique(t["animal"]))),
                note="a = 1 when the receiver ROI profile is the optical copy predicted from the static light model; a = 0 when it follows the receiver's own static profile")


# ---------------------------------------------------------------------------------------------------------------------
# T16 spontaneous copy ratio (lag-1 cross-covariance, shot noise and frame noise immune)
# ---------------------------------------------------------------------------------------------------------------------
def spontaneous_mask(prov: dict, excl_before_s: float = 3.0, excl_after_s: float = 20.0) -> np.ndarray:
    vt = prov["vol_time"]
    ok = np.isfinite(prov["rois"][:, 0, 0])
    for e in prov["meta"]["events"]:
        if e.get("kind") == "stim":
            ok &= ~((vt >= e["t_on"] - excl_before_s) & (vt <= e["t_on"] + excl_after_s))
    return ok


def _highpass(X, win=61):
    base = ndi.median_filter(np.where(np.isfinite(X), X, np.nanmedian(X, 0)), size=(win, 1), mode="nearest")
    return X - base


def spontaneous_pairs(ap: AnimalProvenance, *, max_um: float = FAR_UM[1], frame_correct: bool = True) -> dict:
    X = box_counts(ap.prov)
    if frame_correct:
        X = X - frame_common(ap.prov)
    H = _highpass(X)
    ok = spontaneous_mask(ap.prov)
    seg = ok[:-1] & ok[1:]
    A, Bn = H[:-1][seg], H[1:][seg]
    A = np.where(np.isfinite(A), A, np.nanmean(A, 0))
    Bn = np.where(np.isfinite(Bn), Bn, np.nanmean(Bn, 0))
    A = A - A.mean(0)
    Bn = Bn - Bn.mean(0)
    n = len(A)
    C1 = (A.T @ Bn + Bn.T @ A) / (2 * max(n, 1))       # symmetric lag-1 cross-covariance
    C0 = (A.T @ A) / max(n, 1)
    auto1 = np.diag(C1)
    tg = sorted({int(e["target_nucleus"]) for e in event_windows(ap.prov)})
    rows = {k: [] for k in ("animal", "target", "receiver", "dist_um", "cov1_obs", "cov1_pred_h1", "cov0_obs", "overlap", "var_t", "var_r")}
    K = ap.kap["kappa"]["box"]
    ov = RX.overlap_fraction(ap.prov["nuclei"], ap.prov["mean_volume"].shape).toarray()
    for t in tg:
        d = ap.kap["dist"][:, t]
        for r in np.where((d <= max_um) & (np.arange(len(d)) != t))[0]:
            rows["animal"].append(ap.animal); rows["target"].append(t); rows["receiver"].append(int(r)); rows["dist_um"].append(float(d[r]))
            rows["cov1_obs"].append(float(C1[t, r]))
            rows["cov1_pred_h1"].append(float(K[r, t] * auto1[t] + K[t, r] * auto1[r]))
            rows["cov0_obs"].append(float(C0[t, r])); rows["overlap"].append(float(ov[r, t]))
            rows["var_t"].append(float(C0[t, t])); rows["var_r"].append(float(C0[r, r]))
    out = {k: np.array(v) for k, v in rows.items()}
    out["n_samples"] = np.full(len(out["animal"]), n)
    return out


def t16_spontaneous_excess(sp: dict, seed: int = 0) -> dict:
    """Standardised excess of spontaneous lag-1 cross-covariance over the optical (H1) prediction, near (0-6 um) minus
    far (15-40 um); plus the inverse-variance weighted slope of observed on predicted covariance (1 under H1)."""
    se = np.sqrt(np.maximum(sp["var_t"] * sp["var_r"], 1e-12) / np.maximum(sp["n_samples"], 1))
    zx = (sp["cov1_obs"] - sp["cov1_pred_h1"]) / se
    d = dict(sp, zx=zx, se=se)
    near = d["dist_um"] < NEAR_UM[1]
    far = (d["dist_um"] >= FAR_UM[0]) & (d["dist_um"] < FAR_UM[1])
    t = _sel(d, near | far)
    if near.sum() < 10:
        return dict(estimate=np.nan, ci=[np.nan, np.nan], n_near=int(near.sum()))

    def stat(s):
        nm = s["dist_um"] < NEAR_UM[1]
        a = float(np.mean(s["zx"][nm])) if nm.any() else np.nan
        b = float(np.mean(s["zx"][~nm])) if (~nm).any() else np.nan
        w = 1.0 / s["se"][nm] ** 2
        den = np.sum(w * s["cov1_pred_h1"][nm] ** 2)
        slope = float(np.sum(w * s["cov1_pred_h1"][nm] * s["cov1_obs"][nm]) / den) if den > 0 else np.nan
        return np.array([a - b, a, b, slope])

    est, _, vals = _boot(t, stat, seed)
    vals = np.array(vals)
    pc = lambda k: [float(np.nanpercentile(vals[:, k], 2.5)), float(np.nanpercentile(vals[:, k], 97.5))]
    return dict(estimate=float(est[0]), ci=pc(0), z_near=float(est[1]), z_far=float(est[2]), slope_obs_on_h1=float(est[3]), slope_ci=pc(3),
                n_near=int(near.sum()), n_far=int(far.sum()), n_animals=int(len(np.unique(t["animal"]))))


# ---------------------------------------------------------------------------------------------------------------------
# W descriptives
# ---------------------------------------------------------------------------------------------------------------------
def w9_spatial_copy_curve(rows: dict, bins=(0, 2, 3, 4, 5, 6, 8, 10, 15, 25, 40)) -> list[dict]:
    m = (rows["target_z"] > Z_RESP) & (rows["dC_t"] > 0)
    t = _sel(rows, m)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (t["dist_um"] >= lo) & (t["dist_um"] < hi)
        if s.sum() < 5:
            continue
        obs = float(np.sum(t["dC_n"][s]) / np.sum(t["dC_t"][s]))
        pred = float(np.sum(t["kappa"][s] * t["dC_t"][s]) / np.sum(t["dC_t"][s]))
        out.append(dict(shell=[lo, hi], n=int(s.sum()), observed_copy=obs, kappa_predicted_copy=pred,
                        resp_rate=float(t["resp"][s].mean()), resp_rate_resid=float(t["resp_resid"][s].mean())))
    return out


def w7_amplitude_linearity(rows: dict) -> dict:
    m = (rows["target_z"] > 1.0) & (rows["dist_um"] < NEAR_UM[1]) & np.isfinite(rows["noise_n"])
    t = _sel(rows, m)
    if len(t["animal"]) < 30:
        return dict(n=int(len(t["animal"])))
    x = t["dC_t"] / np.nanmedian(np.abs(t["dC_t"]))
    y = (t["dC_n"] - t["kappa"] * t["dC_t"]) / np.maximum(t["noise_n"], 1.0)
    X = np.stack([np.ones_like(x), x, x ** 2], 1)
    ok = np.all(np.isfinite(X), 1) & np.isfinite(y)
    coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return dict(n=int(ok.sum()), intercept=float(coef[0]), linear=float(coef[1]), quadratic=float(coef[2]))


def w2_alpha_stability(rows: dict, min_events: int = 3) -> dict:
    m = (rows["target_z"] > Z_RESP) & (rows["dist_um"] < NEAR_UM[1]) & (rows["dC_t"] > 0)
    t = _sel(rows, m)
    key = np.char.add(np.char.add(t["animal"].astype(str), "|"), np.char.add(t["target"].astype(str), np.char.add("|", t["receiver"].astype(str))))
    cvs, ratios = [], []
    for k in np.unique(key):
        s = key == k
        if s.sum() < min_events:
            continue
        a = t["dC_n"][s] / t["dC_t"][s]
        cvs.append(float(np.std(a) / max(abs(np.mean(a)), 1e-6)))
        if t["kappa"][s][0] > 0.01:
            ratios.append(float(np.median(a) / t["kappa"][s][0]))
    return dict(n_pairs=len(cvs), median_cv=float(np.median(cvs)) if cvs else np.nan, median_alpha_over_kappa=float(np.median(ratios)) if ratios else np.nan,
                n_pairs_kappa=len(ratios))


def w5_lag(ap: AnimalProvenance, rows: dict, lags_s=np.arange(-0.4, 0.41, 0.02), use_plane_times: bool = True) -> dict:
    """Pooled lag between receiver and target evoked time courses on target-up events, near receivers, using
    per-plane acquisition times (use_plane_times=False: naive volume times, the scan-timing error the correction removes)."""
    X = box_counts(ap.prov)
    NT = nucleus_times(ap.prov) if use_plane_times else np.repeat(ap.prov["vol_time"][:, None], X.shape[1], 1)
    m = (rows["target_z"] > Z_RESP) & (rows["dist_um"] < NEAR_UM[1]) & (rows["animal"] == ap.animal) & (rows["kappa"] > 0.02)
    evs = {int(e["event"]): e for e in event_windows(ap.prov)}
    score = np.zeros(len(lags_s))
    n = 0
    for ev, t, r in zip(rows["event"][m], rows["target"][m], rows["receiver"][m]):
        e = evs[int(ev)]
        v = int(e["vol"])
        sl = slice(v - PRE_VOLS, v + POST_VOLS)
        tt, xt = NT[sl, t], X[sl, t]
        tr, xr = NT[sl, r], X[sl, r]
        ok = np.isfinite(xt) & np.isfinite(xr)
        if ok.sum() < 12:
            continue
        xt_n = (xt - np.nanmean(xt[tt < e["t_on"]])) if np.any(tt < e["t_on"]) else xt - np.nanmean(xt)
        xr_n = (xr - np.nanmean(xr[tr < e["t_on"]])) if np.any(tr < e["t_on"]) else xr - np.nanmean(xr)
        for k, L in enumerate(lags_s):
            pred = np.interp(tr[ok] - L, tt[ok], xt_n[ok])
            c = np.corrcoef(pred, xr_n[ok])[0, 1]
            if np.isfinite(c):
                score[k] += c
        n += 1
    best = float(lags_s[int(np.argmax(score))]) if n else np.nan
    return dict(animal=ap.animal, n_events=n, best_lag_s=best, score=score.tolist(), lags_s=[float(x) for x in lags_s], plane_times=use_plane_times)


def band_transfer(ap: AnimalProvenance, max_um: float = NEAR_UM[1], bands=((0.0, 0.02), (0.02, 0.2), (0.2, 1.0))) -> dict:
    """Spontaneous-period cross-spectral gain (receiver on target) per band, pooled over near pairs, normalised by kappa."""
    X = box_counts(ap.prov) - frame_common(ap.prov)
    ok = spontaneous_mask(ap.prov)
    # longest contiguous spontaneous run lengths are short; use all spontaneous samples with gaps set to zero after detrending
    Y = np.where(ok[:, None] & np.isfinite(X), X - np.nanmean(np.where(ok[:, None], X, np.nan), 0), 0.0)
    F = np.fft.rfft(Y, axis=0)
    fr = np.fft.rfftfreq(Y.shape[0], d=float(np.nanmedian(np.diff(ap.prov["vol_time"]))))
    tg = sorted({int(e["target_nucleus"]) for e in event_windows(ap.prov)})
    K = ap.kap["kappa"]["box"]
    out = []
    for lo, hi in bands:
        s = (fr > lo) & (fr <= hi)
        num = den = 0.0
        for t in tg:
            d = ap.kap["dist"][:, t]
            for r in np.where((d < max_um) & (np.arange(len(d)) != t))[0]:
                if K[r, t] < 0.02:
                    continue
                cross = np.sum(np.real(F[s, r] * np.conj(F[s, t])))
                pt = np.sum(np.abs(F[s, t]) ** 2)
                num += cross / K[r, t]
                den += pt
        out.append(dict(band_hz=[lo, hi], gain_over_kappa=float(num / den) if den > 0 else np.nan))
    return dict(animal=ap.animal, bands=out)


# =====================================================================================================================
# report
# =====================================================================================================================
@dataclass
class SignalProvenanceReport:
    partition: str
    animals: list
    families: dict = field(default_factory=dict)
    descriptives: dict = field(default_factory=dict)
    measurement: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def to_json(self) -> str:
        def conv(o):
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)
        return json.dumps(dict(partition=self.partition, animals=self.animals, families=self.families, descriptives=self.descriptives,
                               measurement=self.measurement, provenance=self.provenance), indent=1, default=conv)


def build_report(partition: str, animals: list[tuple[str, dict]], *, seed: int = 0, sigma_scale: float = 1.0, with_lag: bool = True,
                 input_hashes: dict | None = None) -> tuple[SignalProvenanceReport, dict]:
    aps, prow, sprow, lags, lags_naive, bands, meas = [], [], [], [], [], [], {}
    for name, prov in animals:
        ap = prepare_animal(name, prov, sigma_scale=sigma_scale)
        aps.append(ap)
        r = pair_rows(ap)
        prow.append(r)
        sprow.append(spontaneous_pairs(ap))
        meas[name] = dict(sigma_xy_px=ap.light["sigma_xy_px"], sigma_z_planes=ap.light["sigma_z_planes"], r2_support=ap.light["r2_support"],
                          n_isolated=ap.light["n_isolated"], n_nuclei=int(len(prov["nuclei"])), n_events=len(ap.amps["events"]))
        if with_lag:
            lags.append(w5_lag(ap, r))
            lags_naive.append(w5_lag(ap, r, use_plane_times=False))
        bands.append(band_transfer(ap))
    rows = concat_rows(prow)
    sp = concat_rows(sprow)
    rep = SignalProvenanceReport(partition, [a for a, _ in animals])
    rep.families = dict(
        T13_optical_copy_subtracted_proximity=t13_optical_copy_subtracted(rows, seed),
        T14_brightness_elasticity=t14_brightness_elasticity(rows, seed),
        T15_kappa_profile_calibration=t15_roi_profile(rows, seed),
        T16_spontaneous_excess=t16_spontaneous_excess(sp, seed),
    )
    rep.descriptives = dict(W9_spatial_copy_curve=w9_spatial_copy_curve(rows), W7_amplitude_linearity=w7_amplitude_linearity(rows),
                            W2_alpha_stability=w2_alpha_stability(rows), W5_lag_plane_times=lags, W5_lag_naive_volume_times=lags_naive,
                            W6_band_transfer=bands)
    rep.measurement = dict(per_animal=meas, sigma_scale=sigma_scale)
    rep.provenance = dict(module=VERSION, source_sha256=source_sha256(), roi_extract_sha256=hashlib.sha256(Path(RX.__file__).read_bytes()).hexdigest(),
                          seed=seed, inputs=input_hashes or {})
    return rep, dict(rows=rows, spont=sp)
