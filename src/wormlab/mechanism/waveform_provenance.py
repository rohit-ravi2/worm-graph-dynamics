"""Waveform / residual provenance families (§6-16): W1 high-frequency target-specific residual transfer, W2/W8 copy
ratio across windows, W3 waveform shape identity (small preregistered metric set), W6 coherence and phase.

Decomposition per nucleus (provenance traces, counts):
    X = box - offset - frame_common                        (E: frame-common removed)
    H = X - running_median_61                              (A: slow removed)
    G = regression of H on the brain-wide median z-trace   (global component removed)
    L = regression on the pair's spatial neighbourhood mean (F: nuclei 8-20 um from both members)
    D = first difference of the residual on consecutive valid samples (D: high-frequency residual)
Windows: spontaneous (outside [-3, +20] s of any stimulus), pre-stimulus (v-8..v-1), evoked (0.25-6.5 s), post-quiet (+10..+20 s).

Contamination predicts a window-independent copy ratio (alpha constant across windows, ~kappa in counts where the
target's fluctuations are real light changes) and near-identical evoked waveforms; coupling through the receiver's own
indicator predicts biological-band (not detector-band) transfer, possibly window-dependent. Shot noise is never copied
optically (independent photons per voxel), so W1 is informative only relative to calibrated synthetic worlds.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import signal_provenance as SP

W3_GRID = np.arange(-4.0, 8.01, 0.25)


def _residuals(ap: SP.AnimalProvenance) -> np.ndarray:
    X = SP.box_counts(ap.prov) - SP.frame_common(ap.prov)
    fill = np.where(np.isfinite(X), X, np.nanmedian(X, 0))
    H = X - ndi.median_filter(fill, size=(61, 1), mode="nearest")
    sd = np.nanstd(H, 0)
    Z = H / np.where(sd > 0, sd, 1)
    g = np.nanmedian(Z, 1)
    g = np.where(np.isfinite(g), g, 0.0)
    gg = float(g @ g) or 1.0
    beta = np.nansum(np.where(np.isfinite(H), H, 0) * g[:, None], 0) / gg
    return H - g[:, None] * beta[None, :]


def _window_masks(prov: dict) -> dict:
    vt = prov["vol_time"]
    ok = np.isfinite(prov["rois"][:, 0, 0])
    spont = SP.spontaneous_mask(prov)
    pre = np.zeros(len(vt), bool)
    evk = np.zeros(len(vt), bool)
    quiet = np.zeros(len(vt), bool)
    for e in prov["meta"]["events"]:
        if e.get("kind") != "stim":
            continue
        v = int(e["vol"])
        pre[max(v - 8, 0):v] = True
        evk |= (vt >= e["t_on"] + 0.25) & (vt <= e["t_on"] + 6.5)
        quiet |= (vt >= e["t_on"] + 10) & (vt <= e["t_on"] + 20)
    return dict(spontaneous=spont & ok, pre=pre & ok, evoked=evk & ok, post_quiet=quiet & ok)


def _diff_pairs(R, mask):
    m = mask[1:] & mask[:-1]
    return R[1:][m] - R[:-1][m]


def w1_w2_w8_pairs(ap: SP.AnimalProvenance, near_um=6.0, far=(15.0, 40.0), n_shift=16, seed=0) -> dict:
    R = _residuals(ap)
    masks = _window_masks(ap.prov)
    tg = sorted({int(e["target_nucleus"]) for e in SP.event_windows(ap.prov)})
    Dd = {w: _diff_pairs(R, m) for w, m in masks.items()}
    rng = np.random.default_rng(seed)
    K = ap.kap["kappa"]["box"]
    dist = ap.kap["dist"]
    rows = []
    for t in tg:
        cand = np.where(((dist[:, t] < near_um) | ((dist[:, t] >= far[0]) & (dist[:, t] < far[1]))) & (np.arange(len(dist)) != t))[0]
        if len(cand) > 40:
            near_c = cand[dist[cand, t] < near_um]
            far_c = cand[dist[cand, t] >= far[0]]
            far_c = rng.choice(far_c, min(len(far_c), 40 - min(len(near_c), 40)), replace=False) if len(far_c) else far_c
            cand = np.concatenate([near_c, far_c]).astype(int)
        for n in cand:
            # spatial-neighbourhood component: nuclei 8-20 um from both members
            nb = np.where((dist[:, t] >= 8) & (dist[:, t] <= 20) & (dist[:, n] >= 8) & (dist[:, n] <= 20))[0]
            rec = dict(target=int(t), receiver=int(n), dist_um=float(dist[n, t]), kappa=float(K[n, t]))
            for w, D in Dd.items():
                if len(D) < 20:
                    continue
                dt, dn = D[:, t].copy(), D[:, n].copy()
                ok = np.isfinite(dt) & np.isfinite(dn)
                if nb.size:
                    L = np.nanmean(D[:, nb], 1)
                    ok &= np.isfinite(L)
                    for arr in (dt, dn):
                        ll = L[ok]
                        c = float(np.dot(arr[ok], ll) / max(np.dot(ll, ll), 1e-9))
                        arr[ok] = arr[ok] - c * ll
                a, b = dt[ok], dn[ok]
                if len(a) < 20 or np.dot(a, a) <= 0:
                    continue
                slope = float(np.dot(a, b) / np.dot(a, a))
                rec[f"alpha_{w}"] = slope
                if w == "spontaneous":
                    null = []
                    for s in rng.integers(15, max(len(a) - 15, 16), n_shift):
                        null.append(float(np.dot(np.roll(a, s), b) / np.dot(a, a)))
                    null = np.array(null)
                    rec["w1_z"] = float((slope - null.mean()) / (null.std() + 1e-12))
                    rec["w1_partial_r"] = float(np.corrcoef(a, b)[0, 1])
            rows.append(rec)
    return dict(animal=ap.animal, rows=rows)


def w1_summary(per_animal: list[dict], seed=0) -> dict:
    """Near (<6 um) minus far (15-40 um) mean W1 z; alpha/kappa medians per window for near pairs with kappa > 0.02."""
    A, Z, NEAR = [], [], []
    alph = {w: [] for w in ("spontaneous", "pre", "evoked", "post_quiet")}
    for pa in per_animal:
        for r in pa["rows"]:
            if "w1_z" not in r:
                continue
            A.append(pa["animal"]); Z.append(r["w1_z"]); NEAR.append(r["dist_um"] < 6)
            if r["dist_um"] < 6 and r["kappa"] > 0.02:
                for w in alph:
                    if f"alpha_{w}" in r:
                        alph[w].append(r[f"alpha_{w}"] / r["kappa"])
    if not A:
        return dict(n=0)
    t = dict(animal=np.array(A), z=np.array(Z), near=np.array(NEAR))
    stat = lambda s: float(s["z"][s["near"]].mean() - s["z"][~s["near"]].mean()) if s["near"].any() and (~s["near"]).any() else np.nan
    est, ci, _ = SP._boot(t, stat, seed, n=200)
    return dict(estimate=est, ci=ci, near_mean_z=float(t["z"][t["near"]].mean()) if t["near"].any() else None,
                far_mean_z=float(t["z"][~t["near"]].mean()) if (~t["near"]).any() else None, n_near=int(t["near"].sum()), n_far=int((~t["near"]).sum()),
                alpha_over_kappa_median={w: (float(np.median(v)) if v else None) for w, v in alph.items()},
                alpha_over_kappa_n={w: len(v) for w, v in alph.items()},
                evoked_over_spontaneous_alpha_ratio=(float(np.median(alph["evoked"]) / np.median(alph["spontaneous"]))
                                                     if alph["evoked"] and alph["spontaneous"] and np.median(alph["spontaneous"]) != 0 else None))


def _rise_decay(w: np.ndarray):
    post = W3_GRID >= 0
    y = w[post]
    tt = W3_GRID[post]
    if not np.any(np.isfinite(y)) or np.nanmax(y) <= 0:
        return np.nan, np.nan
    pk = int(np.nanargmax(y))
    mx = y[pk]
    r10 = np.argmax(y[:pk + 1] >= 0.1 * mx) if pk > 0 else 0
    r90 = np.argmax(y[:pk + 1] >= 0.9 * mx) if pk > 0 else 0
    rise = tt[r90] - tt[r10]
    after = np.where(y[pk:] <= 0.5 * mx)[0]
    decay = tt[pk + after[0]] - tt[pk] if len(after) else np.nan
    return float(rise), float(decay)


def w3_waveforms(ap: SP.AnimalProvenance, rows: dict, min_events: int = 1) -> list[dict]:
    """Preregistered W3 metrics per (target, receiver) pair over TARGET-UP events: normalised cross-correlation of the
    event-averaged waveforms, derivative correlation, rise (10-90%) and half-decay time ratios receiver/target; and the
    target's own odd/even split-half waveform correlation as the reliability ceiling."""
    X = SP.box_counts(ap.prov)
    NT = SP.nucleus_times(ap.prov)
    evs = {int(e["event"]): e for e in SP.event_windows(ap.prov)}
    m = (rows["animal"] == ap.animal) & (rows["target_z"] > SP.Z_RESP) & (rows["dist_um"] < 15)
    pairs = {}
    for ev, t, n, d, rz in zip(rows["event"][m], rows["target"][m], rows["receiver"][m], rows["dist_um"][m], rows["resp_z"][m]):
        pairs.setdefault((int(t), int(n)), []).append((int(ev), float(d), float(rz)))
    out = []

    def trace(idx, e):
        v = int(e["vol"])
        sl = slice(v - SP.PRE_VOLS, v + SP.POST_VOLS)
        tt, xx = NT[sl, idx] - e["t_on"], X[sl, idx]
        ok = np.isfinite(tt) & np.isfinite(xx)
        if ok.sum() < 10:
            return None
        base = np.nanmean(xx[ok & (tt < 0)]) if np.any(ok & (tt < 0)) else np.nanmean(xx[ok])
        return np.interp(W3_GRID, tt[ok], xx[ok] - base, left=np.nan, right=np.nan)

    for (t, n), lst in pairs.items():
        if len(lst) < min_events:
            continue
        wt, wn = [], []
        for ev, d, rz in lst:
            a, b = trace(t, evs[ev]), trace(n, evs[ev])
            if a is not None and b is not None:
                wt.append(a); wn.append(b)
        if len(wt) < min_events:
            continue
        WT, WN = np.nanmean(wt, 0), np.nanmean(wn, 0)
        ok = np.isfinite(WT) & np.isfinite(WN)
        if ok.sum() < 20 or np.nanstd(WT[ok]) == 0 or np.nanstd(WN[ok]) == 0:
            continue
        ncc = float(np.corrcoef(WT[ok], WN[ok])[0, 1])
        dcc = float(np.corrcoef(np.diff(WT[ok]), np.diff(WN[ok]))[0, 1])
        rt, dt_ = _rise_decay(WT)
        rn, dn_ = _rise_decay(WN)
        half = None
        if len(wt) >= 4:
            o, e2 = np.nanmean(wt[::2], 0), np.nanmean(wt[1::2], 0)
            k = np.isfinite(o) & np.isfinite(e2)
            if k.sum() > 20 and np.std(o[k]) > 0 and np.std(e2[k]) > 0:
                half = float(np.corrcoef(o[k], e2[k])[0, 1])
        out.append(dict(animal=ap.animal, target=t, receiver=n, dist_um=lst[0][1], n_events=len(wt), mean_resp_z=float(np.mean([x[2] for x in lst])),
                        ncc=ncc, deriv_cc=dcc, rise_ratio=(rn / rt) if rt and np.isfinite(rn) and rt > 0 else None,
                        decay_ratio=(dn_ / dt_) if dt_ and np.isfinite(dn_) and dt_ > 0 else None, target_split_half=half))
    return out


def w3_summary(pairs: list[dict]) -> dict:
    def med(x):
        x = [v for v in x if v is not None and np.isfinite(v)]
        return (float(np.median(x)), len(x)) if x else (None, 0)
    near_resp = [p for p in pairs if p["dist_um"] < 6 and p["mean_resp_z"] > SP.Z_RESP]
    near_non = [p for p in pairs if p["dist_um"] < 6 and p["mean_resp_z"] <= SP.Z_RESP]
    mid = [p for p in pairs if 9 <= p["dist_um"] < 15]
    out = {}
    for name, grp in (("near_responders", near_resp), ("near_nonresponders", near_non), ("mid", mid)):
        out[name] = {k: med([p[k] for p in grp]) for k in ("ncc", "deriv_cc", "rise_ratio", "decay_ratio", "target_split_half")}
    out["closes"] = "event-averaged waveform shape identity (preregistered set: ncc, derivative cc, rise and half-decay ratios; ceiling = target split-half)"
    return out


def w6_coherence(ap: SP.AnimalProvenance, near_um=6.0, bands=((0.0, 0.02), (0.02, 0.2), (0.2, 1.0))) -> dict:
    """Spontaneous-period magnitude-squared coherence and phase per band, pooled cross-spectra over near and far pairs."""
    R = _residuals(ap)
    ok = SP.spontaneous_mask(ap.prov)
    Y = np.where(ok[:, None] & np.isfinite(R), R, 0.0)
    F = np.fft.rfft(Y, axis=0)
    fr = np.fft.rfftfreq(Y.shape[0], d=float(np.nanmedian(np.diff(ap.prov["vol_time"]))))
    tg = sorted({int(e["target_nucleus"]) for e in SP.event_windows(ap.prov)})
    dist = ap.kap["dist"]
    out = {}
    for grp, sel in (("near", lambda d: d < near_um), ("far", lambda d: (d >= 15) & (d < 40))):
        res = []
        for lo, hi in bands:
            s = (fr > lo) & (fr <= hi)
            sxy = 0j; sxx = syy = 0.0
            for t in tg:
                for n in np.where(sel(dist[:, t]) & (np.arange(len(dist)) != t))[0][:40]:
                    sxy += np.sum(F[s, n] * np.conj(F[s, t]))
                    sxx += np.sum(np.abs(F[s, t]) ** 2)
                    syy += np.sum(np.abs(F[s, n]) ** 2)
            coh = float(np.abs(sxy) ** 2 / (sxx * syy)) if sxx > 0 and syy > 0 else np.nan
            res.append(dict(band_hz=[lo, hi], coherence=coh, phase_rad=float(np.angle(sxy)) if sxy != 0 else np.nan))
        out[grp] = res
    return dict(animal=ap.animal, **out)
