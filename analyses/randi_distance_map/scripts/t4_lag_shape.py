"""T4 map: are near-field responses copies of the target (bleed), off-target activation, or propagation?
Per targeted stim: target response state (z_g_full of the target ROI). For non-target ROIs by xy
distance bin: P(up | target up) vs P(up | target flat); among responders with target up: xcorr with
target trace, best lag, latency difference, and |dF_abs_resp / dF_abs_target| (photon ratio), plus
responder baseline brightness F0 relative to target F0. Animal-bootstrap CIs on rates; medians+IQR.
"""
import json, os
import numpy as np
OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
T = np.load(os.path.join(OUT, 'responses_wt.npz'))
animal, stim = T['animal'], T['stim']
key = animal.astype(np.int64) * 100000 + stim
tgtrow = T['is_target']
# per-event target values
tk = key[tgtrow]
tz = dict(zip(tk, T['z_g_full'][tgtrow])); tlat = dict(zip(tk, T['latency_vol'][tgtrow]))
tdF = dict(zip(tk, T['dF_abs_g_early'][tgtrow])); tF0 = dict(zip(tk, T['F0_g'][tgtrow]))
tzr = dict(zip(tk, T['z_r_full'][tgtrow]))
nt = (T['target'] >= 0) & ~tgtrow
k = key[nt]
TZ = np.array([tz.get(x, np.nan) for x in k]); TLAT = np.array([tlat.get(x, np.nan) for x in k])
TDF = np.array([tdF.get(x, np.nan) for x in k]); TF0 = np.array([tF0.get(x, np.nan) for x in k])
TZR = np.array([tzr.get(x, np.nan) for x in k])
z = T['z_g_full'][nt]; dxy = np.hypot(T['dx_target'][nt], T['dy_target'][nt]); adz = np.abs(T['dz_target'][nt])
xc, lag, lat = T['xcorr_target'][nt], T['xcorr_lag'][nt], T['latency_vol'][nt]
dF, F0 = T['dF_abs_g_early'][nt], T['F0_g'][nt]
an = animal[nt]; A = np.unique(an); rng = np.random.default_rng(1)
BINS = [0, 10, 14, 20, 30, 45, 1000]


def rate(mask, stat, B=400):
    num = np.bincount(an[mask], weights=stat[mask].astype(float), minlength=A.max() + 1)
    den = np.bincount(an[mask], minlength=A.max() + 1).astype(float)
    bs = [num[s].sum() / max(den[s].sum(), 1) for s in (rng.choice(A, len(A)) for _ in range(B))]
    return round(float(num.sum() / max(den.sum(), 1)), 4), [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], int(den.sum())


def q(x):
    x = x[np.isfinite(x)]
    return [round(float(np.percentile(x, p)), 3) for p in (25, 50, 75)] + [int(len(x))] if len(x) else None


out = []
fin = np.isfinite(z) & np.isfinite(TZ)
for a, b in zip(BINS[:-1], BINS[1:]):
    m = fin & (dxy >= a) & (dxy < b)
    up_t, flat_t = m & (TZ > 3), m & (np.abs(TZ) < 1)
    r = dict(dxy=[a, b],
             P_up_given_target_up=rate(up_t, z > 3),
             P_up_given_target_flat=rate(flat_t, z > 3),
             P_up_given_target_flat_and_red_flat=rate(flat_t & (np.abs(TZR) < 1), z > 3))
    resp = up_t & (z > 3)
    r['responders_target_up'] = dict(
        xcorr_q=q(xc[resp]), lag_q=q(lag[resp]),
        frac_lag0=round(float(np.mean(lag[resp & np.isfinite(lag)] == 0)), 3),
        latency_minus_target_q=q(lat[resp] - TLAT[resp]),
        photon_ratio_q=q(np.abs(dF[resp] / TDF[resp])),
        F0_ratio_q=q(F0[resp] / TF0[resp]))
    resp_flat = flat_t & (z > 3)
    r['responders_target_flat'] = dict(xcorr_q=q(xc[resp_flat]), lag_q=q(lag[resp_flat]), latency_q=q(lat[resp_flat]))
    out.append(r)
    print(json.dumps(r))
json.dump(out, open(os.path.join(OUT, 't4_lag_shape.json'), 'w'), indent=1)
