"""Event-level response table: one row per (animal, stim, ROI).

Windows (volumes ~0.5 s): pre = [v-10, v), early = [v, v+4) (0-2 s), full = [v, v+20) (0-10 s).
Amplitudes: dFF_* = mean(post)/mean(pre) - 1 for green and red; dF_abs_g = mean(post) - mean(pre)
(raw fluorescence units, for the photon-bleed scaling test).
Robust null: per ROI, z = (amp - median over all stims in that animal) / (1.4826 * MAD); most stims
do not drive a given ROI, so the across-stim distribution is mostly null.
Timing (0-10 s window): latency_vol = first volume reaching half the window max of dFF_g (signed by
the dominant deflection); xcorr of the responder's dFF_g trace [v-4, v+20) with the target's,
max over lags -4..+4 volumes (positive lag = responder trails target).
Geometry: dx, dy from the stim-spot pixel; dz from the target ROI centroid (NaN when the target ROI
is unknown); d_xy_target / d3_units from the target centroid where known.
"""
import glob, json, os
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
PRE, EARLY, FULL, TR0 = 10, 4, 20, 4


def nanmean_win(X, a, b):
    a = max(a, 0); b = min(b, len(X))
    if b <= a:
        return np.full(X.shape[1], np.nan)
    with np.errstate(all='ignore'):
        return np.nanmean(X[a:b], axis=0)


def robust_z(A):
    med = np.nanmedian(A, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(A - med), axis=0)
    with np.errstate(all='ignore'):
        return (A - med) / mad


def animal_table(f):
    d = np.load(f)
    G, R, C = d['G'], d['R'], d['C'].astype(float)
    sv, stg, spot = d['stim_vol'], d['stim_target'], d['spot_xy']
    lab = d['roi_label']
    nS, nN = len(sv), G.shape[1]
    amp = {k: np.full((nS, nN), np.nan) for k in
           ['dFF_g_early', 'dFF_g_full', 'dFF_r_early', 'dFF_r_full', 'dF_abs_g_early', 'F0_g', 'F0_r']}
    lat = np.full((nS, nN), np.nan); xc = np.full((nS, nN), np.nan); xlag = np.full((nS, nN), np.nan)
    for s, v in enumerate(sv):
        g0, r0 = nanmean_win(G, v - PRE, v), nanmean_win(R, v - PRE, v)
        ge, gf = nanmean_win(G, v, v + EARLY), nanmean_win(G, v, v + FULL)
        re_, rf = nanmean_win(R, v, v + EARLY), nanmean_win(R, v, v + FULL)
        with np.errstate(all='ignore'):
            amp['dFF_g_early'][s] = ge / g0 - 1; amp['dFF_g_full'][s] = gf / g0 - 1
            amp['dFF_r_early'][s] = re_ / r0 - 1; amp['dFF_r_full'][s] = rf / r0 - 1
            amp['dF_abs_g_early'][s] = ge - g0
        amp['F0_g'][s] = g0; amp['F0_r'][s] = r0
        a, b = v - TR0, v + FULL
        if a < 0 or b > len(G):
            continue
        with np.errstate(all='ignore'):
            tr = G[a:b] / g0 - 1                        # (TR0+FULL) x N
        post = tr[TR0:]
        sgn = np.sign(np.nansum(post, axis=0))
        sp = post * sgn
        mx = np.nanmax(sp, axis=0)
        with np.errstate(all='ignore'):
            reach = sp >= 0.5 * mx
        reach[:, ~np.isfinite(mx) | (mx <= 0)] = False
        has = reach.any(axis=0)
        lat[s, has] = np.argmax(reach[:, has], axis=0)
        t = stg[s]
        if t >= 0 and t < nN and np.isfinite(tr[:, t]).sum() > 10:
            x = tr[:, t]
            best = np.full(nN, -np.inf); bl = np.zeros(nN)
            for L in range(-4, 5):
                # responder at time k vs target at k-L  (L>0: responder trails)
                if L >= 0:
                    xs, ys = x[:len(x) - L], tr[L:]
                else:
                    xs, ys = x[-L:], tr[:L]
                m = np.isfinite(xs)[:, None] & np.isfinite(ys)
                n = m.sum(0)
                xm = np.where(m, xs[:, None], np.nan); ym = np.where(m, ys, np.nan)
                with np.errstate(all='ignore'):
                    xz = xm - np.nanmean(xm, 0); yz = ym - np.nanmean(ym, 0)
                    r = np.nansum(xz * yz, 0) / np.sqrt(np.nansum(xz ** 2, 0) * np.nansum(yz ** 2, 0))
                r[n < 10] = np.nan
                upd = np.nan_to_num(r, nan=-np.inf) > best
                best[upd] = r[upd]; bl[upd] = L
            best[~np.isfinite(best)] = np.nan
            xc[s] = best; xlag[s] = bl
    z = {k.replace('dFF', 'z'): robust_z(amp[k]) for k in ['dFF_g_early', 'dFF_g_full', 'dFF_r_early', 'dFF_r_full']}
    # geometry
    tgt_c = np.full((nS, 3), np.nan)
    ok = (stg >= 0) & (stg < nN)
    tgt_c[ok] = C[stg[ok]]
    dx = C[None, :, 0] - spot[:, None, 0]; dy = C[None, :, 1] - spot[:, None, 1]
    dz = C[None, :, 2] - tgt_c[:, None, 2]
    tdx = C[None, :, 0] - tgt_c[:, None, 0]; tdy = C[None, :, 1] - tgt_c[:, None, 1]
    rows = dict(stim=np.repeat(np.arange(nS), nN), roi=np.tile(np.arange(nN), nS),
                target=np.repeat(stg, nN), is_target=(np.repeat(stg, nN) == np.tile(np.arange(nN), nS)),
                dx_spot=dx.ravel(), dy_spot=dy.ravel(), dz_target=dz.ravel(),
                dx_target=tdx.ravel(), dy_target=tdy.ravel(),
                latency_vol=lat.ravel(), xcorr_target=xc.ravel(), xcorr_lag=xlag.ravel())
    for k, v in amp.items():
        rows[k] = v.ravel()
    for k, v in z.items():
        rows[k] = v.ravel()
    rows['resp_label'] = np.tile(lab, nS)
    rows['target_label'] = np.array([lab[t] if 0 <= t < nN else '' for t in stg]).repeat(nN)
    return rows


def main():
    files = sorted(glob.glob(os.path.join(OUT, 'cache', 'sub-*.npz')), key=lambda p: int(p.split('sub-')[1].split('.')[0]))
    allrows = {}
    for i, f in enumerate(files):
        r = animal_table(f)
        r['animal'] = np.full(len(r['stim']), i)
        for k, v in r.items():
            allrows.setdefault(k, []).append(v)
        print(os.path.basename(f), len(r['stim']), flush=True)
    tab = {k: np.concatenate(v) for k, v in allrows.items()}
    np.savez_compressed(os.path.join(OUT, 'responses_wt.npz'), **tab,
                        animals=np.array([os.path.basename(f)[:-4] for f in files]))
    print('rows', len(tab['stim']))


if __name__ == '__main__':
    main()
