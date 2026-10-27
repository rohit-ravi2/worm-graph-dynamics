"""PQ08: within-pair near vs far contrast using Randi's own per-event features (closes run-1 T6b's detector gap).

Events: oracle-valid WT events with geometry, off-diagonal. For every labeled pair with >=1 near (d_box < cut) and
>=1 far (d_box >= cut) event: per-event 'extreme' = feature outside the null's central 95% (2.5/97.5 percentiles of
the fixed null used for q; dF/sigma and 2nd derivative separately, and either). Pair-level near and far extreme
rates; statistic = mean over pairs of (near - far) and pooled near/far ratio; 95% CI by pair bootstrap (2000,
seeded). Strata: all mixed pairs; pairs significant in the published map; pairs significant in our A map.
Usage: pq08_within_pair.py <null>
"""
import json, os, sys
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(__file__))
from pq04_reproduce_q import load_events, load_null
from pq02_features import FA

OUT = os.path.join(os.path.dirname(__file__), '..', 'out')


def main():
    null = sys.argv[1] if len(sys.argv) > 1 else 'wtqgt0.5'
    E, valid = load_events('oracle')
    d_box = np.load(f'{OUT}/events_geom.npz')['d_box']
    nd, ns = load_null(null)
    lo_d, hi_d = np.percentile(nd, [2.5, 97.5]); lo_s, hi_s = np.percentile(ns, [2.5, 97.5])
    with h5py.File(FA, 'r') as f:
        qpub = f['wt/q'][:]
    QA = np.load(f'{OUT}/pq06_{null}_maps.npz')['Q_A']
    m = valid & np.isfinite(d_box) & (E['ai'] != E['aj'])
    ext = {'dff': (E['dff'] < lo_d) | (E['dff'] > hi_d), 'sd': (E['sd'] < lo_s) | (E['sd'] > hi_s)}
    ext['either'] = ext['dff'] | ext['sd']
    key = E['ai'].astype(np.int64) * 300 + E['aj']
    rng = np.random.default_rng(8)
    res = dict(null=null, null_central95_dff=[float(lo_d), float(hi_d)], null_central95_sd=[float(lo_s), float(hi_s)])
    for cut in [2.5, 3.5]:
        near = m & (d_box < cut); far = m & (d_box >= cut)
        kn = np.unique(key[near]); kf = np.unique(key[far]); mixed = np.intersect1d(kn, kf)
        strata = {'all_mixed': mixed,
                  'published_sig': mixed[qpub[mixed // 300, mixed % 300] < 0.05],
                  'ours_A_sig': mixed[QA[mixed // 300, mixed % 300] < 0.05]}
        for sname, ks in strata.items():
            for feat, X in ext.items():
                sel_n = near & np.isin(key, ks); sel_f = far & np.isin(key, ks)
                # per pair rates
                order_n = np.searchsorted(ks, key[sel_n]); order_f = np.searchsorted(ks, key[sel_f])
                cn = np.bincount(order_n, minlength=len(ks)); xn = np.bincount(order_n, weights=X[sel_n], minlength=len(ks))
                cf = np.bincount(order_f, minlength=len(ks)); xf = np.bincount(order_f, weights=X[sel_f], minlength=len(ks))
                rn, rf = xn / cn, xf / cf
                diff = rn - rf
                bs_d, bs_r = [], []
                for _ in range(2000):
                    b = rng.integers(0, len(ks), len(ks))
                    bs_d.append(diff[b].mean()); bs_r.append(xn[b].sum() / cn[b].sum() / max(1e-12, xf[b].sum() / cf[b].sum()))
                res[f'cut{cut}_{sname}_{feat}'] = dict(
                    n_pairs=int(len(ks)), n_near_events=int(cn.sum()), n_far_events=int(cf.sum()),
                    pooled_near_rate=round(float(xn.sum() / cn.sum()), 4), pooled_far_rate=round(float(xf.sum() / cf.sum()), 4),
                    pooled_ratio=round(float(xn.sum() / cn.sum() / (xf.sum() / cf.sum())), 3),
                    pooled_ratio_ci95=[round(float(np.percentile(bs_r, 2.5)), 3), round(float(np.percentile(bs_r, 97.5)), 3)],
                    mean_pair_diff=round(float(diff.mean()), 4),
                    mean_pair_diff_ci95=[round(float(np.percentile(bs_d, 2.5)), 4), round(float(np.percentile(bs_d, 97.5)), 4)])
                print(cut, sname, feat, json.dumps(res[f'cut{cut}_{sname}_{feat}']), flush=True)
    json.dump(res, open(f'{OUT}/pq08_{null}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
