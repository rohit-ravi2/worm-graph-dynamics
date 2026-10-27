"""PQ09: composition of significant pairs by per-animal proximity, and where regime B/C losses come from.

For every off-diagonal pair significant in A_geo (q<0.05): fraction of its valid events with d_box < cut.
Categories: all_far (0), minority_near (0<f<=0.5), majority_near (0.5<f<1), all_near (1).
Per category: n pairs, n surviving the regime, mean survival frequency under pairmatch and rand controls.
Also the same composition for the published significant pairs (events = oracle-valid, geometry available).
Usage: pq09_composition.py <null>
"""
import json, os, sys
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(__file__))
from pq04_reproduce_q import load_events
from pq02_features import FA

OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
OFFD = ~np.eye(300, dtype=bool)


def main():
    null = sys.argv[1] if len(sys.argv) > 1 else 'wtqgt0.5'
    E, valid = load_events('oracle')
    d_box = np.load(f'{OUT}/events_geom.npz')['d_box']
    Z = np.load(f'{OUT}/pq06_{null}_maps.npz')
    with h5py.File(FA, 'r') as f:
        qpub = f['wt/q'][:]
    base = valid & np.isfinite(d_box)
    key = E['ai'].astype(np.int64) * 300 + E['aj']
    res = dict(null=null)
    for cut, reg in [(2.5, 'B'), (3.5, 'C')]:
        n_tot = np.bincount(key[base], minlength=90000)
        n_near = np.bincount(key[base & (d_box < cut)], minlength=90000)
        frac = np.where(n_tot > 0, n_near / np.maximum(n_tot, 1), np.nan).reshape(300, 300)
        pstar = json.load(open(f'{OUT}/pq06_{null}.json'))['pstar_from_A_geo']
        surv = (Z[f'Q_{reg}'] < 0.05)
        surv_p = (Z[f'P_{reg}'] <= pstar)
        ctrl = {}
        for c in ['pairmatch', 'rand']:
            ctrl[c] = np.mean([np.unpackbits(s)[:90000].reshape(300, 300) for s in Z[f'{reg}_{c}_sig']], axis=0)
            ctrl[c + '_p'] = np.mean([np.unpackbits(s)[:90000].reshape(300, 300) for s in Z[f'{reg}_{c}_psig']], axis=0)
        for mapname, sig in [('A_geo', (Z['Q_A_geo'] < 0.05) & OFFD), ('published', (qpub < 0.05) & OFFD & (n_tot.reshape(300, 300) > 0))]:
            cats = {'all_far': frac == 0, 'minority_near': (frac > 0) & (frac <= 0.5),
                    'majority_near': (frac > 0.5) & (frac < 1), 'all_near': frac == 1}
            d = {}
            for cname, cm in cats.items():
                m = sig & cm
                d[cname] = dict(n_pairs=int(m.sum()), n_survive_regime=int((m & surv).sum()),
                                mean_survival_pairmatch=round(float(ctrl['pairmatch'][m].mean()), 4) if m.any() else None,
                                mean_survival_rand=round(float(ctrl['rand'][m].mean()), 4) if m.any() else None,
                                n_p_leq_pstar_regime=int((m & surv_p).sum()),
                                mean_p_survival_pairmatch=round(float(ctrl['pairmatch_p'][m].mean()), 4) if m.any() else None,
                                mean_p_survival_rand=round(float(ctrl['rand_p'][m].mean()), 4) if m.any() else None)
            d['n_sig_total'] = int(sig.sum())
            d['median_frac_near_events'] = round(float(np.nanmedian(frac[sig])), 4)
            res[f'cut{cut}_{mapname}'] = d
            print(cut, mapname, json.dumps(d), flush=True)
    # event-level near fraction overall (valid events, off-diagonal)
    offd_ev = base & (E['ai'] != E['aj'])
    res['frac_valid_offdiag_events_dbox_lt_2.5'] = round(float(np.mean(d_box[offd_ev] < 2.5)), 4)
    res['frac_valid_offdiag_events_dbox_lt_3.5'] = round(float(np.mean(d_box[offd_ev] < 3.5)), 4)
    json.dump(res, open(f'{OUT}/pq09_{null}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
