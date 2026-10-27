"""PQ06: near-excluded re-analysis with the reproduced Randi statistic.

Events: oracle-valid WT events (pq03/pq04). Geometry: per-animal d_box (pq05).
Regimes (fixed before running):
  A      all oracle-valid events (reproduction map)
  A_geo  events with per-animal geometry (drops 3 datasets without DANDI segmentation)
  B      A_geo minus events with d_box < 2.5
  C      A_geo minus events with d_box < 3.5
Power-matched controls for B and C (n_draws each, seeded):
  rand       drop the same NUMBER of events uniformly at random from A_geo
  pairmatch  within every pair drop the same number of events that the regime drops from that pair, chosen at
             random within the pair (distance-independent, per-pair event loss identical)
Null: fixed across all regimes and draws (argument; built once from A events, see pq04).
Evidence counts: q<0.05 (Storey, re-estimated per map) and p <= p* (p* = largest p with q<0.05 in A_geo, fixed).
Outputs: out/pq06_{null}.json, out/pq06_{null}_maps.npz (q maps for A/A_geo/B/C; packed sig/testable per draw).
Usage: pq06_regimes.py <null> [n_draws]
"""
import json, os, sys, time
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(__file__))
import pq_stats as S
from pq04_reproduce_q import load_events, load_null
from pq02_features import FA, IDS

OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
OFFD = ~np.eye(300, dtype=bool)


def summarize(Q, qpub, qA, P=None, pstar=None):
    sig = (Q < 0.05) & OFFD
    extra = {} if P is None else dict(n_p_leq_pstar_offdiag=int(((P <= pstar) & OFFD).sum()))
    return dict(**extra,n_testable_offdiag=int((np.isfinite(Q) & OFFD).sum()), n_sig_offdiag=int(sig.sum()),
                n_sig_in_published=int((sig & (qpub < 0.05)).sum()), n_sig_in_A=int((sig & (qA < 0.05)).sum()))


def main():
    null = sys.argv[1] if len(sys.argv) > 1 else 'wtqgt0.5'
    n_draws = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    t0 = time.time()
    E, valid = load_events('oracle')
    d_box = np.load(f'{OUT}/events_geom.npz')['d_box']
    nd, ns = load_null(null)
    with h5py.File(FA, 'r') as f:
        qpub = f['wt/q'][:]
    T = S.PairTester(nd, ns)
    base = valid & np.isfinite(d_box)
    masks = {'A': valid, 'A_geo': base, 'B': base & (d_box >= 2.5), 'C': base & (d_box >= 3.5)}
    maps, pmaps, res = {}, {}, dict(null=null, n_draws=n_draws, null_n=int(len(nd)))
    for name, m in masks.items():
        Q, pi0, P = T.q(E['ai'][m], E['aj'][m], E['dff'][m], E['sd'][m])
        maps[name] = Q; pmaps[name] = P
        res[name] = dict(n_events=int(m.sum()), pi0=pi0)
        print(name, m.sum(), flush=True)
    # fixed per-pair evidence threshold: the largest p declared significant (q<0.05) in A_geo. Counting p <= pstar
    # separates loss of per-pair evidence from Storey re-ranking (q depends on every other pair's p and on pi0).
    pstar = float(np.nanmax(pmaps['A_geo'][maps['A_geo'] < 0.05]))
    res['pstar_from_A_geo'] = pstar
    for name in masks:
        res[name].update(summarize(maps[name], qpub, maps['A'], pmaps[name], pstar))
        res[name]['n_sig_in_A_geo'] = int(((maps[name] < 0.05) & OFFD & (maps['A_geo'] < 0.05)).sum())
    # survivors
    for name in ['B', 'C']:
        sig = (maps[name] < 0.05) & OFFD
        res[name]['survivors'] = sorted([f'{IDS[i]}<-{IDS[j]}' for i, j in zip(*np.where(sig))])
        res[name]['A_geo_sig_lost'] = int(((maps['A_geo'] < 0.05) & OFFD & ~sig).sum())
        psig = (pmaps[name] <= pstar) & OFFD
        res[name]['survivors_p_leq_pstar'] = sorted([f'{IDS[i]}<-{IDS[j]}' for i, j in zip(*np.where(psig))])
    # controls
    pair_key = E['ai'].astype(np.int64) * 300 + E['aj']
    bi = np.where(base)[0]
    packed = {}
    for name in ['B', 'C']:
        drop = base & ~masks[name]
        n_drop = int(drop.sum())
        # per-pair drop counts
        uk, cnt = np.unique(pair_key[drop], return_counts=True)
        drop_count = dict(zip(uk.tolist(), cnt.tolist()))
        by_pair = {}
        for e in bi:
            by_pair.setdefault(int(pair_key[e]), []).append(e)
        for ctrl in ['rand', 'pairmatch']:
            rows, sigs, tests, psigs = [], [], [], []
            for d in range(n_draws):
                rng = np.random.default_rng([20260916, ord(name), len(ctrl), d])
                keep = base.copy()
                if ctrl == 'rand':
                    keep[rng.choice(bi, n_drop, replace=False)] = False
                else:
                    for k, c in drop_count.items():
                        keep[rng.choice(by_pair[k], c, replace=False)] = False
                Q, pi0, P = T.q(E['ai'][keep], E['aj'][keep], E['dff'][keep], E['sd'][keep])
                s = summarize(Q, qpub, maps['A'], P, pstar); s['pi0'] = pi0
                rows.append(s)
                sigs.append(np.packbits(((Q < 0.05) & OFFD).ravel())); tests.append(np.packbits(np.isfinite(Q).ravel()))
                psigs.append(np.packbits(((P <= pstar) & OFFD).ravel()))
            vp = np.array([r['n_p_leq_pstar_offdiag'] for r in rows])
            v = np.array([r['n_sig_offdiag'] for r in rows])
            res[f'{name}_ctrl_{ctrl}'] = dict(n_events_dropped=n_drop, n_sig_offdiag_mean=round(float(v.mean()), 2),
                                              n_sig_offdiag_p2_5=float(np.percentile(v, 2.5)), n_sig_offdiag_p97_5=float(np.percentile(v, 97.5)),
                                              n_sig_offdiag_min=int(v.min()), n_sig_offdiag_max=int(v.max()),
                                              frac_draws_leq_regime=round(float(np.mean(v <= res[name]['n_sig_offdiag'])), 4),
                                              n_testable_mean=round(float(np.mean([r['n_testable_offdiag'] for r in rows])), 1),
                                              n_p_leq_pstar_mean=round(float(vp.mean()), 2), n_p_leq_pstar_p2_5=float(np.percentile(vp, 2.5)),
                                              n_p_leq_pstar_p97_5=float(np.percentile(vp, 97.5)),
                                              frac_draws_p_leq_regime=round(float(np.mean(vp <= res[name]['n_p_leq_pstar_offdiag'])), 4),
                                              draws=rows)
            packed[f'{name}_{ctrl}_sig'] = np.array(sigs); packed[f'{name}_{ctrl}_testable'] = np.array(tests)
            packed[f'{name}_{ctrl}_psig'] = np.array(psigs)
            print(name, ctrl, res[f'{name}_ctrl_{ctrl}']['n_sig_offdiag_mean'], res[name]['n_sig_offdiag'], flush=True)
    T.close()
    res['seconds'] = round(time.time() - t0, 1)
    np.savez_compressed(f'{OUT}/pq06_{null}_maps.npz', **{f'Q_{k}': v for k, v in maps.items()}, **{f'P_{k}': v for k, v in pmaps.items()}, **packed)
    json.dump(res, open(f'{OUT}/pq06_{null}.json', 'w'), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ('draws', 'survivors', 'survivors_p_leq_pstar')}) for k, v in res.items()}, indent=1))


if __name__ == '__main__':
    main()
