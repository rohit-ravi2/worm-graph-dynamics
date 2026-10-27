"""PQ04: reproduce funatlas.h5 wt/q (and q_eq) from per-trial data; report agreement.

Event membership variants:
  oracle  = events whose dF/F exactly matches a published dFF_all entry (pq03): the published autoresponse
            inclusion + nan check, recovered from the released product (98.9% of published events).
  own     = stims inferred included (pq03) x our nan rule (max NaN run <= 0.05*segment, DANDI mask where
            available; no mask otherwise) x loc-std != 0.
Null variants (SUBSTITUTION: Randi's 'No_stim_list' control recordings are not on disk):
  spont   = 2 Leifer spontaneous-activity recordings (randi_raw/spont_data), pseudo-stimulus at every volume.
  spont62 = same, pseudo-stimuli every 62 volumes (~ the 31 s WT inter-stimulus interval), all phases pooled
            per recording is identical to 'spont'; this variant keeps a single phase (offset 0) to mimic null size.
Usage: pq04_reproduce_q.py <membership> <null> [tost]
"""
import json, os, sys, time
import numpy as np, h5py
from scipy.stats import spearmanr, kstest
sys.path.insert(0, os.path.dirname(__file__))
import pq_stats as S
from pq02_features import FA

OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
RNG = np.random.default_rng(20260916)


def load_events(membership):
    E = dict(np.load(f'{OUT}/events_wt.npz')); M = np.load(f'{OUT}/events_match.npz')
    E['aj'] = M['aj_resolved']
    if membership == 'oracle':
        valid = M['matched'].copy()
    elif membership == 'own':
        valid = M['included'] & (E['nanrun'] <= 0.05 * E['seglen']) & ~E['pre_zero'].astype(bool)
    else:
        raise ValueError(membership)
    valid &= (E['aj'] >= 0) & np.isfinite(E['dff']) & np.isfinite(E['sd'])
    return E, valid


def load_null(name):
    if name.startswith('wt'):
        # WT-internal nulls (SUBSTITUTION, partly calibrated on the published product):
        #   wtall        = all oracle-valid WT events pooled (distance- and identity-agnostic grand distribution)
        #   wtallx<s>    = wtall with both features multiplied by s
        #   wtqgt<tau>   = events of pairs whose PUBLISHED q > tau
        #   wtqeq        = events of pairs with PUBLISHED q_eq < 0.05 (declared non-connected)
        E, valid = load_events('oracle')
        with h5py.File(FA, 'r') as f:
            qp = f['wt/q'][:]; qe = f['wt/q_eq'][:]
        if name == 'wtall':
            m = valid
        elif name.startswith('wtallx'):
            # wtallx<s> = all valid WT events with both features scaled by s (1-parameter width calibration)
            sc = float(name[6:])
            return E['dff'][valid] * sc, E['sd'][valid] * sc
        elif name.startswith('wtqgt'):
            m = valid & (qp[E['ai'], E['aj']] > float(name[5:]))
        elif name == 'wtqeq':
            m = valid & (qe[E['ai'], E['aj']] < 0.05)
        else:
            raise ValueError(name)
        return E['dff'][m], E['sd'][m]
    N = dict(np.load(f'{OUT}/null_spont.npz'))
    ok = (N['nanrun'] <= 0.05 * N['seglen']) & ~N['pre_zero'].astype(bool) & np.isfinite(N['dff']) & np.isfinite(N['sd'])
    if name == 'spont62':
        ok &= (N['ee'] - 60) % 62 == 0
    return N['dff'][ok], N['sd'][ok]


def agreement(q, qp, occ_ours=None, occ_pub=None):
    fin = np.isfinite(q) & np.isfinite(qp)
    sp = spearmanr(-np.log10(q[fin]), -np.log10(qp[fin]))[0]
    so, sb = (q < 0.05), (qp < 0.05)
    inter = int((so & sb).sum()); uni = int((so | sb).sum())
    d = dict(n_pairs_finite_ours=int(np.isfinite(q).sum()), n_pairs_finite_pub=int(np.isfinite(qp).sum()),
             n_pairs_finite_both=int(fin.sum()), n_sig_ours=int(so.sum()), n_sig_pub=int(sb.sum()),
             n_sig_both=inter, frac_pub_sig_recovered=round(inter / max(1, sb.sum()), 4),
             frac_ours_sig_in_pub=round(inter / max(1, so.sum()), 4), jaccard=round(inter / max(1, uni), 4),
             spearman_neglog10q_both_finite=round(float(sp), 4))
    offd = ~np.eye(300, dtype=bool)
    d['n_sig_ours_offdiag'] = int((so & offd).sum()); d['n_sig_pub_offdiag'] = int((sb & offd).sum())
    d['jaccard_offdiag'] = round(int((so & sb & offd).sum()) / max(1, int(((so | sb) & offd).sum())), 4)
    return d


def main():
    membership = sys.argv[1] if len(sys.argv) > 1 else 'oracle'
    null = sys.argv[2] if len(sys.argv) > 2 else 'spont'
    tost = len(sys.argv) > 3 and sys.argv[3] == 'tost'
    t0 = time.time()
    E, valid = load_events(membership)
    nd, ns = load_null(null)
    # KS implementation check vs scipy.stats.kstest on random pairs of real events
    idx = np.where(valid)[0]
    nullS = S.SortedNull(nd)
    diffs = []
    for _ in range(60):
        m = RNG.integers(1, 60); smp = E['dff'][RNG.choice(idx, m, replace=False)]
        diffs.append(abs(S.ks_p(nullS, smp) - kstest(nd, smp, alternative='two-sided')[1]))
    tost_diffs = []
    stn = S.null_stats(nd); eps = 1.2 * np.std(nd)
    from statsmodels.stats.weightstats import ttost_ind
    for _ in range(20):
        m = RNG.integers(1, 40); smp = E['dff'][RNG.choice(idx, m, replace=False)]
        a = S.fast_ttost_ind(stn, smp, -eps, eps); b = ttost_ind(nd, smp, -eps, eps)[0]
        if np.isfinite(b): tost_diffs.append(abs(a - b))
    R = S.pair_q(E['ai'][valid], E['aj'][valid], E['dff'][valid], E['sd'][valid], nd, ns, tost=tost)
    with h5py.File(FA, 'r') as f:
        qp = f['wt/q'][:]; qep = f['wt/q_eq'][:]; occ1 = f['wt/occ1'][:]
    res = dict(membership=membership, null=null, null_n=int(len(nd)), n_valid_events=int(valid.sum()),
               ks_fast_vs_scipy_max_abs_pdiff=float(max(diffs)),
               tost_fast_vs_statsmodels_max_abs_pdiff=float(max(tost_diffs)) if tost_diffs else None, pi0=R['pi0'],
               q=agreement(R['q'], qp))
    # published p proxy: rank correlation of our p with published q is identical to -log q Spearman ranks only
    # when pi0 equal; report pi0 implied by published q for context: max(q) ~ pi0 * max(p)
    res['published_q_max'] = float(np.nanmax(qp))
    if tost:
        res['pi0_eq'] = R['pi0_eq']
        fin = np.isfinite(R['q_eq']) & np.isfinite(qep)
        so, sb = R['q_eq'] < 0.05, qep < 0.05
        res['q_eq'] = dict(n_eq_ours=int(so.sum()), n_eq_pub=int(sb.sum()), n_both=int((so & sb).sum()),
                           jaccard=round(int((so & sb).sum()) / max(1, int((so | sb).sum())), 4),
                           spearman_neglog10=round(float(spearmanr(-np.log10(R['q_eq'][fin]), -np.log10(qep[fin]))[0]), 4))
    # n_valid vs published dFF_all lengths
    with h5py.File(FA, 'r') as f:
        DA = f['wt/dFF_all']
        pairs = np.argwhere((occ1 > 0))
        lens = np.array([len(DA[i, j]) for i, j in pairs])
    ours = R['n_valid'][pairs[:, 0], pairs[:, 1]]
    res['n_valid_events_equals_published_dFF_all_len_frac_pairs'] = round(float((ours == lens).mean()), 5)
    res['seconds'] = round(time.time() - t0, 1)
    tag = f'{membership}_{null}'
    np.savez_compressed(f'{OUT}/q_repro_{tag}.npz', **{k: v for k, v in R.items() if isinstance(v, np.ndarray)})
    json.dump(res, open(f'{OUT}/pq04_repro_{tag}.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
