"""Pair-level Randi statistic from per-event features (shared by pq04 reproduction and pq05 regimes).

p (Funatlas.py:2625-2653): for each ordered atlas pair (i <- j) with >=1 valid event, two-sample two-sided KS
of the event dF/sigma values vs control dF/sigma and of the event 2nd-derivative values vs control; p==0 -> 1e-10;
Fisher-combined. q: Storey q value (multipy) over all finite p (Funatlas.py:2684-2689).
q_eq: TOST (statsmodels ttost_ind, eps = 1.2 * std(control)) per feature, Fisher, Storey (Funatlas.py:2590-2696).

KS speed-up: the null is sorted once; D is evaluated at the pair sample's jump points (sup of |F_ctrl - F_pair| is
attained there or at their left limits), p = scipy kstwo.sf(D, round(n1 n2/(n1+n2))) exactly as scipy's ks_2samp
'auto' mode does when max(n1, n2) > 10000 (scipy/stats/_stats_py.py ks_2samp). Verified against scipy.stats.kstest
in pq04 (max |p diff| reported).
"""
import os
from multiprocessing import Pool
import numpy as np
from scipy.stats import kstwo, kstest, combine_pvalues, t as tdist
import randi_q_lib as L


class SortedNull:
    def __init__(self, x):
        self.x = np.sort(np.asarray(x, float)); self.n = len(self.x)

    def cdf_right(self, v):
        return np.searchsorted(self.x, v, side='right') / self.n

    def cdf_left(self, v):
        return np.searchsorted(self.x, v, side='left') / self.n


def ks_p(null, sample):
    s = np.sort(np.asarray(sample, float)); m = len(s)
    if max(null.n, m) <= 10000:
        return kstest(null.x, s, alternative='two-sided')[1]
    u, idx_last = np.unique(s, return_index=False, return_counts=True)
    Fp_right = np.cumsum(idx_last) / m               # F_pair at u (right-continuous)
    Fp_left = np.concatenate(([0.0], Fp_right[:-1]))  # F_pair just below u
    d = max(np.max(np.abs(null.cdf_right(u) - Fp_right)), np.max(np.abs(null.cdf_left(u) - Fp_left)))
    en = null.n * m / (null.n + m)
    return float(np.clip(kstwo.sf(d, np.round(en)), 0, 1))


def fast_ttost_ind(null_stats, x2, low, upp):
    """statsmodels ttost_ind(x1=null, x2, low, upp, usevar='pooled') with precomputed null (n, mean, sumsquares):
    CompareMeans.std_meandiff_pooledvar (weightstats.py:986-999), ttest_ind (1016-1071), _tstat_generic
    ('larger' -> t.sf, 'smaller' -> t.cdf), ttost_ind returns max of the two p (1219-1221)."""
    n1, m1, ss1 = null_stats
    x2 = np.asarray(x2, float); n2 = len(x2); m2 = x2.mean(); ss2 = np.sum((x2 - m2) ** 2)
    dof = n1 - 1 + n2 - 1
    sd = np.sqrt((ss1 + ss2) / dof * (1.0 / n1 + 1.0 / n2))
    p1 = tdist.sf((m1 - m2 - low) / sd, dof)
    p2 = tdist.cdf((m1 - m2 - upp) / sd, dof)
    return float(np.maximum(p1, p2))


def null_stats(x):
    x = np.asarray(x, float); m = x.mean()
    return (len(x), m, np.sum((x - m) ** 2))


_G = {}


def _init(null_dff, null_sd, tost):
    _G['nd'] = SortedNull(null_dff); _G['ns'] = SortedNull(null_sd); _G['tost'] = tost
    _G['st_dff'] = null_stats(null_dff); _G['st_sd'] = null_stats(null_sd)
    _G['eps_dff'] = 1.2 * np.std(null_dff); _G['eps_sd'] = 1.2 * np.std(null_sd)  # tost_th = 1.2 (qvalues_map.py:79)


def _pair(args):
    key, dff, sd = args
    pd_ = ks_p(_G['nd'], dff); ps_ = ks_p(_G['ns'], sd)
    if pd_ == 0: pd_ = 1e-10
    if ps_ == 0: ps_ = 1e-10
    p = combine_pvalues([pd_, ps_], method='fisher')[1]
    pt = np.nan
    if _G['tost']:
        with np.errstate(all='ignore'):
            t1 = fast_ttost_ind(_G['st_dff'], dff, -_G['eps_dff'], _G['eps_dff'])
            t2 = fast_ttost_ind(_G['st_sd'], sd, -_G['eps_sd'], _G['eps_sd'])
            if t1 == 0: t1 = 1e-10
            if t2 == 0: t2 = 1e-10
            pt = combine_pvalues([t1, t2], method='fisher')[1]
    return key, p, pt


def pair_q(ai, aj, dff, sd, null_dff, null_sd, tost=False, procs=None, n_atlas=300):
    """ai, aj, dff, sd: arrays over VALID events. Returns dict of 300x300 p, q (and tost p, q_eq)."""
    procs = procs or int(os.environ.get('PQ_PROCS', 12))
    key = ai.astype(np.int64) * n_atlas + aj
    o = np.argsort(key, kind='stable')
    ks, starts = np.unique(key[o], return_index=True)
    ends = np.append(starts[1:], len(o))
    jobs = [(int(k), dff[o[a:b]], sd[o[a:b]]) for k, a, b in zip(ks, starts, ends)]
    P = np.full((n_atlas, n_atlas), np.nan); PT = np.full((n_atlas, n_atlas), np.nan)
    N = np.zeros((n_atlas, n_atlas), int)
    for k, a, b in zip(ks, starts, ends):
        N[k // n_atlas, k % n_atlas] = b - a
    with Pool(procs, initializer=_init, initargs=(null_dff, null_sd, tost)) as pool:
        for k, p, pt in pool.imap_unordered(_pair, jobs, chunksize=200):
            P[k // n_atlas, k % n_atlas] = p; PT[k // n_atlas, k % n_atlas] = pt
    Q, pi0 = L.q_matrix(P)
    out = dict(p=P, q=Q, pi0=pi0, n_valid=N)
    if tost:
        # Funatlas.py:2692 quirk: zeros replaced using the regular-p mask (no effect unless p==0)
        PT2 = np.copy(PT); PT2[P == 0] = 1e-10
        fin = np.isfinite(PT2); QE = np.full_like(PT2, np.nan)
        qe, pi0e = L.storey_qvalue(PT2[fin]); QE[fin] = qe
        out.update(p_tost=PT, q_eq=QE, pi0_eq=pi0e)
    return out


class PairTester:
    """Persistent worker pool for repeated q maps against one fixed null (no TOST)."""

    def __init__(self, null_dff, null_sd, procs=None):
        self.pool = Pool(procs or int(os.environ.get('PQ_PROCS', 12)), initializer=_init,
                         initargs=(null_dff, null_sd, False))

    def q(self, ai, aj, dff, sd, n_atlas=300):
        key = ai.astype(np.int64) * n_atlas + aj
        o = np.argsort(key, kind='stable')
        ks, starts = np.unique(key[o], return_index=True)
        ends = np.append(starts[1:], len(o))
        jobs = [(int(k), dff[o[a:b]], sd[o[a:b]]) for k, a, b in zip(ks, starts, ends)]
        P = np.full((n_atlas, n_atlas), np.nan)
        for k, p, _ in self.pool.imap_unordered(_pair, jobs, chunksize=200):
            P[k // n_atlas, k % n_atlas] = p
        Q, pi0 = L.q_matrix(P)
        return Q, pi0, P

    def close(self):
        self.pool.close(); self.pool.join()
