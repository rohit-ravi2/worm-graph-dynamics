"""T6b: within-labeled-pair test + power-matched control for T6.

(1) Within-pair: for each ordered labeled pair (source -> responder) with events both NEAR
    (d_box < 2.5 in that animal) and FAR (d_box >= 2.5), compare response rate (z_g_full > 3) near vs
    far. Contamination predicts near >> far for the same identities; identity-determined propagation
    predicts near ~= far. Pooled rates with pair-cluster bootstrap; strata: funatlas WT q<0.05 pairs,
    connectome-edge pairs (chem+gap > 0, s->r), all pairs. Also a per-pair logistic-free summary:
    mean(near rate - far rate) over pairs.
(2) Power-matched control for T6 regime B: drop the same number of events at random (distance-
    independent), recompute the number of BH-significant pairs (20 draws).
"""
import json, os
import h5py
import numpy as np
from collections import defaultdict
from scipy.stats import binom
OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
WNA = (os.path.join(os.path.dirname(__import__('wormneuroatlas').__file__), 'data') + '/')
with h5py.File(WNA + 'funatlas.h5') as h:
    ids = [x.decode() for x in h['neuron_ids'][:]]
    qwt = h['wt/q'][:]; occ = h['wt/occ1'][:]
with h5py.File(WNA + 'aconnectome_default.h5') as h:
    chem, gap = h['chem'][:], h['gap'][:]
ix = {n: i for i, n in enumerate(ids)}
T = np.load(os.path.join(OUT, 'responses_wt.npz'))
animal, stim, rl, tl = T['animal'], T['stim'], T['resp_label'], T['target_label']
dup = set()
for a in np.unique(animal):
    m = (animal == a) & (stim == 0)
    labs, c = np.unique(rl[m], return_counts=True)
    dup.update((a, l) for l, k in zip(labs, c) if k > 1)
ok = (T['target'] >= 0) & ~T['is_target'] & np.isin(rl, ids) & np.isin(tl, ids) & (rl != tl)
ok &= np.isfinite(T['z_g_full']) & np.isfinite(T['dz_target'])
ok &= np.array([(a, r) not in dup and (a, t) not in dup for a, r, t in zip(animal, rl, tl)])
idx = np.where(ok)[0]
d_box = np.sqrt((T['dx_target'][idx] / 13) ** 2 + (T['dy_target'][idx] / 13) ** 2 + (T['dz_target'][idx] / 7) ** 2)
y = (T['z_g_full'][idx] > 3).astype(int)
S, R = tl[idx], rl[idx]
pair = np.array([s + '>' + r for s, r in zip(S, R)])
near = d_box < 2.5
rng = np.random.default_rng(7)

# --- (1) within-pair
stats = defaultdict(lambda: [0, 0, 0, 0])  # n_near, k_near, n_far, k_far
for p, nr, yy in zip(pair, near, y):
    v = stats[p]
    if nr:
        v[0] += 1; v[1] += yy
    else:
        v[2] += 1; v[3] += yy
both = {p: v for p, v in stats.items() if v[0] >= 1 and v[2] >= 1}
def attrs(p):
    s, r = p.split('>')
    q = qwt[ix[r], ix[s]]
    return dict(funatlas_sig=bool(np.isfinite(q) and q < 0.05 and occ[ix[r], ix[s]] >= 2),
                conn_edge=bool(chem[ix[r], ix[s]] + gap[ix[r], ix[s]] > 0))
res = {}
for stratum in ['all', 'funatlas_sig', 'conn_edge', 'no_conn_edge']:
    ps = [p for p in both if stratum == 'all' or (stratum == 'no_conn_edge' and not attrs(p)['conn_edge']) or (stratum not in ('all', 'no_conn_edge') and attrs(p)[stratum])]
    if len(ps) < 5:
        continue
    V = np.array([both[p] for p in ps], float)
    def pooled(Vs):
        return Vs[:, 1].sum() / Vs[:, 0].sum(), Vs[:, 3].sum() / Vs[:, 2].sum()
    est = pooled(V)
    bs = np.array([pooled(V[rng.integers(0, len(V), len(V))]) for _ in range(2000)])
    diff = bs[:, 0] - bs[:, 1]
    res[stratum] = dict(n_pairs=len(ps), n_near_events=int(V[:, 0].sum()), n_far_events=int(V[:, 2].sum()),
                        rate_near=[round(est[0], 4)] + np.percentile(bs[:, 0], [2.5, 97.5]).round(4).tolist(),
                        rate_far=[round(est[1], 4)] + np.percentile(bs[:, 1], [2.5, 97.5]).round(4).tolist(),
                        near_minus_far=[round(est[0] - est[1], 4)] + np.percentile(diff, [2.5, 97.5]).round(4).tolist(),
                        ratio_near_over_far=round(est[0] / max(est[1], 1e-9), 2))
    print(stratum, res[stratum])

# --- (2) power-matched random drop control for T6 regime B
def n_sig(keep):
    floor = y[d_box >= 5].mean()
    agg = defaultdict(lambda: [0, 0])
    for p, yy in zip(pair[keep], y[keep]):
        v = agg[p]; v[0] += 1; v[1] += yy
    v = np.array([x for x in agg.values() if x[0] >= 2])
    pv = binom.sf(v[:, 1] - 1, v[:, 0], floor)
    n = len(pv); o = np.argsort(pv)
    q = np.empty(n); q[o] = np.minimum.accumulate((pv[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return int((q < 0.05).sum())
full = n_sig(np.ones(len(y), bool)); dropB = n_sig(d_box >= 2.5)
n_drop = int((d_box < 2.5).sum())
rand = []
for _ in range(20):
    keep = np.ones(len(y), bool); keep[rng.choice(len(y), n_drop, replace=False)] = False
    rand.append(n_sig(keep))
ctrl = dict(n_sig_all=full, n_sig_drop_near=dropB, n_events_dropped=n_drop,
            n_sig_random_drop_median=float(np.median(rand)), n_sig_random_drop_range=[int(min(rand)), int(max(rand))])
print('power-matched', ctrl)
json.dump(dict(within_pair=res, power_matched_control=ctrl), open(os.path.join(OUT, 't6b_within_pair.json'), 'w'), indent=1)
