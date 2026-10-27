"""T6: rebuild a Randi-style labeled pair map from per-trial WT data and re-score predictors,
with and without events in the contamination zone (measured per-animal distance).

Regimes: A = all events; B = drop events with d_box < 2.5; C = drop events with d_box < 3.5
(d_box in extraction-box units, from T1b/T2). Per ordered labeled pair (source s -> responder r):
n events, k responders (z_g_full > 3); significant if one-sided binomial p vs the regime's far floor
(events with d_box >= 5) passes BH q < 0.05 over pairs with n >= 2.
Predictors (fixed before scoring): -atlas distance; connectome W = log1p(chem[r,s] + gap[r,s])
(wormneuroatlas convention chem[i,j] = j->i, so chem[r,s] = s->r).
Metric: within-source AUROC averaged over sources with >=1 significant and >=1 non-significant pair;
source-bootstrap CIs; connectome AUROC also within atlas-distance terciles.
"""
import json, os
import h5py
import numpy as np
from collections import defaultdict
from scipy.stats import binom, rankdata
OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
WNA = (os.path.join(os.path.dirname(__import__('wormneuroatlas').__file__), 'data') + '/')
with h5py.File(WNA + 'funatlas.h5') as h:
    ids = [x.decode() for x in h['neuron_ids'][:]]
with h5py.File(WNA + 'aconnectome_default.h5') as h:
    chem, gap = h['chem'][:], h['gap'][:]
ix = {n: i for i, n in enumerate(ids)}
names = open(WNA + 'anatlas_neuron_positions.txt').readline().lstrip('#').split()
P = np.loadtxt(WNA + 'anatlas_neuron_positions.txt', comments='#'); apos = {n: P[i] for i, n in enumerate(names)}

T = np.load(os.path.join(OUT, 'responses_wt.npz'))
animal, stim, rl, tl = T['animal'], T['stim'], T['resp_label'], T['target_label']
dup = set()
for a in np.unique(animal):
    m = (animal == a) & (stim == 0)
    labs, c = np.unique(rl[m], return_counts=True)
    dup.update((a, l) for l, k in zip(labs, c) if k > 1)
ok = (T['target'] >= 0) & ~T['is_target'] & np.isin(rl, ids) & np.isin(tl, ids) & (rl != tl) & np.isin(rl, names) & np.isin(tl, names)
ok &= np.isfinite(T['z_g_full']) & np.isfinite(T['dz_target'])
ok &= np.array([(a, r) not in dup and (a, t) not in dup for a, r, t in zip(animal, rl, tl)])
idx = np.where(ok)[0]
d_box = np.sqrt((T['dx_target'][idx] / 13) ** 2 + (T['dy_target'][idx] / 13) ** 2 + (T['dz_target'][idx] / 7) ** 2)
y = T['z_g_full'][idx] > 3
src, rsp = tl[idx], rl[idx]
rng = np.random.default_rng(6)


def auroc(score, lab):
    pos = lab.sum(); neg = len(lab) - pos
    if pos == 0 or neg == 0:
        return np.nan
    r = rankdata(score)
    return (r[lab].sum() - pos * (pos + 1) / 2) / (pos * neg)


def bh(p):
    n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


out = {}
for reg, cut in [('A_all', 0), ('B_drop_dbox_lt_2.5', 2.5), ('C_drop_dbox_lt_3.5', 3.5)]:
    keep = d_box >= cut
    floor = y[d_box >= 5].mean()
    agg = defaultdict(lambda: [0, 0])
    for s_, r_, yy in zip(src[keep], rsp[keep], y[keep]):
        v = agg[(s_, r_)]; v[0] += 1; v[1] += int(yy)
    pairs = [(k, v) for k, v in agg.items() if v[0] >= 2]
    n = np.array([v[0] for _, v in pairs]); kk = np.array([v[1] for _, v in pairs])
    p = binom.sf(kk - 1, n, floor); q = bh(p); sig = q < 0.05
    S = np.array([k[0] for k, _ in pairs]); R = np.array([k[1] for k, _ in pairs])
    dat = -np.array([np.linalg.norm(apos[s_] - apos[r_]) for s_, r_ in zip(S, R)])
    W = np.array([np.log1p(chem[ix[r_], ix[s_]] + gap[ix[r_], ix[s_]]) for s_, r_ in zip(S, R)])
    per = defaultdict(dict)
    for s_ in np.unique(S):
        m = S == s_
        if sig[m].sum() == 0 or sig[m].sum() == m.sum():
            continue
        per[s_]['dist'] = auroc(dat[m], sig[m]); per[s_]['conn'] = auroc(W[m], sig[m])
    srcs = list(per)
    def summ(key):
        v = np.array([per[s_][key] for s_ in srcs])
        bs = [np.nanmean(v[rng.integers(0, len(v), len(v))]) for _ in range(2000)]
        return [round(float(np.nanmean(v)), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    diff = np.array([per[s_]['dist'] - per[s_]['conn'] for s_ in srcs])
    bsd = [np.nanmean(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(2000)]
    # connectome within atlas-distance terciles (pooled across sources)
    terc = np.quantile(dat, [0, 1 / 3, 2 / 3, 1])
    within = []
    for i in range(3):
        m = (dat >= terc[i]) & (dat <= terc[i + 1])
        within.append(dict(tercile=i, n_pairs=int(m.sum()), sig_rate=round(float(sig[m].mean()), 4), auroc_conn=round(float(auroc(W[m], sig[m])), 4)))
    out[reg] = dict(floor=round(float(floor), 4), n_events=int(keep.sum()), n_pairs=int(len(pairs)), n_sig=int(sig.sum()),
                    n_sources_scored=len(srcs), auroc_neg_atlas_distance=summ('dist'), auroc_connectome=summ('conn'),
                    dist_minus_conn=[round(float(np.nanmean(diff)), 4), round(float(np.percentile(bsd, 2.5)), 4), round(float(np.percentile(bsd, 97.5)), 4)],
                    connectome_within_distance_terciles=within)
    print(reg, json.dumps(out[reg]))
json.dump(out, open(os.path.join(OUT, 't6_corrected_map.json'), 'w'), indent=1)
