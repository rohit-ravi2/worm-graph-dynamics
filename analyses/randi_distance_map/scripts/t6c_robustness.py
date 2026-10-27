"""T6c: robustness of the within-pair near/far contrast (funatlas-significant pairs and all pairs)
to detector threshold (z 2.5/3/4), window (early 0-2 s vs full 0-10 s), and near cutoff
(d_box 1.5/2.0/2.5/3.0). Also restricts to NON-overlapping boxes to remove shared-volume bleed."""
import json, os, itertools
import h5py
import numpy as np
from collections import defaultdict
OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
WNA = (os.path.join(os.path.dirname(__import__('wormneuroatlas').__file__), 'data') + '/')
with h5py.File(WNA + 'funatlas.h5') as h:
    ids = [x.decode() for x in h['neuron_ids'][:]]; qwt = h['wt/q'][:]; occ = h['wt/occ1'][:]
ix = {n: i for i, n in enumerate(ids)}
T = np.load(os.path.join(OUT, 'responses_wt.npz'))
animal, stim, rl, tl = T['animal'], T['stim'], T['resp_label'], T['target_label']
dup = set()
for a in np.unique(animal):
    m = (animal == a) & (stim == 0); labs, c = np.unique(rl[m], return_counts=True)
    dup.update((a, l) for l, k in zip(labs, c) if k > 1)
ok = (T['target'] >= 0) & ~T['is_target'] & np.isin(rl, ids) & np.isin(tl, ids) & (rl != tl) & np.isfinite(T['dz_target'])
ok &= np.array([(a, r) not in dup and (a, t) not in dup for a, r, t in zip(animal, rl, tl)])
idx = np.where(ok)[0]
dx, dy, dz = T['dx_target'][idx], T['dy_target'][idx], T['dz_target'][idx]
d_box = np.sqrt((dx / 13) ** 2 + (dy / 13) ** 2 + (dz / 7) ** 2)
overlap = (np.abs(dx) < 13) & (np.abs(dy) < 13) & (np.abs(dz) < 7)
S, R = tl[idx], rl[idx]
pair = np.array([s + '>' + r for s, r in zip(S, R)])
fsig = np.array([np.isfinite(qwt[ix[r], ix[s]]) and qwt[ix[r], ix[s]] < 0.05 and occ[ix[r], ix[s]] >= 2 for s, r in zip(S, R)])
rng = np.random.default_rng(8)
rows = []
for win, thr, cut, excl_ov in itertools.product(['g_full', 'g_early'], [2.5, 3, 4], [1.5, 2.0, 2.5, 3.0], [False, True]):
    z = T['z_' + win][idx]
    base = np.isfinite(z) & (~overlap if excl_ov else True)
    y = (z > thr).astype(int); near = d_box < cut
    for stratum, sel in [('all', base), ('funatlas_sig', base & fsig)]:
        st = defaultdict(lambda: [0, 0, 0, 0])
        for p, nr, yy in zip(pair[sel], near[sel], y[sel]):
            v = st[p]
            if nr: v[0] += 1; v[1] += yy
            else: v[2] += 1; v[3] += yy
        V = np.array([v for v in st.values() if v[0] >= 1 and v[2] >= 1], float)
        if len(V) < 20:
            continue
        f = lambda W: (W[:, 1].sum() / W[:, 0].sum()) - (W[:, 3].sum() / W[:, 2].sum())
        bs = [f(V[rng.integers(0, len(V), len(V))]) for _ in range(500)]
        rn, rf = V[:, 1].sum() / V[:, 0].sum(), V[:, 3].sum() / V[:, 2].sum()
        rows.append(dict(window=win, z_thr=thr, near_cut=cut, exclude_overlap=excl_ov, stratum=stratum, n_pairs=len(V),
                         rate_near=round(rn, 4), rate_far=round(rf, 4), ratio=round(rn / max(rf, 1e-9), 2),
                         diff_ci=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]))
json.dump(rows, open(os.path.join(OUT, 't6c_robustness.json'), 'w'), indent=1)
ratios = [r['ratio'] for r in rows]; lows = [r['diff_ci'][0] for r in rows]
print('configs', len(rows), 'ratio min/median/max', min(ratios), np.median(ratios), max(ratios), '| configs with diff CI lower <= 0:', sum(l <= 0 for l in lows))
for r in rows:
    if r['stratum'] == 'funatlas_sig' and r['z_thr'] == 3 and r['window'] == 'g_full':
        print(r)
for r in sorted(rows, key=lambda r: r['ratio'])[:4]: print('weakest', r)
