"""T6d: is the within-pair near/far contrast a label-error artifact?
Per ROI NeuroPAL label confidence (DANDI seg labels_confidences via neuropal_ids), counted only when
the seg label equals the Leifer-exported label used in the analysis; otherwise confidence = NaN.
Re-run the T6b within-pair contrast restricted to events where BOTH target and responder labels
have confidence >= c (c = 0.8, 0.9, 1.0). Label errors predict the contrast shrinks toward 1 as c rises."""
import glob, json, os
import h5py
import numpy as np
from collections import defaultdict
OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
SEG = os.environ.get('RANDI_SEG_DIR', 'data/randi2023_dandi001075_seg')
WNA = (os.path.join(os.path.dirname(__import__('wormneuroatlas').__file__), 'data') + '/')
T = np.load(os.path.join(OUT, 'responses_wt.npz'))
animals = list(T['animals'])
conf = {}
for ai, sub in enumerate(animals):
    lab = np.load(os.path.join(OUT, 'cache', sub + '.npz'))['roi_label']
    with h5py.File(os.path.join(SEG, sub + '_seg.nwb'), 'r') as h:
        ph = h['processing/ophys']
        npid = ph['PumpProbeGreenSegmentations/PumpProbeGreenPlaneSegmentation/neuropal_ids'][:]
        N = ph['NeuroPALSegmentations/NeuroPALPlaneSegmentation']
        ids = N['id'][:]; labs = N['labels'][:]; cf = N['labels_confidences'][:]
    m = {int(i): (l.decode(), float(c)) for i, l, c in zip(ids, labs, cf)}
    for j, x in enumerate(npid):
        x = x.decode()
        if x.strip().isdigit() and int(x) in m and m[int(x)][0] == lab[j] and lab[j]:
            conf[(ai, j)] = m[int(x)][1]
with h5py.File(WNA + 'funatlas.h5') as h:
    fid = [x.decode() for x in h['neuron_ids'][:]]; qwt = h['wt/q'][:]; occ = h['wt/occ1'][:]
ix = {n: i for i, n in enumerate(fid)}
animal, stim, rl, tl = T['animal'], T['stim'], T['resp_label'], T['target_label']
dup = set()
for a in np.unique(animal):
    mm = (animal == a) & (stim == 0); labs_, c = np.unique(rl[mm], return_counts=True)
    dup.update((a, l) for l, k in zip(labs_, c) if k > 1)
ok = (T['target'] >= 0) & ~T['is_target'] & np.isin(rl, fid) & np.isin(tl, fid) & (rl != tl) & np.isfinite(T['dz_target']) & np.isfinite(T['z_g_full'])
ok &= np.array([(a, r) not in dup and (a, t) not in dup for a, r, t in zip(animal, rl, tl)])
idx = np.where(ok)[0]
cr = np.array([conf.get((a, j), np.nan) for a, j in zip(animal[idx], T['roi'][idx])])
ct = np.array([conf.get((a, j), np.nan) for a, j in zip(animal[idx], T['target'][idx])])
d_box = np.sqrt((T['dx_target'][idx] / 13) ** 2 + (T['dy_target'][idx] / 13) ** 2 + (T['dz_target'][idx] / 7) ** 2)
y = (T['z_g_full'][idx] > 3).astype(int); near = d_box < 2.5
S, R = tl[idx], rl[idx]; pair = np.array([s + '>' + r for s, r in zip(S, R)])
fsig = np.array([np.isfinite(qwt[ix[r], ix[s]]) and qwt[ix[r], ix[s]] < 0.05 and occ[ix[r], ix[s]] >= 2 for s, r in zip(S, R)])
rng = np.random.default_rng(9)
print('events with both confidences known:', np.isfinite(cr + ct).sum(), 'of', len(idx))
rows = []
for c in [None, 0.8, 0.9, 1.0]:
    sel0 = np.ones(len(idx), bool) if c is None else (np.nan_to_num(cr, nan=-9) >= c) & (np.nan_to_num(ct, nan=-9) >= c)
    for stratum, sel in [('all', sel0), ('funatlas_sig', sel0 & fsig)]:
        st = defaultdict(lambda: [0, 0, 0, 0])
        for p, nr, yy in zip(pair[sel], near[sel], y[sel]):
            v = st[p]
            if nr: v[0] += 1; v[1] += yy
            else: v[2] += 1; v[3] += yy
        V = np.array([v for v in st.values() if v[0] >= 1 and v[2] >= 1], float)
        if len(V) < 10:
            rows.append(dict(min_conf=c, stratum=stratum, n_pairs=len(V))); continue
        f = lambda W: (W[:, 1].sum() / W[:, 0].sum(), W[:, 3].sum() / W[:, 2].sum())
        bs = np.array([f(V[rng.integers(0, len(V), len(V))]) for _ in range(1000)])
        rn, rf = f(V)
        rows.append(dict(min_conf=c, stratum=stratum, n_pairs=len(V), n_near=int(V[:, 0].sum()), n_far=int(V[:, 2].sum()),
                         rate_near=round(rn, 4), rate_far=round(rf, 4), ratio=round(rn / max(rf, 1e-9), 2),
                         diff_ci=np.percentile(bs[:, 0] - bs[:, 1], [2.5, 97.5]).round(4).tolist()))
    for r in rows[-2:]: print(r)
json.dump(rows, open(os.path.join(OUT, 't6d_label_confidence.json'), 'w'), indent=1)
