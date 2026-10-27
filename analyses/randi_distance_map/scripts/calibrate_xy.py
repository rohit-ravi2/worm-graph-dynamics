"""Calibrate xy pixel size (um/px) given z plane spacing 1.22 um (from depth_per_frame_in_um).

Identifiability: animals are rotated differently relative to the imaging axes, so the same
labeled neuron pair has the same physical distance across animals only at the correct
anisotropy sx/sz. Pick sx minimizing the median across-animal CV of labeled pair distances.
"""
import glob, json, os, itertools
import numpy as np
from collections import defaultdict
SZ = 1.22
OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
vecs = defaultdict(list)
for f in sorted(glob.glob(os.path.join(OUT, 'cache', 'sub-*.npz'))):
    d = np.load(f)
    lab, C = d['roi_label'], d['C'].astype(float)
    idx = {l: i for i, l in enumerate(lab) if l and list(lab).count(l) == 1}
    names = sorted(idx)
    for a, b in itertools.combinations(names, 2):
        vecs[(a, b)].append(C[idx[a]] - C[idx[b]])
pairs = {k: np.array(v) for k, v in vecs.items() if len(v) >= 8}
print('pairs seen in >=8 animals:', len(pairs))
res = []
for sx in np.round(np.arange(0.20, 0.81, 0.02), 3):
    cvs, meds = [], []
    for v in pairs.values():
        dist = np.sqrt((sx * v[:, 0]) ** 2 + (sx * v[:, 1]) ** 2 + (SZ * v[:, 2]) ** 2)
        if np.median(dist) > 3:
            cvs.append(np.std(dist) / np.mean(dist)); meds.append(np.median(dist))
    res.append(dict(sx=float(sx), median_cv=float(np.median(cvs)), median_pair_dist_um=float(np.median(meds))))
best = min(res, key=lambda r: r['median_cv'])
for r in res: print(r)
print('BEST', best)
# extent sanity: span of ROI centroids in um at best sx
spans = []
for f in sorted(glob.glob(os.path.join(OUT, 'cache', 'sub-*.npz'))):
    C = np.load(f)['C'].astype(float)
    spans.append(np.ptp(C, 0) * np.array([best['sx'], best['sx'], SZ]))
print('median ROI span (um) x,y,z:', np.median(spans, 0).round(1))
json.dump(dict(sz=SZ, curve=res, best=best, median_span_um=np.median(spans, 0).tolist()),
          open(os.path.join(OUT, 'calibration_xy.json'), 'w'), indent=1)
