"""T3 map: axial vs lateral coupling among NON-overlapping boxes, in extraction-box units
(lateral = hypot(dx,dy)/13, axial = |dz|/7). Assumption (flagged): Leifer's box is roughly
nucleus-shaped physically, so biology is ~isotropic in box units; axial PSF elongation predicts
excess along the axial direction at matched d_box. Readouts: response rate (z_g_full>3, target up)
on a (lateral, axial) grid, and axial-dominated vs lateral-dominated at matched d_box bins.
"""
import json, os
import numpy as np
OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
T = np.load(os.path.join(OUT, 'responses_wt.npz'))
animal = T['animal']; key = animal.astype(np.int64) * 100000 + T['stim']
tg = T['is_target']; tz = dict(zip(key[tg], T['z_g_full'][tg]))
TZ = np.array([tz.get(k, np.nan) for k in key])
dx, dy, dz = T['dx_target'], T['dy_target'], T['dz_target']
lat, ax = np.hypot(dx, dy) / 13, np.abs(dz) / 7
overlap = (np.abs(dx) < 13) & (np.abs(dy) < 13) & (np.abs(dz) < 7)
base = (T['target'] >= 0) & ~tg & np.isfinite(T['z_g_full']) & np.isfinite(dz) & ~overlap & (TZ > 3)
y = T['z_g_full'] > 3
A = np.unique(animal); rng = np.random.default_rng(4)


def rate(m, B=400):
    num = np.bincount(animal[m], weights=y[m].astype(float), minlength=A.max() + 1)
    den = np.bincount(animal[m], minlength=A.max() + 1).astype(float)
    bs = [num[s].sum() / max(den[s].sum(), 1) for s in (rng.choice(A, len(A)) for _ in range(B))]
    return [round(float(num.sum() / max(den.sum(), 1)), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4), int(den.sum())]


grid = []
LB = [0, 0.5, 1, 1.5, 2, 3, 100]; AB = [0, 0.5, 1, 1.5, 2, 3, 100]
for i in range(len(LB) - 1):
    for j in range(len(AB) - 1):
        m = base & (lat >= LB[i]) & (lat < LB[i + 1]) & (ax >= AB[j]) & (ax < AB[j + 1])
        if m.sum() >= 80:
            grid.append(dict(lateral=[LB[i], LB[i + 1]], axial=[AB[j], AB[j + 1]], rate=rate(m)))
dbox = np.hypot(lat, ax)
matched = []
for lo, hi in [(1, 1.5), (1.5, 2), (2, 3), (3, 5)]:
    m = base & (dbox >= lo) & (dbox < hi)
    matched.append(dict(d_box=[lo, hi], axial_dominated=rate(m & (ax > lat)), lateral_dominated=rate(m & (lat >= ax))))
floor = rate(base & (dbox >= 5))
out = dict(grid=grid, matched=matched, floor_dbox_ge5=floor)
json.dump(out, open(os.path.join(OUT, 't3_anisotropy.json'), 'w'), indent=1)
print('floor', floor)
for g in grid: print('lat', g['lateral'], 'ax', g['axial'], g['rate'])
for r in matched: print(r)
