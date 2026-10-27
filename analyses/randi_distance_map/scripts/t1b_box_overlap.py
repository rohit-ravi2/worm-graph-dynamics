"""T1b: does the near-field excess survive when extraction boxes do not overlap?
Leifer wormdatamodel box (3,5,5) -> in DANDI units extent x,y: c+-6, z: c+-3 (voxel_mask stride 3).
Overlap fraction = prod_axis max(0, span - |d|)/span with span x,y = 13, z = 7.
Strata within each xy-distance bin: overlap>0 vs overlap==0 (no shared box volume).
"""
import json, os
import numpy as np
OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
T = np.load(os.path.join(OUT, 'responses_wt.npz'))
nt = (T['target'] >= 0) & ~T['is_target']
dx, dy, dz = T['dx_target'], T['dy_target'], T['dz_target']
ov = (np.clip(13 - np.abs(dx), 0, None) / 13) * (np.clip(13 - np.abs(dy), 0, None) / 13) * (np.clip(7 - np.abs(dz), 0, None) / 7)
dxy = np.hypot(dx, dy); d3 = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
animal = T['animal']; A = np.unique(animal); rng = np.random.default_rng(2)
key = animal.astype(np.int64) * 100000 + T['stim']
tg = T['is_target']; tz = dict(zip(key[tg], T['z_g_full'][tg]))
TZ = np.array([tz.get(k, np.nan) for k in key])


def rate(mask, stat, B=400):
    num = np.bincount(animal[mask], weights=stat[mask].astype(float), minlength=A.max() + 1)
    den = np.bincount(animal[mask], minlength=A.max() + 1).astype(float)
    bs = [num[s].sum() / max(den[s].sum(), 1) for s in (rng.choice(A, len(A)) for _ in range(B))]
    return dict(rate=round(float(num.sum() / max(den.sum(), 1)), 4), ci=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], n=int(den.sum()))


out = {}
for chan in ['g_full', 'r_full']:
    z = T['z_' + chan]; base = nt & np.isfinite(z) & np.isfinite(dz)
    rows = []
    for a, b in [(0, 8), (8, 12), (12, 16), (16, 20), (20, 30), (30, 45), (45, 1000)]:
        for lab, sel in [('overlap', ov > 0), ('no_overlap', ov == 0), ('no_overlap_|dz|>6', (ov == 0) & (np.abs(dz) > 6)), ('no_overlap_|dz|<=6', (ov == 0) & (np.abs(dz) <= 6))]:
            m = base & sel & (dxy >= a) & (dxy < b)
            if m.sum() < 50:
                continue
            r = dict(dxy=[a, b], stratum=lab, all=rate(m, z > 3), target_up=rate(m & (TZ > 3), z > 3), target_flat=rate(m & (np.abs(TZ) < 1), z > 3))
            rows.append(r)
            print(chan, r)
    # 3D-distance bins, no overlap only
    for a, b in [(10, 14), (14, 20), (20, 30), (30, 45), (45, 1000)]:
        m = base & (ov == 0) & (d3 >= a) & (d3 < b)
        r = dict(d3=[a, b], stratum='no_overlap_3D', all=rate(m, z > 3), target_up=rate(m & (TZ > 3), z > 3), target_flat=rate(m & (np.abs(TZ) < 1), z > 3))
        rows.append(r); print(chan, r)
    out[chan] = rows
json.dump(out, open(os.path.join(OUT, 't1b_box_overlap.json'), 'w'), indent=1)
