"""T0/T1 map: response rate vs distance from the stimulated neuron, green vs red channel.
Targeted stims only (target ROI known), non-target ROIs. Distance bins in data units
(xy units; |dz| in z units). Rates for z>3 (increase), z<-3 (decrease). Animal-cluster bootstrap CIs.
"""
import json, os
import numpy as np
OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
T = np.load(os.path.join(OUT, 'responses_wt.npz'))
tgt = T['target'] >= 0
nt = tgt & ~T['is_target']
dxy = np.hypot(T['dx_target'], T['dy_target']); adz = np.abs(T['dz_target'])
animal = T['animal']
rng = np.random.default_rng(0)
A = np.unique(animal)
XY_BINS = [0, 6, 10, 14, 20, 30, 45, 70, 110, 1000]


def rate_ci(mask, stat, B=500):
    # per-animal counts then bootstrap animals
    num = np.bincount(animal[mask], weights=stat[mask].astype(float), minlength=A.max() + 1)
    den = np.bincount(animal[mask], minlength=A.max() + 1).astype(float)
    est = num.sum() / max(den.sum(), 1)
    bs = []
    for _ in range(B):
        s = rng.choice(A, len(A))
        bs.append(num[s].sum() / max(den[s].sum(), 1))
    return float(est), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], int(den.sum())


out = {}
for chan in ['g_full', 'g_early', 'r_full', 'r_early']:
    z = T['z_' + chan]
    fin = np.isfinite(z)
    rows = []
    for zsel, name in [(np.abs(adz) <= 1, 'same_plane_|dz|<=1'), (np.isfinite(adz), 'all_dz')]:
        for a, b in zip(XY_BINS[:-1], XY_BINS[1:]):
            m = nt & fin & zsel & (dxy >= a) & (dxy < b)
            up = rate_ci(m, z > 3); dn = rate_ci(m, z < -3)
            rows.append(dict(sel=name, dxy=[a, b], n=up[2], rate_up=up[0], ci_up=up[1], rate_down=dn[0], ci_down=dn[1]))
    m = tgt & T['is_target'] & fin
    up = rate_ci(m, z > 3); dn = rate_ci(m, z < -3)
    rows.append(dict(sel='TARGET_itself', n=up[2], rate_up=up[0], ci_up=up[1], rate_down=dn[0], ci_down=dn[1]))
    out[chan] = rows
    print('==', chan)
    for r in rows:
        print(f"{r['sel']:>20} {str(r.get('dxy','')):>12} n={r['n']:>7}  up {r['rate_up']:.4f} [{r['ci_up'][0]:.4f},{r['ci_up'][1]:.4f}]  down {r['rate_down']:.4f} [{r['ci_down'][0]:.4f},{r['ci_down'][1]:.4f}]")
json.dump(out, open(os.path.join(OUT, 't01_distance_profiles.json'), 'w'), indent=1)
