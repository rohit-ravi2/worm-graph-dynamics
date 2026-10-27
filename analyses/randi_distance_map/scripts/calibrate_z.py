"""z-unit calibration by local packing isotropy: choose k (xy-units per z-unit) such that
nearest-neighbour displacement components have equal mean magnitude along x, y and z.
Assumption (flagged): nuclear packing in head ganglia is locally isotropic."""
import glob, json, os
import numpy as np
OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
Cs = [np.load(f)['C'].astype(float) for f in glob.glob(os.path.join(OUT, 'cache', 'sub-*.npz'))]
res = []
for k in np.arange(0.5, 8.01, 0.25):
    comp = []
    for C in Cs:
        P = C * np.array([1, 1, k])
        D = np.linalg.norm(P[:, None] - P[None], axis=2); np.fill_diagonal(D, np.inf)
        j = D.argmin(1)
        comp.append(np.abs(P[j] - P))
    comp = np.concatenate(comp).mean(0)
    res.append(dict(k=float(k), mean_abs_dx=comp[0], mean_abs_dy=comp[1], mean_abs_dz=comp[2],
                    ratio_z_over_xy=comp[2] / comp[:2].mean()))
best = min(res, key=lambda r: abs(np.log(r['ratio_z_over_xy'])))
for r in res[::4]: print({a: round(b, 3) for a, b in r.items()})
print('BEST', best)
json.dump(dict(curve=res, best=best), open(os.path.join(OUT, 'calibration_z.json'), 'w'), indent=1, default=float)
