"""PQ05: attach per-animal target-responder geometry (run-1 table out/responses_wt.npz, DANDI:001075 centroids)
to every PQ02 event. d_box = sqrt((dx/13)^2 + (dy/13)^2 + (dz/7)^2) (extraction-box units, run-1 convention).
Events with no geometry (3 exported datasets without a DANDI segmentation NWB, -3 manually located targets)
get NaN. Output: out/events_geom.npz (d_box aligned to events_wt.npz rows) + out/pq05_geometry.json.
"""
import json, os
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
RUN1 = os.path.join(os.path.dirname(__file__), '..', '..', 'out', 'responses_wt.npz')
E = np.load(f'{OUT}/events_wt.npz')
idx = json.load(open(f'{OUT}/prep_index_wt.json'))
T = np.load(RUN1)
animals = list(T['animals'])
sub_of_k = {int(r['k']): r['sub'] for r in idx}
a_of_k = {k: animals.index(s) for k, s in sub_of_k.items() if s in animals}

key_T = (T['animal'].astype(np.int64) * 1000 + T['stim']) * 1000 + T['roi']
dbox_T = np.sqrt((T['dx_target'] / 13) ** 2 + (T['dy_target'] / 13) ** 2 + (T['dz_target'] / 7) ** 2)
order = np.argsort(key_T); ks = key_T[order]

ev_a = np.array([a_of_k.get(int(k), -1) for k in E['ds']])
key_E = (ev_a.astype(np.int64) * 1000 + E['stim']) * 1000 + E['roi']
pos = np.searchsorted(ks, key_E)
pos = np.clip(pos, 0, len(ks) - 1)
hit = (ev_a >= 0) & (ks[pos] == key_E)
d_box = np.full(len(key_E), np.nan)
d_box[hit] = dbox_T[order[pos[hit]]]
tgt_ok = np.full(len(key_E), False)
tgt_ok[hit] = T['target'][order[pos[hit]]] == E['target_roi'][hit]
d_box[hit & ~tgt_ok] = np.nan  # target disagreement -> no geometry
res = dict(n_events=int(len(key_E)), n_with_geometry=int(np.isfinite(d_box).sum()),
           n_dataset_without_run1_animal=int((ev_a < 0).sum()), n_target_mismatch=int((hit & ~tgt_ok).sum()),
           datasets_without_geometry=sorted({int(k) for k in E['ds'][ev_a < 0]}),
           frac_dbox_lt_2p5=round(float(np.nanmean(d_box < 2.5)), 4), frac_dbox_lt_3p5=round(float(np.nanmean(d_box < 3.5)), 4))
np.savez_compressed(f'{OUT}/events_geom.npz', d_box=d_box)
json.dump(res, open(f'{OUT}/pq05_geometry.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
