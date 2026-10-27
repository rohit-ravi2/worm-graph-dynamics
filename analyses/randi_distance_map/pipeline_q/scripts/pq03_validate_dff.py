"""PQ03: validate the event reconstruction against published per-trial dF/F (funatlas.h5 wt/dFF_all) and
recover the published stimulus inclusion set.

funatlas.h5 wt/dFF_all[i,j] holds the per-event <dF/F0>_30s values behind the published map
(funatlas_plot_intensity_map.py:172-215 with --correct-decaying): only events that passed the autoresponse
requirement (Methods 'Inclusion criteria'; Funatlas.get_occurrence_matrix req_auto_response, Funatlas.py:778-781)
and the nan check. Each of our events (every labeled responder x labeled-target stim) is matched to the list for
its (i,j) with |rel diff| < 1e-9. Chance float coincidences at that tolerance are negligible.
Outputs: out/events_match.npz (per-event match flag), out/pq03_validation.json (match rates, inferred stim
inclusion, nan-threshold consistency).
"""
import json, os, sys
import numpy as np, h5py
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from pq02_features import FA, IDS

OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
E = dict(np.load(f'{OUT}/events_wt.npz'))
E['pre_zero'] = E['pre_zero'].astype(bool)
n = len(E['ds'])
with h5py.File(FA, 'r') as f:
    DA = f['wt/dFF_all']; occ1 = f['wt/occ1'][:]
    pub = {}
    for i, j in zip(*np.where(occ1 > 0)):
        pub[(i, j)] = np.asarray(DA[i, j][:], float)

# resolve aj for stims whose target was manually located (-3): pick the atlas column whose published lists
# contain the most exact matches of that stim's responder values
from collections import Counter
vindex = defaultdict(set)
for (i, j), Lv in pub.items():
    for v in Lv:
        vindex[float('%.9g' % v)].add((i, j))
resolved = {}
for sk in np.unique((E['ds'] * 1000 + E['stim'])[E['aj'] == -3]):
    sel = np.where((E['ds'] * 1000 + E['stim']) == sk)[0]
    c = Counter()
    for e in sel:
        if np.isfinite(E['dFF'][e]):
            for (i, j) in vindex.get(float('%.9g' % E['dFF'][e]), ()):
                if i == E['ai'][e]:
                    c[j] += 1
    best = c.most_common(1)
    resolved[int(sk)] = dict(aj=int(best[0][0]) if best else -1, name=IDS[best[0][0]] if best else None,
                             n_hits=int(best[0][1]) if best else 0, n_events=int(len(sel)))
    E['aj'][sel] = resolved[int(sk)]['aj']

matched = np.zeros(n, bool)
used = {k: np.zeros(len(v), bool) for k, v in pub.items()}
order = np.lexsort((E['roi'], E['stim'], E['ds']))
for e in order:
    key = (E['ai'][e], E['aj'][e]); v = E['dFF'][e]
    if key not in pub or not np.isfinite(v):
        continue
    L = pub[key]
    hit = np.where((np.abs(L - v) <= 1e-9 * max(1.0, abs(v))) & ~used[key])[0]
    if len(hit):
        matched[e] = True; used[key][hit[0]] = True

# stim inclusion: a stim is 'published-included' if any of its events matched
stim_key = E['ds'] * 1000 + E['stim']
inc_stims = set(np.unique(stim_key[matched]))
included = np.isin(stim_key, list(inc_stims))
n_stims = len(np.unique(stim_key))

# published values never matched by us
unmatched_pub = sum(int((~u).sum()) for u in used.values())
total_pub = sum(len(v) for v in pub.values())

# occ1 check: included-stim inclusive counts vs published occ1
O = np.zeros((300, 300), int)
np.add.at(O, (E['ai'][included], E['aj'][included]), 1)
m = (O > 0) | (occ1 > 0)

# nan threshold consistency among included events, columns with DANDI mask vs without
prep_src = {}
for ds in np.unique(E['ds']):
    prep_src[ds] = np.load(f'{OUT}/prep/wt_{ds}.npz')['mask_src']
has_mask = np.array([prep_src[d][r] == 'dandi' for d, r in zip(E['ds'], E['roi'])])
res = dict(resolved_minus3_stims=resolved, n_minus3_resolved=int(sum(r['aj'] >= 0 for r in resolved.values())), n_events_all_labeled_stims=int(n), n_labeled_stims=int(n_stims),
           published_values_total=int(total_pub), published_values_matched=int(total_pub - unmatched_pub),
           frac_published_values_reproduced_exactly=round(1 - unmatched_pub / total_pub, 5),
           n_stims_inferred_included=len(inc_stims), frac_stims_included=round(len(inc_stims) / n_stims, 4),
           occ1_inferred_equals_published_frac_pairs=round(float((O[m] == occ1[m]).mean()), 5),
           occ1_inferred_sum=int(O.sum()), occ1_published_sum=int(occ1.sum()))
nth = {}
for label, sel in [('dandi_mask_cols', has_mask), ('no_mask_cols', ~has_mask)]:
    s = included & sel
    fr = E['nanrun'][s] / E['seglen'][s]
    d = {}
    for th in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01]:
        pred = (E['nanrun'][s] <= th * E['seglen'][s]) & ~E['pre_zero'][s]
        d[str(th)] = dict(pred_in=int(pred.sum()), agree=round(float((pred == matched[s]).mean()), 5))
    d['n_included_events'] = int(s.sum()); d['n_matched'] = int(matched[s].sum())
    d['matched_max_nanrun_frac'] = round(float(fr[matched[s]].max()) if matched[s].any() else -1, 4)
    d['unmatched_min_nanrun_frac'] = round(float(fr[~matched[s]].min()) if (~matched[s]).any() else -1, 4)
    nth[label] = d
res['nan_threshold_scan_on_included_events'] = nth
print(json.dumps(res, indent=1))
np.savez_compressed(f'{OUT}/events_match.npz', matched=matched, included=included, has_mask=has_mask, aj_resolved=E['aj'])
json.dump(res, open(f'{OUT}/pq03_validation.json', 'w'), indent=1)
