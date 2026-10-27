"""PQ10: render report tables (markdown) from committed JSON outputs -> out/tables.md. No new computation."""
import json, os

OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
J = lambda f: json.load(open(f'{OUT}/{f}'))
L = []

v = J('pq03_validation.json')
L += ['## B0. Event reconstruction vs published per-trial dF/F (pq03_validation.json)', '',
      '| metric | value |', '|---|---|',
      f"| published per-event dF/F values | {v['published_values_total']} |",
      f"| reproduced exactly (rel. diff < 1e-9) | {v['published_values_matched']} ({v['frac_published_values_reproduced_exactly']}) |",
      f"| labeled stims / inferred autoresponse-included | {v['n_labeled_stims']} / {v['n_stims_inferred_included']} |",
      f"| occ1 (inferred inclusion) == published, frac of pairs | {v['occ1_inferred_equals_published_frac_pairs']} (sum {v['occ1_inferred_sum']} vs {v['occ1_published_sum']}) |",
      f"| nan_th=0.05 predicts membership (DANDI-mask columns) | agree {v['nan_threshold_scan_on_included_events']['dandi_mask_cols']['0.05']['agree']} (0.3: {v['nan_threshold_scan_on_included_events']['dandi_mask_cols']['0.3']['agree']}) |", '']

L += ['## B1. q reproduction by null substitute (oracle membership; pq04_repro_oracle_<null>.json)', '',
      '| null | null n | pi0 | n sig (pub 1310) | pub sig recovered | ours in pub | Jaccard | Jaccard off-diag | Spearman -log q | q_eq Jaccard (n_eq ours / pub 13398) |',
      '|---|---|---|---|---|---|---|---|---|---|']
for nl in ['spont', 'spont62', 'wtall', 'wtallx0.5', 'wtallx0.6', 'wtallx0.7', 'wtallx0.8', 'wtqeq', 'wtqgt0.3', 'wtqgt0.4', 'wtqgt0.45', 'wtqgt0.5', 'wtqgt0.55', 'wtqgt0.6']:
    r = J(f'pq04_repro_oracle_{nl}.json'); q = r['q']
    L.append(f"| {nl} | {r['null_n']} | {r['pi0']:.3f} | {q['n_sig_ours']} | {q['frac_pub_sig_recovered']} | {q['frac_ours_sig_in_pub']} | {q['jaccard']} | {q['jaccard_offdiag']} | {q['spearman_neglog10q_both_finite']} | {r['q_eq']['jaccard']} ({r['q_eq']['n_eq_ours']}; pi0_eq {r['pi0_eq']:.3g}) |")
r = J('pq04_repro_own_wtqgt0.5.json'); q = r['q']
L += ['', f"Own membership (no oracle; our NaN masks) with wtqgt0.5: n sig {q['n_sig_ours']}, recovered {q['frac_pub_sig_recovered']}, precision {q['frac_ours_sig_in_pub']}, Jaccard {q['jaccard']}, Spearman {q['spearman_neglog10q_both_finite']}; per-pair valid-event count equals published dFF_all length for {r['n_valid_events_equals_published_dFF_all_len_frac_pairs']} of pairs (oracle: {J('pq04_repro_oracle_wtqgt0.5.json')['n_valid_events_equals_published_dFF_all_len_frac_pairs']}).", '']

for nl in ['wtqgt0.5', 'wtallx0.7']:
    if not os.path.exists(f'{OUT}/pq06_{nl}.json'):
        continue
    r = J(f'pq06_{nl}.json')
    L += [f'## C1. Regimes, null={nl} (pq06_{nl}.json; p* = {r["pstar_from_A_geo"]:.3g})', '',
          '| regime | events | pi0 | testable off-diag pairs | q<0.05 off-diag | p<=p* off-diag | q-sig also in published |', '|---|---|---|---|---|---|---|']
    for k in ['A', 'A_geo', 'B', 'C']:
        x = r[k]
        L.append(f"| {k} | {x['n_events']} | {x['pi0']:.3f} | {x['n_testable_offdiag']} | {x['n_sig_offdiag']} | {x['n_p_leq_pstar_offdiag']} | {x['n_sig_in_published']} |")
    L += ['', '| control | draws | events dropped | q<0.05 mean [2.5, 97.5] (min-max) | frac draws <= regime | p<=p* mean [2.5, 97.5] | frac draws <= regime (p) | testable mean |', '|---|---|---|---|---|---|---|---|']
    for k in ['B_ctrl_rand', 'B_ctrl_pairmatch', 'C_ctrl_rand', 'C_ctrl_pairmatch']:
        x = r[k]
        L.append(f"| {k} | {r['n_draws']} | {x['n_events_dropped']} | {x['n_sig_offdiag_mean']} [{x['n_sig_offdiag_p2_5']}, {x['n_sig_offdiag_p97_5']}] ({x['n_sig_offdiag_min']}-{x['n_sig_offdiag_max']}) | {x['frac_draws_leq_regime']} | {x['n_p_leq_pstar_mean']} [{x['n_p_leq_pstar_p2_5']}, {x['n_p_leq_pstar_p97_5']}] | {x['frac_draws_p_leq_regime']} | {x['n_testable_mean']} |")
    L += ['', f"B survivors (q<0.05, n={len(r['B']['survivors'])}): " + ', '.join(r['B']['survivors']), '',
          f"C survivors (q<0.05, n={len(r['C']['survivors'])}): " + ', '.join(r['C']['survivors']), '']
    if os.path.exists(f'{OUT}/pq09_{nl}.json'):
        c = J(f'pq09_{nl}.json')
        L += [f'### C2. Composition of significant pairs by per-animal proximity (pq09_{nl}.json)', '',
              '| cut | map | category | n sig pairs | survive regime (q) | pairmatch survival (q) | rand survival (q) | p<=p* in regime | pairmatch (p) | rand (p) |', '|---|---|---|---|---|---|---|---|---|---|']
        for cut in ['2.5', '3.5']:
            for mp in ['A_geo', 'published']:
                d = c[f'cut{cut}_{mp}']
                for cat in ['all_far', 'minority_near', 'majority_near', 'all_near']:
                    x = d[cat]
                    L.append(f"| {cut} | {mp} (n={d['n_sig_total']}, median near frac {d['median_frac_near_events']}) | {cat} | {x['n_pairs']} | {x['n_survive_regime']} | {x['mean_survival_pairmatch']} | {x['mean_survival_rand']} | {x['n_p_leq_pstar_regime']} | {x['mean_p_survival_pairmatch']} | {x['mean_p_survival_rand']} |")
        L.append('')
    if os.path.exists(f'{OUT}/pq08_{nl}.json'):
        w = J(f'pq08_{nl}.json')
        L += [f'### C3. Within-pair near vs far, Randi features (pq08_{nl}.json)', '',
              '| cut | stratum | feature | pairs | near / far events | near rate | far rate | pooled ratio [95% CI] | mean pair diff [95% CI] |', '|---|---|---|---|---|---|---|---|---|']
        for cut in ['2.5', '3.5']:
            for st in ['all_mixed', 'published_sig', 'ours_A_sig']:
                for ft in ['dff', 'sd', 'either']:
                    x = w[f'cut{cut}_{st}_{ft}']
                    L.append(f"| {cut} | {st} | {ft} | {x['n_pairs']} | {x['n_near_events']} / {x['n_far_events']} | {x['pooled_near_rate']} | {x['pooled_far_rate']} | {x['pooled_ratio']} {x['pooled_ratio_ci95']} | {x['mean_pair_diff']} {x['mean_pair_diff_ci95']} |")
        L.append('')
    if os.path.exists(f'{OUT}/pq07_{nl}.json'):
        s = J(f'pq07_{nl}.json')
        keys = ['dist', 'conn', 'hop2', 'lr_dist', 'lr_conn', 'lr_conn_dist', 'dist_minus_conn', 'lr_conn_dist_minus_lr_dist']
        L += [f'### C4. Predictor within-source AUROC (pq07_{nl}.json)', '',
              '| map | pairs | sig | sources | ' + ' | '.join(keys) + ' |', '|' + '---|' * (4 + len(keys))]
        for k in ['published', 'A', 'A_geo', 'B', 'C', 'A_p_leq_pstar', 'A_geo_p_leq_pstar', 'B_p_leq_pstar', 'C_p_leq_pstar']:
            x = s.get(k)
            if not x:
                continue
            cells = [f"{x[kk]['mean']} {x[kk].get('ci95', '')}" if kk in x else '-' for kk in keys]
            L.append(f"| {k} | {x['n_pairs']} | {x['n_sig']} | {x['n_sources_scored']} | " + ' | '.join(cells) + ' |')
        for k in ['B_ctrl_rand', 'B_ctrl_pairmatch', 'C_ctrl_rand', 'C_ctrl_pairmatch', 'B_ctrl_rand_p_leq_pstar', 'B_ctrl_pairmatch_p_leq_pstar', 'C_ctrl_rand_p_leq_pstar', 'C_ctrl_pairmatch_p_leq_pstar']:
            x = s.get(k)
            if not x:
                continue
            cells = [f"{x[kk]['mean_over_draws']} [{x[kk]['p2_5']}, {x[kk]['p97_5']}]" if kk in x else '-' for kk in keys]
            L.append(f"| {k} (draws={x['n_draws']}) | - | - | {x['n_sources_scored_mean']} | " + ' | '.join(cells) + ' |')
        L.append('')
open(f'{OUT}/tables.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
