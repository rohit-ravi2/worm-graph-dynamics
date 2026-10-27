# Hostile adjudicator audit and policy 1.3 (§29, §30, §53)

Date: 2026-09-17. Scope: `src/wormlab/mechanism/adjudicator.py` policies 1.0-1.2, `evidence_graph.py`, the JUDGE loop in
`engine.py`. Method: attack each rule with the construction that would make the engine claim more than the evidence
supports, write the failing case as a regression test, then add the smallest rule that closes it as **policy 1.3**.
Policies 1.0-1.2 are unchanged: the g0/g1/g2 runs replay to their stored verdicts (`wormlab mechanism adjudicate`,
deterministic_match True for all three after the change) and nothing stored in a capsule is rewritten.

Regression tests: `tests/test_mechanism_policy13.py` (8 tests, all pass).

## Findings

| # | Weakness (policies 1.0-1.2) | Concrete failure | Real consequence found | 1.3 rule | Test |
|---|---|---|---|---|---|
| A1 | A NECESSARY prediction is trusted without checking that the hypothesis' own generator produces it. | One mis-specified prediction plus one CONFIRMATION outcome gives FALSIFIED_WITHIN_MODEL. | **H2 (off-target photostimulation) was STRONGLY_DISFAVORED in g0-g2 because of "T6 POS NECESSARY".** Synthetic fixture S16 shows that a stimulation-effectiveness latent shared by the target and off-target neighbours produces a NULL target-flat near field under pure H2. The disfavoring was a spec error, not evidence. | `invalid_predictions`: (family, test) pairs that fail the synthetic prediction audit are weak. g3 also revises H2's T6 prediction to ANY, with the synthetic fixture as motivating evidence. | `test_invalid_necessary_prediction_is_weak_under_13` |
| A2 | Evidence families only merge outcomes of the same statistic family. | Three different statistics computed on the same CONFIRMATION animals, events and response variables count as three independent families. SUPPORTED needs only two. | The g3 families (T13/T16/T17/T18/T19) share animals and receivers; under 1.2 they could make a hypothesis SUPPORTED on a single partition. | Strict families: same partition, same data stream, overlapping units makes one family. SUPPORTED needs CONFIRMATION-grade NECESSARY support on at least 2 partitions (independent animals). | `test_strict_families_collapse_same_partition_same_stream`, `test_supported_requires_replication_across_partitions_under_13` |
| A3 | Mixtures only as pre-registered categorical pairs (each component contradicted where the other is supported). | A 70/30 contamination/coupling mixture (fixture S18) gives POS on the unmixed test, contradicting H1 and supporting H3. Nothing contradicts H3, so the verdict can become MECHANISM_SUPPORTED(H3) even though contamination explains ~70% of the box effect. | Not yet triggered on real data. It is exactly the regime a partial near-field biological effect would fall into. | Quantitative-mixture overlay: a CONFIRMATION-grade POS outcome carrying a contamination explained-fraction CI inside (0.2, 0.9) gives MULTIPLE_MECHANISMS_REQUIRED. | `test_quantitative_mixture_overlay_under_13` |
| A4 | Hypotheses with only ANY predictions on the phenomenon remain "survivors". | H4/H5 (chemical, polysynaptic) never commit on whether they explain a distance effect. They stay NEUTRAL and generate NON_IDENTIFIABLE pairs with real rivals. | g1/g2 survivor-pair reports list H4/H5 pairs that no proximity experiment could separate. | Explanatory relevance: only hypotheses committing POS on a phenomenon-family test (T0, T17) are paired. Others are reported as "not a rival explanation". | `test_non_explanations_are_not_paired_under_13` |
| A5 | JUDGE picks the next CONFIRMATION test from states updated by earlier CONFIRMATION outcomes, and stops at the first terminal verdict. | Selective stopping: which frozen tests are run on held-out data depends on held-out results. | The g1 run stopped at MECHANISM_SUPPORTED(H3) under 1.1 with frozen tests still unrun. | `RunConfig.judge_all`: every frozen test runs on each judge partition in fixed (id) order; adjudication happens once after all tests. | `test_adjudication_is_order_invariant` (verdict independent of outcome order) |
| A6 | DISCOVERY outcomes on UNTOUCHED families count as PARTIAL. | Statistics developed and tuned on DISCOVERY can STRONGLY_DISFAVOR a hypothesis with two DISCOVERY outcomes. | g3 statistics were debugged on DISCOVERY. | DISCOVERY is always EXPLORATORY. | `test_discovery_is_exploratory_under_13` |
| A7 | NULL calls mean "effect inside ±floor", and FALSIFIED_WITHIN_MODEL is read as "mechanism absent". | Wording drift in reports. | FINAL_REPORT wording was checked; H7 "FALSIFIED" is scoped to the global-breadth operationalization. | No code change. The claim registry carries the scope wording and a forbidden-phrase check. | `campaign.ClaimRegistry.check_text` |
| A9 | **(found in 1.3 itself, before any g3 sealed access)** Strict dependence clusters that mixed DIFFERENT tests with opposite relations were treated as "unresolved" and discarded. | Under 1.3 the g0-g2 reinterpretation left every hypothesis NEUTRAL, including H7, which g1 falsified on untouched RESERVE (T10 NULL). | Policy 1.3 would have silently erased decisive evidence. | **Policy 1.4:** a cluster contributes one support and one contradiction, each at its own best grade and strength; only a test that itself carries both relations is unresolved. | `test_policy14_mixed_cluster_keeps_decisive_contradiction` |
| A10 | Under strict families, an all-animal HISTORICAL look (e.g. the run-1 T0 on every WT animal) links disjoint CONFIRMATION and RESERVE outcomes through union-find transitivity. | Replicated contradictions on two held-out partitions count as one (H7 in g2 stays WEAKENED). | Replication across partitions becomes impossible to credit. | Strict families: HISTORICAL outcomes do not bridge different partitions (loose 1.0-1.2 rule unchanged). | `test_strict_families_historical_look_does_not_bridge_partitions` |
| A11 | The synthetic prediction audit covered only the newest generation. | g0-g2 H2 still carried "T6 POS NECESSARY", so its 1.4 reinterpretation re-falsified H2 through the invalidated prediction. | — | The audit covers every generation's predictions; (H2, T6) is invalid. | `prediction_audit.json` |
| A8 | Phenomenon gate passes on any-grade POS. | A historical exploratory POS alone passes the gate. | Not triggered (the phenomenon replicated on CONFIRMATION and RESERVE). | No change; recorded. | — |

## Reinterpretation of g0-g2 under policy 1.4 (post-hoc interpretation capsules, not new evidence)

| run | recorded (policy) | 1.4 verdict | 1.4 states |
|---|---|---|---|
| g0 | MECHANISM_DISFAVORED (1.0) | NEW_EXPERIMENT_REQUIRED | all WEAKENED |
| g1 | MECHANISM_SUPPORTED H3 (1.1) | NEW_EXPERIMENT_REQUIRED | H5, H7 FALSIFIED; H3 WEAKLY_SUPPORTED; H1/H2/H4/H6/H9 WEAKENED |
| g2 | NEW_EXPERIMENT_REQUIRED (1.2) | NEW_EXPERIMENT_REQUIRED | H4, H5, H7 STRONGLY_DISFAVORED; H1/H2/H3/H6/H9 WEAKENED |

Capsules: `results/_capsules/mechanism_<run>__interpretation__policy1.4__audit2` (parent = run capsule). The earlier
1.3 and untagged 1.4 interpretation capsules were produced before findings A9-A11 were fixed and are superseded (kept, not
deleted).

The g3 preregistration was amended (`g3_preregistration_a1.json`, amends the original's sha256) to policy 1.4 before any
CONFIRMATION/RESERVE access.

## Evidence-family dependence audit (§30)

Raw vs effective counts are computed for the g0-g2 runs and the g3 run by recomputing `EvidenceGraph.families()` (1.2) and
`families(strict=True)` (1.3) on the stored graphs. See `results/mechanism/deep_campaign/evidence_dependence_audit.json`
(script `scripts/deep_campaign/evidence_dependence_audit.py`).

## Policy versioning (§53)

* Old verdicts are never rewritten. The g0-g2 capsules keep their stored verdicts (1.0 / 1.1 / 1.2).
* Reproduction under the old policy: `wormlab mechanism adjudicate <run>` replays the stored `policy_version`.
* Reinterpretation under 1.3 is published as a separate interpretation capsule linked to its parent run capsule
  (`scripts/mechanism_reinterpret.py --policy 1.3`).
* Every 1.3 rule has a regression fixture above.
