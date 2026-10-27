# Mechanism adjudication policy (implemented: `mechanism/adjudicator.py`)

Evidence grade of an outcome: CONFIRMATION (stage CONFIRMATION & UNTOUCHED), PARTIAL (CONFIRMATION & PARTIAL, or
DISCOVERY & UNTOUCHED), EXPLORATORY (DISCOVERY with PARTIAL/CONTAMINATED, HISTORICAL).
Per family: conflicting relations → unresolved (count for nothing); otherwise one relation at the best grade.

Hypothesis state, first rule that matches:
1. FALSIFIED_WITHIN_MODEL — ≥ 1 strong contradiction at CONFIRMATION grade.
2. STRONGLY_DISFAVORED — ≥ 2 strong PARTIAL, or 1 strong PARTIAL + 1 weak CONFIRMATION.
3. WEAKENED — any other contradiction at PARTIAL/CONFIRMATION, or a strong EXPLORATORY contradiction.
4. SUPPORTED — ≥ 2 supporting CONFIRMATION families, no PARTIAL+ contradiction.
5. WEAKLY_SUPPORTED — ≥ 1 supporting PARTIAL+ family, no PARTIAL+ contradiction.
6. NEUTRAL — only exploratory support / touches nothing committed.
7. UNTESTED.
Overlay NON_IDENTIFIABLE — survivors whose committed predictions agree on every decided test with no usable
separating test left.

Pre-registered mixture qualifies when both components are contradicted at CONFIRMATION grade on tests where the
other component is supported at CONFIRMATION grade, and the mixture has no contradiction.

Verdict order: INSUFFICIENT_SIGNAL (phenomenon test never decided POS) → CONTINUE (a permitted discriminating
test remains) → MULTIPLE_MECHANISMS_REQUIRED (qualifying mixture, no single SUPPORTED) → MECHANISM_SUPPORTED
(exactly one SUPPORTED, no weakly supported rival, no non-identifiable pair) → MECHANISM_FALSIFIED_WITHIN_TESTED_MODEL
/ MECHANISM_DISFAVORED (no survivors) → NON_IDENTIFIABLE_FROM_AVAILABLE_DATA → MEASUREMENT_MODEL_INADEQUATE (tests
blocked by UNKNOWN fields) → NEW_EXPERIMENT_REQUIRED. The default under non-discriminating evidence is uncertainty.

## Policy 1.1 (declared 2026-09-16 after the g0 real run; conservative)
Motivation (recorded, not hidden): under 1.0 the g0 run returned MECHANISM_DISFAVORED because every single hypothesis carried
at least one contradiction (weak, partial or exploratory), and 1.0 did not count WEAKENED hypotheses as survivors. That label
overstates what the evidence says (nothing was falsified at confirmation grade) and suppressed experiment design.
Changes (apply to runs that declare `policy="1.1"`; g0 stays adjudicated under 1.0 and replays deterministically):
* WEAKENED hypotheses remain unresolved survivors.
* MECHANISM_DISFAVORED / MECHANISM_FALSIFIED_WITHIN_TESTED_MODEL only when no single hypothesis is WEAKENED or better.
* With ≥ 2 survivors: NON_IDENTIFIABLE_FROM_AVAILABLE_DATA if some survivor pair has no separating test at all (or the
  1.0 non-identifiable overlay fires); MEASUREMENT_MODEL_INADEQUATE only if EVERY survivor pair's separating tests are all
  blocked by UNKNOWN measurement fields; otherwise NEW_EXPERIMENT_REQUIRED, with a per-pair report of separating/blocked tests.
* Experiment proposals are generated for DISFAVORED/FALSIFIED verdicts too.
**RETRACTED CLAIM:** this section originally said "the change can only move a verdict toward less commitment". That was false: by counting WEAKENED rivals as survivors without requiring separation from them, 1.1 let a hypothesis supported only by predictions it shares with a weakened rival reach MECHANISM_SUPPORTED (this happened in the g1 real run; reproduced by fixture S13). Fixture S11 checks the DISFAVORED case.

## Policy 1.2 (declared 2026-09-16 after the g1 real run; strictly adds requirements to support)
Motivation (recorded): g1 under 1.1 returned MECHANISM_SUPPORTED for H3.g1 although H3's NECESSARY discriminating prediction (T1)
was INCONCLUSIVE and every CONFIRMATION-grade support was a SUPPORTIVE prediction shared with H1.g1, whose only contradiction was
an EXPLORATORY-grade imported result that cleared its floor by 0.0003.
Additions to 1.1:
* SUPPORTED state requires, besides >= 2 supporting CONFIRMATION families, at least one CONFIRMATION-grade supporting family on a
  NECESSARY prediction; otherwise the state is at most WEAKLY_SUPPORTED.
* MECHANISM_SUPPORTED requires, for every other surviving single hypothesis (WEAKENED included), a DECIDED CONFIRMATION-grade
  outcome on a test where their committed predictions differ and the leader's prediction matched. Unseparated rivals →
  NON_IDENTIFIABLE_FROM_AVAILABLE_DATA with the unseparated pairs listed.
Because 1.2 only adds conditions for SUPPORTED / MECHANISM_SUPPORTED on top of 1.1's survivor logic, it cannot create a support
verdict that 1.1 would not have produced. Fixtures: S13 (the g1 defect: 1.1 → MECHANISM_SUPPORTED, 1.2 → not supported),
S1 under 1.2 (a world where confirmation-grade outcomes do separate H2 from every rival still yields MECHANISM_SUPPORTED).
Runs already completed keep their recorded policy; re-adjudications under 1.2 are published as POST-HOC INTERPRETATION child
capsules, never as new evidence.
