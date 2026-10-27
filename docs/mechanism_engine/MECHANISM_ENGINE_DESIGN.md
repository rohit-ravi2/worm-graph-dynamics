# Mechanism Adjudication Engine — design (as implemented)

Package: `src/wormlab/mechanism/` (commit history in the repo's git, initialised for this build; all
pre-existing capsules untouched).

## Purpose
Turn an unexpected predictor into: explicit competing mechanisms → compiled falsifiable predictions →
ranked discriminating analyses → firewall-controlled execution → dependence-aware evidence →
bounded verdict or experiment proposal.

## Components and the bottleneck each removes

| Module | Object(s) | What can be distinguished after it exists |
|---|---|---|
| `spec.py` | `HypothesisSpec`, `Prediction`, `MechanismTestSpec`, `TestOutcome`, `mixture_of`, `compile_test_against_hypotheses` | Claims become executable: every hypothesis must carry an observation that counts against it; tests that cannot separate their claimed hypotheses are rejected at compile time (this rejected T5 until a light-artifact hypothesis H9 existed). |
| `measurement.py` | `MeasurementModelSpec` (KNOWN/INFERRED/ASSUMED/UNKNOWN), `onset_interval`, `delay_interval`, `delay_call` | Scan-order timing artifacts vs propagation; tests needing UNKNOWN metadata are blocked rather than silently run. |
| `firewall.py` | `AccessLedger` (hash-chained), `FreezeRegistry`, `AccessBroker`, `HistoricalContamination` | Prospective vs adaptive evidence; generations; single look; contaminated families. |
| `observations.py` | `Connectome`, `build_table` | One event-level table for real AND synthetic payloads (identical downstream pipeline). |
| `contrasts.py` | T0–T11 contrasts, `to_outcome` | Distance vs topology, latency, stimulation geometry, control channel, target mediation, sham, breadth, baseline covariance. |
| `generators.py`, `synthetic.py` | mechanism simulators, `SyntheticProvider` | Ground-truth fixtures S1–S6 and hostile nulls. |
| `nulls.py` | `compare` | Whether a single mechanism class reproduces the observed discovery signature. |
| `evidence_graph.py` | `EvidenceGraph`, `families`, `support_summary` | Analyses vs independent evidence families. |
| `discriminator.py` | `rank_tests`, `expected_information_gain`, `identifiability_matrix` | Which analysis to run next; `NO_AVAILABLE_TEST_HAS_DISCRIMINATING_POWER`. |
| `adjudicator.py` | `adjudicate`, `assess` | Deterministic hypothesis states and terminal verdicts. |
| `design.py` | `propose`, manipulation catalogue | Cheapest separating intervention when data cannot decide. |
| `engine.py` | `MechanismRun`, `DataProvider`, `RunConfig` | End-to-end loop with scientific termination. |
| `capsule_build.py` | `build` | Immutable, verified capsule of the whole run. |
| `cli.py` | `wormlab mechanism …` | Narrow CLI. |
| `randi/` | `raw_traces.py` (frozen extractor), `partitions.py`, `benchmark.py`, `provider.py`, `run.py` | The Randi proximity benchmark. |

## Run loop (`engine.MechanismRun.run`)
1. COMPILER: validate measurement model, hypotheses; pre-register all pairwise mixtures; compile tests
   (non-discriminating tests rejected and logged as deviations; the phenomenon gate is exempt); freeze.
2. HISTORICAL imports enter as exploratory evidence.
3. SCOUT: discriminator ranking on DISCOVERY; every test with EIG > 0 is run; adjudicate; optional
   after-scout hook (synthetic-null comparison).
4. JUDGE: phenomenon gate first; then repeatedly rank remaining tests for CONFIRMATION, run the best
   (single look per family), adjudicate; stop at a terminal verdict or when no test discriminates.
5. DESIGN: experiment proposal for non-identifiable / new-experiment / measurement-inadequate / mixture verdicts.
6. CAPSULE: `capsule_build.build` + one-byte tamper check on a copy.

## Not built (scope control)
No general-purpose framework, no causal discovery, no relational grammar, no visualisation, no model zoo.
