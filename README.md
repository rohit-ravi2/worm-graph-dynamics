# Worm graph dynamics

Does the C. elegans wiring diagram help predict what the nervous system does next? I built frozen, held-out benchmarks on
two public whole-brain imaging datasets (Atanas et al. 2023, Kato et al. 2015) and tested the connectome against matched
null graphs with linear and learned models. In a learned graph neural network evaluated on 38 held-out animals, the real
connectome scores R² 0.5776 and three degree-preserving shuffles of it score 0.5770, 0.5783 and 0.5771 (difference
+0.0001, 95% CI [−0.0007, +0.0009]). A graph of each neuron's nearest physical neighbours does better than the connectome.

That is a clean negative result for the learned model, the linear model points the same way (physical neighbours match or
beat the connectome), and a perturbation benchmark on the second dataset gives no sign that the connectome is special.
Every benchmark spec, split
and result is frozen and hashed before scoring, and each run is sealed in an evidence capsule that fails verification if
a single byte changes.

The same question on the causal side: in the Randi et al. 2023 optogenetic signal-propagation atlas, the physical
distance between two neurons predicts whether stimulating one evokes a response in the other far better than the
connectome does (AUROC 0.727 vs 0.566), and the connectome adds nothing once distance is known. Most of that proximity
effect is not a simple optical copy of the stimulated neuron's signal, and its biological share is still open. That
work is in the "Randi atlas" section below.

## Results

All numbers are per-animal pooled R² (or AUROC) on held-out animals, with animal-bootstrap 95% confidence intervals and
paired comparisons, from `results/gd/benchmarks/`.

### L2: learned typed GNN, Atanas 2023 (38 animals, 5-fold animal split, 2.4 s ahead)

The architecture is fixed and only the edge structure changes: per-node recurrent state, separate message maps for
chemical (directed) and gap-junction (undirected) edges, and a learned gate per edge with an L1 penalty. 89 neurons with
canonical NeuroPAL identities.

| Graph | Held-out R² | 95% CI |
|---|---|---|
| Nearest physical neighbours (k matched to the connectome) | 0.5799 | [0.5648, 0.5952] |
| Degree-preserving shuffle 1 | 0.5783 | [0.5627, 0.5944] |
| Connectome (White 1986 + Witvliet, via `wormneuroatlas`) | 0.5776 | [0.5622, 0.5936] |
| Degree-preserving shuffle 2 | 0.5771 | [0.5617, 0.5930] |
| Degree-preserving shuffle 0 | 0.5770 | [0.5616, 0.5927] |
| No edges | 0.5742 | [0.5586, 0.5902] |
| Complete graph | −0.1685 | [−0.2466, −0.0980] |

Paired contrasts (`L2_results.json`):

| Contrast | Mean | 95% CI |
|---|---|---|
| connectome − shuffled | +0.0001 | [−0.0007, +0.0009] |
| connectome − physical neighbours | −0.0023 | [−0.0036, −0.0007] |
| connectome − no edges | +0.0034 | [+0.0017, +0.0051] |
| GNN − linear model on the same connectome | +0.0147 | [+0.0132, +0.0162] |

So some sparse graph helps a little, which sparse graph does not matter, the nonlinearity is worth about 1.5 points of
R², and dense message passing at this sample size is destructive.

I also checked whether the learned edge gates could be read as mechanistic candidates (`L2_gate_stability.json`). On
the connectome they are fairly stable across folds (mean pairwise Pearson 0.61 for chemical edges, 0.78 for gap
junctions), but gates learned on shuffled graphs are just as stable (chemical 0.38, 0.64, 0.76; gap 0.74, 0.82, 0.78).
Gate stability is a property of training, not of the wiring, so the gates do not identify important synapses.

### A2: linear forecasting, Atanas 2023

Node-specific ridge regression on five lags of a source set, at 0.6, 2.4 and 6 s ahead. Here the connectome does beat
degree-preserving shuffles by a small, significant margin (+0.0016, +0.0023 and +0.0015 R²; all CIs exclude zero), but
k nearest physical neighbours match or beat it at every horizon (connectome − neighbours = −0.0004, −0.0006, −0.0016;
the 2.4 s interval [−0.0015, +0.0002] includes zero). Adding connectome neighbours to a neuron's own history adds +0.003
to +0.017,
and a dense all-to-all VAR does worse than the sparse connectome model.

### A1: does neural state predict behaviour beyond behaviour history? Atanas 2023

Strictly past-only models, comparing behaviour history alone with behaviour history plus neural activity. Reversal state
is the robust case: neural activity adds +0.036 AUROC concurrently, +0.019 at 0.6 s, +0.033 at 2.4 s and +0.035 at 6 s
ahead, every interval excluding zero. Velocity gains +0.016 R² concurrently and +0.003 at 0.6 s; head curvature only
concurrently (+0.047).

### K1: held-out perturbation, Kato 2015

Trained on wild-type dynamics (12 recordings), each model simulates AVA silenced by clamping it, and the predicted change
in each of 20 other neurons is compared with the observed AVA::HisCl recordings. The manipulation check passes (AVA
log-variance ratio −1.44 and −2.26 for the two AVA neurons), but rank correlations spread widely (connectome 0.39, shuffles
0.29 to 0.52, random graphs 0.04 to 0.52, physical neighbours 0.52). With five animals per arm this benchmark cannot
separate the graphs, and I report it as uninformative rather than as evidence either way.

## Randi atlas: proximity, not wiring

Randi et al. (2023) stimulated single neurons optogenetically in over a hundred animals and called a pair (stimulated j,
responder i) connected when the responder's per-event responses differ from a no-stimulation null. I rebuilt their
significance statistic from per-trial data, then asked what predicts the published map and what the near-neighbour
responses are made of.

### Reproducing the atlas statistic

`analyses/randi_distance_map/pipeline_q/` re-implements the published pipeline (`randi_q_lib.py` ports the relevant
functions from the authors' `pumpprobe` and `wormdatamodel` code and `multipy`, citing source lines; code-reading notes
in `METHODS_randi_q.md`). On the authors' text exports, 108,116 of 109,271 published per-trial ΔF/F values (98.9%) are
reproduced bit-for-bit (`out/pq03_validation.json`). The authors' no-stimulation null recordings are not public, so the
q values can only be reproduced up to the null: with a substitute null chosen to match the published π0, my map
recovers 88.5% of the published significant pairs, 98.4% of mine are published, Jaccard 0.872, Spearman 0.989 on
−log q (`out/pq04_repro_oracle_wtqgt0.5.json`).

### Distance predicts the published map; the connectome does not add to it

Per-source AUROC for predicting which pairs are significant in the published map, averaged over 158 stimulated neurons
(23,902 pairs, 1,151 significant; `pq07_predictors.py` → `out/pq07_wtqgt0.5.json`, block `published`):

| predictor | AUROC | 95% CI |
|---|---|---|
| atlas distance | 0.727 | [0.703, 0.749] |
| two-hop paths in the connectome | 0.628 | [0.607, 0.650] |
| connectome weight | 0.566 | [0.550, 0.584] |
| logistic, connectome + distance, minus distance alone | −0.0005 | [−0.0068, +0.0057] |

When events with the responder physically next to the target (under 2.5 extraction-box units in that animal) are
removed and the map is rebuilt, connectome + distance does beat distance by +0.030 AUROC, but removing the same number
of events per pair at random gives +0.023 on average (95% range 0.017 to 0.032; 4 of 50 control maps reach the observed
value; `pq07b_lr_pairmatch_control.py` → `out/pq07b_wtqgt0.5.json`, regime `B`). Most of that apparent connectome gain is
reproduced by event loss alone.

Within a single labelled pair, the same responder responds more often on trials where it happens to sit near the
target. For the 345 published-significant pairs that have both near and far events, the rate of events outside the
null's central 95% is 0.235 near vs 0.134 far, a ratio of 1.75× [1.48, 2.11] (pair bootstrap;
`pq08_within_pair.py` → `out/pq08_wtqgt0.5.json`, `cut2.5_published_sig_dff`). Identity is held fixed here, so this is a
proximity effect, not a property of which neurons are paired.

### What the near-field responses are made of

To ask why proximity matters, I built a mechanism engine (`src/wormlab/mechanism/`, design notes in
`docs/mechanism_engine/`): competing hypotheses (optical contamination, off-target photostimulation, electrical
coupling, chemical and polysynaptic propagation, neuromodulation, shared state, stimulation-light artifact, and their
pairwise mixtures) are written as specs that each commit to predictions; tests that cannot separate their hypotheses are
rejected at compile time; the 110 wild-type animals of DANDI:001075 are split once into DISCOVERY (24), CONFIRMATION
(40) and RESERVE (46); held-out traces are extracted into a sealed store that records only hashes, every read of a
sealed partition goes through a hash-chained access ledger, and statistics are frozen on DISCOVERY before any held-out
read. Traces come from my own extractor on the raw imaging volumes, not from the published traces. The final
preregistered run is `results/mechanism/randi_proximity/runs/randi_proximity_g3_20260917T183056Z/`; per-test results are
in `outputs/outcomes.json` and the verdict in `outputs/verdict.json`.

| held-out result (`outcomes.json`) | CONFIRMATION | RESERVE |
|---|---|---|
| proximity effect, risk difference near (< 6 µm) vs far (15–40 µm) (T0) | 0.091 [0.059, 0.124] | 0.076 [0.045, 0.105] |
| share of it explained by subtracting a static optical kernel fit to the mean image (T13) | 6.7% [−0.3%, 17.1%] | 13.1% [2.6%, 24.7%] |
| residual near-field excess after evoked-volume unmixing with a free halo (T17) | 0.064 [0.037, 0.096] | 0.060 [0.019, 0.105] |

The proximity effect replicates on both held-out partitions. The optical kernel explains 29% [18%, 45%] of it on
DISCOVERY, where it was developed, but only 7% and 13% on held-out animals. After unmixing, a near-field excess of about
0.06 remains on both partitions (called positive on CONFIRMATION; on RESERVE the CI excludes zero but the preregistered
call is inconclusive). The engine's verdict is `NEW_EXPERIMENT_REQUIRED`. Optical contamination on its own and a shared global state are
falsified within the tested model, direct chemical and polysynaptic propagation are strongly disfavoured, and the two
weakly supported explanations are mixtures of contamination with a second, local mechanism (a local field effect, or
extrasynaptic diffusion). Off-target photostimulation and electrical coupling are weakened but not excluded. The
existing recordings cannot separate the survivors, so the engine's output is the experiment that would (`outputs/experiment_proposal.json`).

### A direct test for fluctuation transfer between neighbours

If neighbouring neurons were electrically coupled, fast fluctuations in a target should transfer to its neighbours
beyond what light spill predicts. The first estimator I used for this turned out to be biased upward by light leaking
the other way, from receiver to target (`scripts/cross_rig/alpha_bias_check.py`). The corrected two-way estimator
(`alpha_twoway_real.py`) was calibrated on rendered synthetic recordings before any held-out use
(`results/mechanism/cross_rig/twoway_calibration_n6.json`): pure-contamination worlds give 0.86 to 1.17, a realistic
misspecified-kernel world 1.9 to 2.2, and a world with electrical coupling 4.3 to 5.2. On held-out animals
(`twoway_heldout_result.json`, preregistered in `twoway_heldout_prereg.json`) the median two-way ratio is 0.45 to 0.52 on
CONFIRMATION (37 animals, 532 pairs) and 0.38 to 0.48 on RESERVE (43 animals, 734 pairs) across the four time windows,
with every CI below 1. Measured fast transfer between near neighbours never exceeds what the optical kernel predicts, so
there is no sign of a coupling excess in fast fluctuations; the kernel bounds it but does not calibrate its size.

### A look-ahead check on behaviour decoding

The same audit habit applied to an earlier decoding result on Atanas 2023: a neural gain over an AR(3) speed baseline
had been read as "neural activity predicts speed 1–2 s ahead". `scripts/audits/worm_past_only_audit.py` re-ran the
frozen pipeline with the lag grid restricted. The gain is real when the neural window may include the target frame
(+0.0105 to +0.0170 R² across the 12 top representations) but with a strictly past-only window none of the 13
representations gives a positive gain above the 0.0045 floor (`results/_audits/worm_past_only_2026-09-16/past_only_audit_results.json`;
write-up in `docs/audits/WORM_PAST_ONLY_AUDIT_2026-09-16.md`). The decoded signal follows behaviour rather than leading
it. The decoding pipeline being audited is not part of this repository; the audit script and its outputs are.

## How it is built

- `wormlab.gd.sources`: typed loaders for both datasets. Atanas 2023 is streamed from DANDI at a pinned version and
  every extracted file records the DANDI asset sha256; Kato 2015 files are checked against OSF's own hashes.
- `wormlab.gd.identity`: every neuron label is normalised against 300 canonical names and tagged canonical, class_only,
  ambiguous, uncertain or unknown. Only canonical labels enter graph models; nothing is dropped silently.
- `wormlab.gd.graph`, `interventions`: the connectome as shipped by `wormneuroatlas` (autapses zeroed), plus validated,
  content-hashed interventions (`NodeClamp`, `NodeAblate`, `EdgeDelete`, `EdgeScale`, `DegreePreservingShuffle`).
- `wormlab.gd.models`: `linear.SourceSetRidge` (exact ridge per node; every source set's Gram matrix is a slice of one
  full Gram, so many graph variants are cheap to fit) and `neural.TypedGNN`.
- `wormlab.gd.bench.splits`: the 5-fold animal split is content-addressed, written read-only, and refused if a rerun
  would change it. Each benchmark script writes its spec before scoring and asserts on rerun that it has not changed.
- `wormlab.evidence`: the capsule. `wormlab gd run <BENCH>` copies spec, split, data manifests, code, environment and
  results into a content-addressed store with a single root hash; `wormlab gd verify` recomputes it.

One data caveat worth knowing: in Kato's AVA_HisCl arm the deposited behavioural state labels contradict the neural
data, so they are flagged as conflicted in provenance and never used as ground truth.

## How to run

```bash
pip install -e ".[data,models,dev]"
pytest                       # unit tests; the two data tests skip until the caches exist
wormlab gd data              # download Kato 2015, stream Atanas 2023, verify hashes, write data manifests
wormlab gd run A2            # run the frozen benchmark if results are missing, then capsule and verify
wormlab gd run L2 --rerun    # rerun from scratch; the spec and split must match the frozen ones
wormlab gd verify results/_capsules/<capsule>
```

L2 trains seven graph variants per fold; `scripts/gd/run_L2_all.sh` runs the five folds in parallel on one GPU
(`WORMLAB_DEVICE=cuda`) and took about 3.6 hours on one RTX 4060 Ti. A1, A2 and K1 run on CPU.

The Randi analyses are plain scripts run from the repository root, with the data paths given by environment variables
(defaults under `data/`, see Data):

```bash
pip install -e ".[data,randi]"
export RANDI_EXPORT_DIR=data/randi_raw RANDI_SEG_DIR=data/randi2023_dandi001075_seg RANDI_RAW_DIR=data/randi2023_dandi001075
python analyses/randi_distance_map/scripts/build_cache.py          # per-animal geometry cache
python analyses/randi_distance_map/scripts/build_responses.py
Q=analyses/randi_distance_map/pipeline_q/scripts
python $Q/pq01_preprocess.py wt && python $Q/pq02_features.py wt && python $Q/pq03_validate_dff.py
python $Q/pq04_reproduce_q.py oracle wtqgt0.5 && python $Q/pq05_geometry.py && python $Q/pq06_regimes.py wtqgt0.5
python $Q/pq07_predictors.py wtqgt0.5 && python $Q/pq07b_lr_pairmatch_control.py && python $Q/pq08_within_pair.py wtqgt0.5
pytest tests/test_mechanism_*.py   # engine unit and synthetic-recovery tests, no data needed
```

The engine runs (`scripts/mechanism_extract.py`, `provenance_extract.py`, `event_crops_extract.py`, then
`scripts/mechanism_run_randi.py`) read the frozen partition manifest in `results/mechanism/randi_proximity/` and refuse to
proceed if an extractor or preregistration hash has changed. Raw-volume extraction for all 110 animals needs the full
DANDI:001075 imaging NWBs (several hundred GB).

Data are cached under `~/.cache/wormlab`. To move the cache, or to point at an existing copy, write
`wormlab.local.toml` in the repository root:

```toml
cache_dir = "/big/disk/wormlab-cache"
mirrors = ["/path/to/an/existing/copy"]   # files whose sha256 matches are linked instead of downloaded
```

## Data

No data are included; the loaders download and verify them.

- Atanas et al. 2023, Cell: freely moving whole-brain imaging with NeuroPAL identities, 38 animals. DANDI:000776,
  version 0.241009.1509. Only traces, labels, neuron coordinates and behaviour channels are extracted.
- Kato et al. 2015, Cell: immobilised whole-brain imaging, WT_NoStim (5), WT_Stim (7) and AVA_HisCl (5). OSF node
  `2395t`.
- Connectome and neuron atlas: the `wormneuroatlas` package (White et al. 1986 and Witvliet et al., as shipped).
- Randi et al. 2023, Nature: raw imaging and segmentation NWBs from DANDI:001075 (into `data/randi2023_dandi001075/` and
  `data/randi2023_dandi001075_seg/`); the authors' text exports of traces, labels and stimulation targets for wild type,
  *unc-31* and spontaneous recordings from their OSF deposit (osf.io/9mecf, osf.io/ty6sp, osf.io/36y7t; into
  `data/randi_raw/exported_data`, `exported_data_unc31`, `spont_data`), and the full per-volume export (osf.io/e2syt,
  into `data/randi_osf_e2syt/`). The published atlas (`funatlas.h5`) and neuron positions come with `wormneuroatlas`.

## Repository layout

| Path | Contents |
|---|---|
| `src/wormlab/gd/` | data sources, identity, preprocessing, graphs, interventions, models, splits, workflow, CLI |
| `src/wormlab/evidence/` | evidence capsule: create, verify, inspect, compare |
| `scripts/gd/` | one script per benchmark (`run_A1.py`, `run_A2.py`, `run_K1.py`, `run_L2.py`, `run_L2_gates.py`) |
| `results/gd/benchmarks/` | frozen specs and results for A1, A2, K1 and L2, plus the L2 gate-stability control |
| `results/gd/splits/` | the frozen 5-fold animal split |
| `src/wormlab/mechanism/` | mechanism engine: hypothesis specs, measurement model, access firewall, contrasts, synthetic worlds, adjudicator; Randi benchmark in `randi/` |
| `analyses/randi_distance_map/` | distance maps on the Randi atlas (`scripts/`, `out/`) and the reproduced significance statistic (`pipeline_q/`) |
| `scripts/mechanism_*.py`, `scripts/*_extract.py` | sealed extraction and engine runs on the Randi data |
| `scripts/cross_rig/` | copy-ratio estimators, their synthetic calibration and the held-out two-way test |
| `scripts/audits/` | look-ahead audit of the behaviour-decoding result |
| `results/mechanism/` | partition manifest, preregistrations, the g3 run outputs, cross-rig results |
| `results/_audits/` | look-ahead audit outputs |
| `docs/` | engine design notes, adjudication policy, measurement model, look-ahead audit write-up |
| `tests/` | data-layer, intervention and capsule tests |

## Notes

`analyses/randi_distance_map/pipeline_q/scripts/randi_q_lib.py` is a port of functions from the Leifer lab's
`pumpprobe` and `wormdatamodel` and from `multipy`, with line references; those parts remain under their original
licences. Raw data, per-animal caches (`.npz`) and sealed extraction payloads are not included.

## License

MIT; see `LICENSE`. The datasets keep their own licences.
