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

## Repository layout

| Path | Contents |
|---|---|
| `src/wormlab/gd/` | data sources, identity, preprocessing, graphs, interventions, models, splits, workflow, CLI |
| `src/wormlab/evidence/` | evidence capsule: create, verify, inspect, compare |
| `scripts/gd/` | one script per benchmark (`run_A1.py`, `run_A2.py`, `run_K1.py`, `run_L2.py`, `run_L2_gates.py`) |
| `results/gd/benchmarks/` | frozen specs and results for A1, A2, K1 and L2, plus the L2 gate-stability control |
| `results/gd/splits/` | the frozen 5-fold animal split |
| `tests/` | data-layer, intervention and capsule tests |

## License

MIT; see `LICENSE`. The datasets keep their own licences.
