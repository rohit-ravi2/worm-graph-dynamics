"""L2 gate-stability CONTROL: is cross-fold edge-gate stability a property of the connectome, or of training?

The L2 run saved gates for the connectome model only (fold-pairwise corr 0.61 chem / 0.79 gap). That number is not
interpretable until the same statistic is computed for graphs with no biological content. This script retrains, with
the identical frozen config and seeds (1000 + fold), the connectome model and the matched controls
(gnn_shuffled_0..2, gnn_geometry) and saves, per model and fold: the gates, the free per-edge offsets u_ij (the part
of the gate not explained by the edge's synapse count), and the per-test-animal R^2 (a replay check against fold*.json).
Stability statistics (aggregate): fold-pairwise Pearson/Spearman of gates over edges, the same for u_ij, and the
gate-vs-log1p(weight) correlation. Nothing in L2_spec.json changes; outputs go to L2/gate_controls/.
Usage: run_L2_gates.py <fold> | run_L2_gates.py aggregate
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_L2 as L2  # noqa: E402
from wormlab.gd import preprocess as P  # noqa: E402
from wormlab.gd.bench import forecast as F, splits as SPL  # noqa: E402
from wormlab.gd.models.linear import Persistence  # noqa: E402
from wormlab.gd.models.neural import TypedGNN  # noqa: E402
from wormlab.gd.sources import atanas2023 as A  # noqa: E402

OUT = L2.OUT / "gate_controls"
MODELS = ("gnn_connectome", "gnn_shuffled_0", "gnn_shuffled_1", "gnn_shuffled_2", "gnn_geometry")


def run_fold(fold):
    t0 = time.time()
    recs = [P.standardize(r) for r in A.load()]
    nodes = P.node_space(recs, 0.5)
    split = SPL.load("A2_atanas_5fold")
    spec = json.loads((L2.OUT / "L2_spec.json").read_text())
    assert spec["nodes"] == nodes and spec["split_sha256"] == split["sha256"] and spec["h"] == L2.H \
        and spec["epochs"] == L2.EPOCHS and spec["lr"] == L2.LR and spec["gate_l1"] == L2.L1, "spec drift"
    gs = L2.graphs(nodes, recs)
    data = {r.animal_id: P.to_nodes(r, nodes) for r in recs}
    train = [data[a] for a, f in split["fold_of"].items() if f != fold]
    test = [a for a, f in split["fold_of"].items() if f == fold]
    pers = Persistence().fit(train, L2.H)
    OUT.mkdir(parents=True, exist_ok=True)
    out = dict(fold=fold, seed=1000 + fold, models={})
    for name in MODELS:
        cw, gw = gs[name]
        m = TypedGNN(cw, gw, epochs=L2.EPOCHS, lr=L2.LR, l1=L2.L1, seed=1000 + fold).fit(train, L2.H)
        res = {a: F.score_animal(*data[a], m.predict(*data[a]), pers.predict(*data[a]), L2.H) for a in test}
        gc, gg = m.gate_report()
        out["models"][name] = dict(
            r2={a: (v["r2"] if v else None) for a, v in res.items()},
            gates=dict(chem=gc.tolist(), gap=gg.tolist()),
            u=dict(chem=m.net.u_chem.detach().cpu().numpy().tolist(), gap=m.net.u_gap.detach().cpu().numpy().tolist()),
            weights=dict(chem=np.asarray(cw).tolist(), gap=np.asarray(gw).tolist()))
        print(f"fold {fold} {name} {np.mean([v['r2'] for v in res.values() if v]):.4f} {time.time()-t0:.0f}s", flush=True)
        (OUT / f"fold{fold}.json").write_text(json.dumps(out, default=float))   # checkpoint after every model


def _stab(mats, edge):
    V = np.array([m[edge] for m in mats])                       # (folds, n_edges)
    if V.shape[1] == 0:
        return None
    iu = np.triu_indices(len(V), 1)
    pear = np.corrcoef(V)[iu]
    spear = np.array([spearmanr(V[i], V[j]).correlation for i, j in zip(*iu)])
    return dict(n_edges=int(V.shape[1]), pearson_mean=float(pear.mean()), pearson_min=float(pear.min()),
                spearman_mean=float(spear.mean()), mean=float(V.mean()), sd_across_edges=float(V.mean(0).std()))


def aggregate():
    folds = [json.loads((OUT / f"fold{f}.json").read_text()) for f in range(5)]
    orig = [json.loads((L2.OUT / f"fold{f}.json").read_text()) for f in range(5)]
    report = {}
    for name in MODELS:
        if not all(name in d["models"] for d in folds):
            continue
        rep = {}
        for ch in ("chem", "gap"):
            W = np.array(folds[0]["models"][name]["weights"][ch]); E = W > 0
            if ch == "gap":
                E = E & np.triu(np.ones_like(E, bool), 1)           # undirected: count each gap edge once
            G_ = [np.array(d["models"][name]["gates"][ch]) for d in folds]
            U_ = [np.array(d["models"][name]["u"][ch]) for d in folds]
            gate_vs_w = [float(np.corrcoef(g[E], np.log1p(W[E]))[0, 1]) for g in G_]
            rep[ch] = dict(gates=_stab(G_, E), u_offsets=_stab(U_, E),
                           gate_vs_log1p_weight_corr=dict(mean=float(np.mean(gate_vs_w)), per_fold=gate_vs_w))
        # replay check against the committed L2 fold results
        dev = []
        for d, o in zip(folds, orig):
            for a, r in d["models"][name]["r2"].items():
                r0 = o["results"][name].get(a)
                if r is not None and r0:
                    dev.append(r - r0["r2"])
        rep["replay_r2_minus_original"] = dict(mean=float(np.mean(dev)), max_abs=float(np.max(np.abs(dev))), n=len(dev))
        report[name] = rep
    # contrast: connectome stability minus the mean over shuffles
    shuf = [m for m in report if m.startswith("gnn_shuffled")]
    if "gnn_connectome" in report and shuf:
        report["contrast"] = {ch: {q: dict(
            connectome=report["gnn_connectome"][ch][q]["pearson_mean"],
            shuffled_mean=float(np.mean([report[s][ch][q]["pearson_mean"] for s in shuf])),
            shuffled_each=[report[s][ch][q]["pearson_mean"] for s in shuf],
            geometry=report["gnn_geometry"][ch][q]["pearson_mean"] if "gnn_geometry" in report else None)
            for q in ("gates", "u_offsets")} for ch in ("chem", "gap")}
    (L2.OUT / "L2_gate_stability.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report.get("contrast", report), indent=1, default=float))


if __name__ == "__main__":
    aggregate() if sys.argv[1] == "aggregate" else run_fold(int(sys.argv[1]))
