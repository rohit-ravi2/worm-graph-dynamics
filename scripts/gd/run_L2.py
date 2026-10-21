"""L2 (Atanas 2023): learned typed relational GNN, does the connectome help a nonlinear model?

Same architecture for every variant; only the edge structure differs (so any gain is attributable to the graph):
  gnn_connectome   chemical (directed, log-count weights) + gap-junction (undirected) edges
  gnn_none         no edges (per-node recurrent model with identity embeddings; graph-free)
  gnn_full         complete graph in both channels (weights 1)
  gnn_shuffled_s   degree-preserving shuffles of the connectome, s = 0, 1, 2 (weights travel with edges)
  gnn_geometry     k_i nearest physical neighbours (median within-animal distance, k matched to the connectome):
                   directed kNN in the chemical slot, mutual kNN in the gap slot, weights 1
Config frozen from the fold-0 inner capacity check (training animals only): 80 epochs, lr 2e-3, gate-L1 1e-3, d 16.
Horizon h = 4 frames (2.4 s); the A2 5-fold animal split; per-animal pooled R^2; the connectome model's edge gates
are saved per fold for the stability report.
Usage: run_L2.py <fold> | run_L2.py aggregate
"""
import json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_A2 as A2  # noqa: E402
from wormlab.gd import graph as G, preprocess as P  # noqa: E402
from wormlab.gd.bench import forecast as F, splits as SPL  # noqa: E402
from wormlab.gd.models.linear import Persistence  # noqa: E402
from wormlab.gd.models.neural import TypedGNN  # noqa: E402
from wormlab.gd.sources import atanas2023 as A  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/gd/benchmarks/L2"
H, EPOCHS, LR, L1, N_SHUF = 4, 80, 2e-3, 1e-3, 3


def graphs(nodes, recs):
    C = G.load_connectome()
    Cs = C.sub(nodes)
    K = len(nodes)
    out = {"gnn_connectome": (Cs.chem, Cs.gap), "gnn_none": (np.zeros((K, K)), np.zeros((K, K))),
           "gnn_full": (1.0 - np.eye(K), 1.0 - np.eye(K))}
    for s in range(N_SHUF):
        S = G.shuffle_degree_preserving(C, seed=s).sub(nodes)
        out[f"gnn_shuffled_{s}"] = (S.chem, S.gap)
    D = A2.median_pairwise_distance(recs, nodes)
    k = np.array([len(n) for n in Cs.neighbors("any")])
    knn = np.zeros((K, K))
    for i in range(K):
        d = D[i].copy(); d[i] = np.inf; d[~np.isfinite(d)] = np.inf
        knn[i, np.argsort(d)[: k[i]]] = 1.0
    out["gnn_geometry"] = (knn, knn * knn.T)
    return out


def run_fold(fold):
    t0 = time.time()
    recs = [P.standardize(r) for r in A.load()]
    nodes = P.node_space(recs, 0.5)
    split = SPL.load("A2_atanas_5fold")
    gs = graphs(nodes, recs)
    spec = dict(task="L2_atanas_learned_gnn", nodes=nodes, split_sha256=split["sha256"], h=H, epochs=EPOCHS, lr=LR, gate_l1=L1,
                models=sorted(gs), preprocess=P.PREPROCESS_VERSION)
    OUT.mkdir(parents=True, exist_ok=True)
    sp = OUT / "L2_spec.json"
    if not sp.exists():
        try:
            fd = os.open(sp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            os.write(fd, json.dumps(dict(spec, frozen_utc=datetime.now(timezone.utc).isoformat()), indent=1).encode()); os.close(fd)
        except FileExistsError:
            pass
    old = json.loads(sp.read_text()); assert all(old[k] == json.loads(json.dumps(spec))[k] for k in spec), "spec drift"
    data = {r.animal_id: P.to_nodes(r, nodes) for r in recs}
    train = [data[a] for a, f in split["fold_of"].items() if f != fold]
    test = [a for a, f in split["fold_of"].items() if f == fold]
    pers = Persistence().fit(train, H)
    res, gates = {}, {}
    for name, (cw, gw) in gs.items():
        m = TypedGNN(cw, gw, epochs=EPOCHS, lr=LR, l1=L1, seed=1000 + fold).fit(train, H)
        res[name] = {a: F.score_animal(*data[a], m.predict(*data[a]), pers.predict(*data[a]), H) for a in test}
        if name == "gnn_connectome":
            gc, gg = m.gate_report(); gates = dict(chem=gc.tolist(), gap=gg.tolist())
        print(f"fold {fold} {name} {np.mean([v['r2'] for v in res[name].values()]):.4f} {time.time()-t0:.0f}s", flush=True)
    (OUT / f"fold{fold}.json").write_text(json.dumps(dict(fold=fold, results=res, gates=gates), default=float))


def aggregate():
    folds = [json.loads((OUT / f"fold{f}.json").read_text()) for f in range(5)]
    res = {}
    for d in folds:
        for m, per in d["results"].items():
            res.setdefault(m, {}).update(per)
    animals = sorted(res["gnn_connectome"])
    r2 = lambda m, a: res[m][a]["r2"]
    shuf = [m for m in res if m.startswith("gnn_shuffled")]
    summ = dict(r2={m: F.boot_mean([r2(m, a) for a in animals]) for m in res},
                connectome_minus_none=F.boot_mean([r2("gnn_connectome", a) - r2("gnn_none", a) for a in animals]),
                connectome_minus_full=F.boot_mean([r2("gnn_connectome", a) - r2("gnn_full", a) for a in animals]),
                connectome_minus_shuffled=F.boot_mean([r2("gnn_connectome", a) - np.mean([r2(s, a) for s in shuf]) for a in animals]),
                connectome_minus_geometry=F.boot_mean([r2("gnn_connectome", a) - r2("gnn_geometry", a) for a in animals]),
                geometry_minus_shuffled=F.boot_mean([r2("gnn_geometry", a) - np.mean([r2(s, a) for s in shuf]) for a in animals]))
    a2 = json.loads((REPO / "results/gd/benchmarks/A2_results.json").read_text())["per_animal"][str(H)]
    summ["gnn_connectome_minus_linear_connectome"] = F.boot_mean([r2("gnn_connectome", a) - a2["connectome"][a]["r2"] for a in animals])
    # edge-gate stability across folds (connectome edges only)
    gc = np.array([d["gates"]["chem"] for d in folds]); gg = np.array([d["gates"]["gap"] for d in folds])
    def stab(Gm):
        E = Gm[0] > 0
        V = Gm[:, E]
        c = np.corrcoef(V)
        return dict(n_edges=int(E.sum()), mean_pairwise_fold_corr=float(c[np.triu_indices(len(V), 1)].mean()),
                    gate_mean=float(V.mean()), gate_sd_across_edges=float(V.mean(0).std()))
    summ["gate_stability"] = dict(chem=stab(gc), gap=stab(gg))
    (OUT / "L2_results.json").write_text(json.dumps(dict(summary=summ, per_animal=res), indent=1, default=float))
    print(json.dumps({k: (v if k != "r2" else {m: round(x["mean"], 4) for m, x in v.items()}) for k, v in summ.items()}, indent=1, default=float))


if __name__ == "__main__":
    aggregate() if sys.argv[1] == "aggregate" else run_fold(int(sys.argv[1]))
