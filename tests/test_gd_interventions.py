"""Unified intervention API: validation before execution, graph/state application, hashing."""
import numpy as np
import pytest

from wormlab.gd import graph as G, interventions as IV

NODES = ["AVAL", "AVAR", "AVEL", "RIML", "AIBL"]


@pytest.fixture(scope="module")
def C():
    return G.load_connectome().sub(NODES)


def test_ablate_removes_all_edges_and_activity(C):
    iv = IV.NodeAblate(("AVAL",)).validate(NODES)
    chem, gap = iv.apply_graph(C.chem, C.gap, NODES)
    assert chem[0].sum() == chem[:, 0].sum() == gap[0].sum() == 0
    assert np.all(iv.apply_state(np.ones((3, 5)), NODES)[:, 0] == 0)


def test_invalid_requests_rejected_before_execution(C):
    with pytest.raises(IV.InvalidIntervention):
        IV.NodeClamp(("XYZ",)).validate(NODES)
    with pytest.raises(IV.InvalidIntervention):
        IV.EdgeDelete((("AIBL", "AVAL", "gap"),)).validate(NODES, C.chem, C.gap)
    with pytest.raises(IV.InvalidIntervention):
        IV.EdgeScale("peptide", 0.0).validate(NODES)


def test_edge_scale_and_hash(C):
    chem, gap = IV.EdgeScale("gap", 0.0).validate(NODES).apply_graph(C.chem, C.gap, NODES)
    assert gap.sum() == 0 and np.array_equal(chem, C.chem)
    assert IV.NodeClamp(("AVAL",)).content_hash() != IV.NodeClamp(("AVAR",)).content_hash()


def test_shuffle_preserves_degrees():
    full = G.load_connectome()
    S = G.shuffle_degree_preserving(full, 3)
    assert np.array_equal((S.chem > 0).sum(0), (full.chem > 0).sum(0))
    assert np.array_equal((S.chem > 0).sum(1), (full.chem > 0).sum(1))
    assert np.array_equal((S.gap > 0).sum(1), (full.gap > 0).sum(1))
