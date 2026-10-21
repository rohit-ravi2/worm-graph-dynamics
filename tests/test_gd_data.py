"""WormLab Graph Dynamics data layer: identity normalisation, content-verified fetch, Recording schema, adapters."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from wormlab.gd import identity as I
from wormlab.gd.fetch import IntegrityError, ensure_file
from wormlab.gd.schema import Recording


@pytest.mark.parametrize("raw,status,name", [
    ("AVAL", "canonical", "AVAL"), ("ava", "class_only", None), ("AVAL?", "uncertain", None), ("AIY?", "uncertain", None),
    ("AVAL/AVAR", "ambiguous", None), ("12", "unknown", None), ("", "unknown", None), ("ALA", "canonical", "ALA"),
])
def test_normalize(raw, status, name):
    lab = I.normalize(raw)
    assert lab.status == status and lab.name == name


def test_homolog_and_class():
    assert I.homolog("AVAL") == "AVAR" and I.homolog("ALA") is None
    assert I.neuron_class("RMDDL") == "RMDD" and len(I.canonical_names()) == 300


def test_fetch_verifies_content(tmp_path, monkeypatch):
    cfg = tmp_path / "c.toml"
    mirror = tmp_path / "mirror"; mirror.mkdir()
    (mirror / "f.bin").write_bytes(b"hello")
    cfg.write_text(f'cache_dir = "{tmp_path / "cache"}"\nmirrors = ["{mirror}"]\n')
    monkeypatch.setenv("WORMLAB_CONFIG", str(cfg))
    good = hashlib.sha256(b"hello").hexdigest()
    p = ensure_file("ds", "f.bin", "http://invalid.invalid/f.bin", good, download=False)
    assert p.read_bytes() == b"hello"
    with pytest.raises(IntegrityError):
        ensure_file("ds", "f.bin", "http://invalid.invalid/f.bin", "0" * 64, download=False)


def _rec(**kw):
    T, N = 10, 3
    base = dict(dataset="x", animal_id="a", condition="c", t=np.arange(T, dtype=float), neural=np.zeros((T, N)),
                mask=np.ones((T, N), bool), labels=[I.normalize(n) for n in ("AVAL", "AVAR", "")], signal_kind="test")
    base.update(kw)
    return Recording(**base)


def test_schema_validates_and_hashes():
    r = _rec().validate()
    assert r.canonical_index() == {"AVAL": 0, "AVAR": 1}
    assert len(r.content_hash()) == 64
    with pytest.raises(AssertionError):
        _rec(t=np.zeros(10)).validate()
    with pytest.raises(AssertionError):
        _rec(labels=[I.normalize("AVAL")]).validate()


def test_duplicate_canonical_labels_are_dropped():
    r = _rec(labels=[I.normalize(n) for n in ("AVAL", "AVAL", "AVAR")]).validate()
    assert r.canonical_index() == {"AVAR": 2}


def test_kato_loads_if_available():
    from wormlab.gd.sources import kato2015 as K
    try:
        recs = K.load("WT_Stim", download=False)
    except FileNotFoundError:
        pytest.skip("Kato 2015 not in cache or mirrors")
    assert len(recs) == 7
    assert any(e.kind == "stimulus" for e in recs[0].events)


def test_atanas_loads_if_cached():
    from wormlab.gd.sources import atanas2023 as A
    try:
        idx = A.asset_index()
    except Exception:
        pytest.skip("no Atanas asset index / network")
    if not (A._dir() / f"{idx[0]['subject']}.h5").exists():
        pytest.skip("Atanas cache not built")
    r = A.load([idx[0]["subject"]])[0]
    assert r.neural.shape[0] >= 1600 and "velocity" in r.behavior
