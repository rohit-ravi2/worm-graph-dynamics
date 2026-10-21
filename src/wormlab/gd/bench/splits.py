"""Frozen, content-addressed animal splits. Written once, read-only, refused if they would change."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
SPLIT_DIR = REPO / "results" / "gd" / "splits"


def canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def freeze(name: str, animals: list[str], *, n_folds: int = 5, test_frac: float | None = None, seed: int) -> dict:
    """K-fold assignment of animals (every animal is a test animal exactly once). Idempotent: an existing split with
    the same inputs is returned; a different one raises."""
    animals = sorted(animals)
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(animals))
    folds = {a: int(i % n_folds) for i, a in enumerate(order)}
    body = dict(name=name, seed=seed, n_folds=n_folds, animals=animals, fold_of=folds)
    body["sha256"] = canonical_sha(body)
    p = SPLIT_DIR / f"{name}.json"
    if p.exists():
        old = json.loads(p.read_text())
        if old["sha256"] != body["sha256"]:
            raise RuntimeError(f"split {name} already frozen with different content")
        return old
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(body, frozen_utc=datetime.now(timezone.utc).isoformat()), indent=1))
    os.chmod(p, 0o444)
    return body


def load(name: str) -> dict:
    d = json.loads((SPLIT_DIR / f"{name}.json").read_text())
    assert d["sha256"] == canonical_sha({k: v for k, v in d.items() if k not in ("sha256", "frozen_utc")}), "split tampered"
    return d
