"""Capsule manifest + content-addressed root-hash scheme.

The *manifest* is the single, deterministic source of a capsule's **ROOT hash**.
It is a canonicalized, sorted list of ``(logical key -> content sha256)`` rows
over every artifact the capsule content-addresses, plus an explicit row for every
byte that is known to be *missing* (a ``MISSING_UPSTREAM_PROVENANCE`` gap). The
ROOT hash is ``sha256`` of the canonical JSON of that manifest -- and of nothing
else.

Deliberately, the root depends **only** on the manifest, never on the
surrounding capsule metadata (timestamps, notes, replay command). That keeps
``verify`` non-circular: you can recompute the manifest from the recorded
artifact list, hash it, and compare against the stored root without the root ever
having to hash a structure that contains itself.

A missing byte is *not* silently dropped: it appears as a row with
``present=False`` and its *expected* sha256, so it is load-bearing in the root.
Filling that gap with the wrong bytes (or pretending it never existed) changes
the root and is caught by ``verify``.
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable


def canonical_bytes(obj) -> bytes:
    """Deterministic JSON encoding used for every hash in the capsule layer."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest_rows(
    present: Iterable[dict],
    missing: Iterable[dict],
) -> list[dict]:
    """Build the canonical manifest rows.

    ``present``: dicts with keys ``logical_key``, ``sha256``, ``store_path``.
    ``missing``: dicts with keys ``logical_key`` and ``expected_sha256``.

    Returns a deterministically sorted list of manifest rows.
    """
    rows: list[dict] = []
    for a in present:
        rows.append(
            {
                "logical_key": a["logical_key"],
                "store_path": a["store_path"],
                "sha256": a["sha256"],
                "present": True,
            }
        )
    for m in missing:
        rows.append(
            {
                "logical_key": m["logical_key"],
                "store_path": None,
                "sha256": m["expected_sha256"],
                "present": False,
            }
        )
    rows.sort(key=lambda r: (r["logical_key"], r["sha256"] or "", r["present"]))
    return rows


def manifest_root(rows: list[dict]) -> str:
    """The single capsule ROOT hash = sha256 of the canonical manifest."""
    return sha256_bytes(canonical_bytes(rows))
