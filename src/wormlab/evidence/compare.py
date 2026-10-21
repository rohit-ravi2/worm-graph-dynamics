"""compare -- structured diff of two evidence capsules.

Answers: for two capsules (typically a capsule and the one that supersedes it),
what actually differs -- spec, execution plan, outputs, and code snapshot? Diff
is by content hash over the ``(role, original_path)`` key, so a byte change to any
artifact shows up as ``changed`` while renames/additions/removals show up as
``only_in_a`` / ``only_in_b``.
"""
from __future__ import annotations

import os
from pathlib import Path

from .capsule import (
    ROLE_CODE,
    ROLE_OUTPUT,
    ROLE_PLAN,
    ROLE_SPEC,
    EvidenceCapsule,
)


def _index(cap: EvidenceCapsule, role: str) -> dict[str, dict]:
    """Map original_path -> artifact for one role."""
    return {a["original_path"]: a for a in cap.artifacts_of(role)}


def _diff_role(a: EvidenceCapsule, b: EvidenceCapsule, role: str) -> dict:
    ia, ib = _index(a, role), _index(b, role)
    keys_a, keys_b = set(ia), set(ib)
    changed = []
    identical = []
    for k in sorted(keys_a & keys_b):
        if ia[k]["sha256"] != ib[k]["sha256"]:
            changed.append(
                {"original_path": k, "a_sha256": ia[k]["sha256"],
                 "b_sha256": ib[k]["sha256"]}
            )
        else:
            identical.append(k)
    return {
        "only_in_a": sorted(keys_a - keys_b),
        "only_in_b": sorted(keys_b - keys_a),
        "changed": changed,
        "identical": identical,
        "differs": bool(changed or (keys_a ^ keys_b)),
    }


def compare_capsules(
    a: str | os.PathLike | EvidenceCapsule,
    b: str | os.PathLike | EvidenceCapsule,
) -> dict:
    """Return a structured diff of two capsules: spec, plan, outputs, code."""
    ca = a if isinstance(a, EvidenceCapsule) else EvidenceCapsule(Path(a))
    cb = b if isinstance(b, EvidenceCapsule) else EvidenceCapsule(Path(b))

    sections = {
        "spec": _diff_role(ca, cb, ROLE_SPEC),
        "execution_plan": _diff_role(ca, cb, ROLE_PLAN),
        "outputs": _diff_role(ca, cb, ROLE_OUTPUT),
        "code_snapshot": _diff_role(ca, cb, ROLE_CODE),
    }
    return {
        "a": {"capsule_id": ca.capsule_id, "root_hash": ca.root_hash},
        "b": {"capsule_id": cb.capsule_id, "root_hash": cb.root_hash},
        "root_hash_equal": ca.root_hash == cb.root_hash,
        "b_parent_is_a": (cb.meta.get("parent") or {}).get("root_hash") == ca.root_hash,
        "sections": sections,
        "any_difference": any(s["differs"] for s in sections.values())
        or ca.root_hash != cb.root_hash,
    }
