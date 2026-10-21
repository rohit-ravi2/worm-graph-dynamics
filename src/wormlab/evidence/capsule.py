"""EvidenceCapsule: copy-on-freeze, content-addressed, immutable evidence.

Creating a capsule copies the bytes of every referenced artifact (spec, split, data manifests, code, environment,
results) into a content-addressed store (``store/<sha256>``) inside the capsule directory, so the capsule never depends
on a mutable working-tree path. Editing the working tree afterwards cannot change what the capsule contains.

Guarantees
----------
* Content-addressed store: every blob lives at ``store/<sha256>``; identical bytes dedupe to one blob.
* Single root hash (``manifest.manifest_root``) over the artifact list. One changed byte means the blob's name no
  longer matches its content and/or the recomputed root differs, and verification fails.
* Write-once: ``create_capsule`` refuses to write into an existing non-empty directory. A child capsule may reference
  a parent by hash but never writes into it.
* Explicit missing bytes: unrecoverable upstream bytes are recorded as ``MISSING_UPSTREAM_PROVENANCE`` rows
  (path, expected hash, note), reported by ``verify`` rather than hidden.
* The replay block references artifacts by sha256 / store path only, never a working-tree path.

Operations: ``create_capsule``, ``verify``, ``inspect`` and ``compare_capsules`` (in ``compare.py``).
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .manifest import (
    build_manifest_rows,
    canonical_bytes,
    manifest_root,
    sha256_file,
)

SCHEMA_VERSION = "1.0"
STORE_DIR = "store"
CAPSULE_JSON = "CAPSULE.json"
ROOT_FILE = "ROOT.sha256"

# Controlled artifact-role vocabulary.
ROLE_SPEC = "spec"
ROLE_PLAN = "execution_plan"
ROLE_INPUT_MANIFEST = "input_manifest"
ROLE_CODE = "code_snapshot"
ROLE_ENV = "environment_lock"
ROLE_RAW_INPUT_ROOT = "raw_input_root"
ROLE_PROCESSED_INPUT_ROOT = "processed_input_root"
ROLE_OUTPUT = "output"
ROLE_SUMMARY = "summary"
ROLE_DEVIATION = "deviation"
ROLE_SOURCE_MANIFEST = "source_manifest"

ROLES = frozenset(
    {
        ROLE_SPEC,
        ROLE_PLAN,
        ROLE_INPUT_MANIFEST,
        ROLE_CODE,
        ROLE_ENV,
        ROLE_RAW_INPUT_ROOT,
        ROLE_PROCESSED_INPUT_ROOT,
        ROLE_OUTPUT,
        ROLE_SUMMARY,
        ROLE_DEVIATION,
        ROLE_SOURCE_MANIFEST,
    }
)

# Working-tree prefixes a replay block must never reference (the whole point).
_MUTABLE_PREFIXES = ("src/", "results/", "tests/", "/home/", "/mnt/")


# --------------------------------------------------------------------------
# input specs
# --------------------------------------------------------------------------
@dataclass
class ArtifactSpec:
    """One artifact to copy-on-freeze into the capsule's content store.

    ``original_path`` is recorded as provenance (where the bytes came from in the
    working tree / external root). ``src`` is the file actually read; if omitted,
    ``original_path`` is read directly.
    """

    role: str
    original_path: str
    src: str | os.PathLike | None = None

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"unknown artifact role {self.role!r}")


@dataclass
class MissingSpec:
    """An explicitly-recorded MISSING_UPSTREAM_PROVENANCE gap.

    The bytes with ``expected_sha256`` are known to have existed (pinned by
    ``manifest_source``) but were overwritten before archival and are
    unrecoverable. Recorded honestly -- never fabricated.
    """

    role: str
    original_path: str
    expected_sha256: str
    manifest_source: str
    note: str


# --------------------------------------------------------------------------
# create
# --------------------------------------------------------------------------
def _logical_key(role: str, original_path: str) -> str:
    return f"{role}::{original_path}"


def _copy_into_store(store: Path, src: Path) -> tuple[str, int]:
    """Copy ``src`` into the content-addressed store; return (sha256, size)."""
    sha = sha256_file(src)
    dest = store / sha
    if not dest.exists():
        tmp = store / (sha + ".tmp")
        shutil.copyfile(src, tmp)
        # verify the copy landed byte-identical before publishing it
        if sha256_file(tmp) != sha:
            tmp.unlink(missing_ok=True)
            raise IOError(f"content-store copy mismatch for {src}")
        os.replace(tmp, dest)
        os.chmod(dest, 0o444)  # read-only: reinforce immutability
    size = dest.stat().st_size
    return sha, size


def create_capsule(
    capsule_dir: str | os.PathLike,
    *,
    capsule_id: str,
    replay_command: str,
    spec: ArtifactSpec | None = None,
    execution_plan: ArtifactSpec | None = None,
    environment_lock: ArtifactSpec | None = None,
    summary: ArtifactSpec | None = None,
    raw_input_root: ArtifactSpec | None = None,
    processed_input_root: ArtifactSpec | None = None,
    input_manifests: list[ArtifactSpec] = (),
    code_snapshot: list[ArtifactSpec] = (),
    outputs: list[ArtifactSpec] = (),
    deviations: list[ArtifactSpec] = (),
    missing_upstream: list[MissingSpec] = (),
    parent: "str | os.PathLike | EvidenceCapsule | None" = None,
    description: str = "",
    external_inputs: list[dict] = (),
) -> "EvidenceCapsule":
    """Freeze a set of working-tree artifacts into a new immutable capsule.

    COPIES the bytes of every artifact into ``capsule_dir/store/<sha256>``; the
    capsule never again depends on the source paths. Write-once: fails if
    ``capsule_dir`` already exists and is non-empty. Never writes into a parent.
    """
    cdir = Path(capsule_dir)
    if cdir.exists() and any(cdir.iterdir()):
        raise FileExistsError(
            f"write-once violation: capsule dir {cdir} already exists and is "
            "non-empty. Capsules are immutable; create a new capsule (optionally "
            "referencing this one as parent) instead."
        )
    store = cdir / STORE_DIR
    store.mkdir(parents=True, exist_ok=True)

    # --- parent reference (READ ONLY: never touch the parent directory) ---
    parent_ref = None
    if parent is not None:
        pdir = Path(parent.dir) if isinstance(parent, EvidenceCapsule) else Path(parent)
        proot = (pdir / ROOT_FILE).read_text().strip()
        pmeta = EvidenceCapsule(pdir).meta
        parent_ref = {
            "capsule_id": pmeta["capsule_id"],
            "root_hash": proot,
            "path": str(pdir),
        }

    # --- copy every artifact into the content-addressed store ---
    artifacts: list[dict] = []

    def _add(a: ArtifactSpec):
        src = Path(a.src) if a.src is not None else Path(a.original_path)
        sha, size = _copy_into_store(store, src)
        artifacts.append(
            {
                "role": a.role,
                "original_path": str(a.original_path),
                "sha256": sha,
                "size": size,
                "store_path": f"{STORE_DIR}/{sha}",
                "logical_key": _logical_key(a.role, str(a.original_path)),
            }
        )

    singles = [spec, execution_plan, environment_lock, summary,
               raw_input_root, processed_input_root]
    for a in singles:
        if a is not None:
            _add(a)
    for group in (input_manifests, code_snapshot, outputs, deviations):
        for a in group:
            _add(a)

    # --- missing-upstream-provenance (explicit, never fabricated) ---
    missing_rows = []
    for m in missing_upstream:
        missing_rows.append(
            {
                "role": m.role,
                "original_path": m.original_path,
                "expected_sha256": m.expected_sha256,
                "manifest_source": m.manifest_source,
                "note": m.note,
                "status": "MISSING_UPSTREAM_PROVENANCE",
                "logical_key": "MISSING::" + _logical_key(m.role, m.original_path),
            }
        )

    # --- manifest + single ROOT hash ---
    rows = build_manifest_rows(
        present=[
            {"logical_key": a["logical_key"], "sha256": a["sha256"],
             "store_path": a["store_path"]}
            for a in artifacts
        ],
        missing=[
            {"logical_key": m["logical_key"], "expected_sha256": m["expected_sha256"]}
            for m in missing_rows
        ],
    )
    root = manifest_root(rows)

    # --- replay: references ONLY content-addressed store entries ---
    replay = {
        "command": replay_command,
        "environment_lock_sha256": next(
            (a["sha256"] for a in artifacts if a["role"] == ROLE_ENV), None
        ),
        "inputs": [
            {"role": a["role"], "sha256": a["sha256"], "store_path": a["store_path"]}
            for a in artifacts
            if a["role"] in (ROLE_SPEC, ROLE_PLAN, ROLE_INPUT_MANIFEST,
                             ROLE_CODE, ROLE_RAW_INPUT_ROOT,
                             ROLE_PROCESSED_INPUT_ROOT)
        ],
    }
    _assert_replay_content_addressed(replay)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "capsule_id": capsule_id,
        "description": description,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent": parent_ref,
        "artifacts": artifacts,
        "missing_upstream_provenance": missing_rows,
        "external_inputs": list(external_inputs),
        "manifest": rows,
        "root_hash": root,
        "replay": replay,
    }

    (cdir / CAPSULE_JSON).write_bytes(_pretty(meta))
    (cdir / ROOT_FILE).write_text(root + "\n")
    return EvidenceCapsule(cdir)


def _pretty(meta: dict) -> bytes:
    import json

    return (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _assert_replay_content_addressed(replay: dict) -> None:
    """Fail if any replay input names a bare mutable working-tree path."""
    for inp in replay.get("inputs", []):
        for k in ("store_path", "sha256"):
            if k not in inp:
                raise ValueError(f"replay input missing content address: {inp}")
        sp = inp["store_path"]
        if not sp.startswith(STORE_DIR + "/"):
            raise ValueError(f"replay input not content-addressed: {sp}")
        if any(p in sp for p in _MUTABLE_PREFIXES):
            raise ValueError(f"replay references mutable path: {sp}")


# --------------------------------------------------------------------------
# capsule handle + verify / inspect
# --------------------------------------------------------------------------
@dataclass
class VerifyReport:
    ok: bool
    n_checked: int
    root_hash: str
    root_ok: bool
    failures: list[str] = field(default_factory=list)
    missing: list[dict] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


class EvidenceCapsule:
    """A handle to an on-disk immutable capsule directory."""

    def __init__(self, capsule_dir: str | os.PathLike):
        self.dir = Path(capsule_dir)
        cj = self.dir / CAPSULE_JSON
        if not cj.exists():
            raise FileNotFoundError(f"not a capsule (no {CAPSULE_JSON}): {self.dir}")
        import json

        self.meta = json.loads(cj.read_text())

    # -- convenience accessors --
    @property
    def capsule_id(self) -> str:
        return self.meta["capsule_id"]

    @property
    def root_hash(self) -> str:
        return self.meta["root_hash"]

    @property
    def artifacts(self) -> list[dict]:
        return self.meta["artifacts"]

    def artifacts_of(self, role: str) -> list[dict]:
        return [a for a in self.artifacts if a["role"] == role]

    def blob_path(self, artifact: dict) -> Path:
        return self.dir / artifact["store_path"]

    # -- operations --
    def verify(self) -> VerifyReport:
        return verify(self.dir)

    def inspect(self) -> dict:
        return inspect(self.dir)


def _load_meta(cdir: Path) -> dict:
    import json

    return json.loads((cdir / CAPSULE_JSON).read_text())


def verify(capsule_dir: str | os.PathLike) -> VerifyReport:
    """Recompute every content hash + the root. One changed byte -> ok=False.

    A recorded MISSING_UPSTREAM_PROVENANCE row is reported (not hidden) but does
    not, by itself, fail verification -- a documented gap is neither a silent pass
    nor a hard fail.
    """
    cdir = Path(capsule_dir)
    meta = _load_meta(cdir)
    failures: list[str] = []
    n = 0

    for a in meta["artifacts"]:
        blob = cdir / a["store_path"]
        n += 1
        if not blob.exists():
            failures.append(f"missing blob {a['store_path']} for {a['original_path']}")
            continue
        got = sha256_file(blob)
        # blob is content-addressed: filename must equal its own content hash
        if blob.name != got:
            failures.append(
                f"content-store corruption: {a['store_path']} hashes to {got}"
            )
        if got != a["sha256"]:
            failures.append(
                f"hash drift for {a['original_path']}: recorded {a['sha256']} "
                f"!= on-disk {got}"
            )

    # recompute the manifest + root from the recorded artifact list
    rows = build_manifest_rows(
        present=[
            {"logical_key": a["logical_key"], "sha256": a["sha256"],
             "store_path": a["store_path"]}
            for a in meta["artifacts"]
        ],
        missing=[
            {"logical_key": m["logical_key"], "expected_sha256": m["expected_sha256"]}
            for m in meta.get("missing_upstream_provenance", [])
        ],
    )
    recomputed_root = manifest_root(rows)
    root_ok = (
        recomputed_root == meta["root_hash"]
        and recomputed_root == (cdir / ROOT_FILE).read_text().strip()
        and rows == meta["manifest"]
    )
    if not root_ok:
        failures.append(
            f"ROOT mismatch: recomputed {recomputed_root} != recorded "
            f"{meta['root_hash']} / ROOT.sha256"
        )

    missing = meta.get("missing_upstream_provenance", [])
    return VerifyReport(
        ok=(not failures),
        n_checked=n,
        root_hash=meta["root_hash"],
        root_ok=root_ok,
        failures=failures,
        missing=missing,
    )


def inspect(capsule_dir: str | os.PathLike) -> dict:
    """List contents + provenance of a capsule (no recomputation)."""
    cdir = Path(capsule_dir)
    meta = _load_meta(cdir)
    contents = [
        {
            "role": a["role"],
            "original_path": a["original_path"],
            "sha256": a["sha256"],
            "size": a["size"],
            "store_path": a["store_path"],
        }
        for a in meta["artifacts"]
    ]
    return {
        "capsule_id": meta["capsule_id"],
        "root_hash": meta["root_hash"],
        "created_utc": meta["created_utc"],
        "description": meta.get("description", ""),
        "parent": meta.get("parent"),
        "n_artifacts": len(contents),
        "roles": sorted({a["role"] for a in contents}),
        "contents": contents,
        "missing_upstream_provenance": meta.get("missing_upstream_provenance", []),
        "external_inputs": meta.get("external_inputs", []),
        "replay": meta["replay"],
    }
