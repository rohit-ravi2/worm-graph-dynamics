"""Content-addressed evidence capsules: copy-on-freeze snapshots of a run's spec, inputs, code and outputs,
verified by a Merkle root over sha256 hashes."""
from .capsule import ArtifactSpec, EvidenceCapsule, VerifyReport, create_capsule, inspect, verify
from .compare import compare_capsules
from .manifest import manifest_root, sha256_bytes, sha256_file

__all__ = ["ArtifactSpec", "EvidenceCapsule", "VerifyReport", "create_capsule", "verify", "inspect",
           "compare_capsules", "manifest_root", "sha256_file", "sha256_bytes"]
