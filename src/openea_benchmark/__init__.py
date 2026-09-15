"""
OpenEA-Benchmark scientific workflow infrastructure.

No production electron-affinity method is frozen yet.
"""

from .checkpoint_fingerprint import (
    CheckpointAuditSettings,
    CheckpointDeduplicationResult,
    CheckpointFingerprintAudit,
    CheckpointFingerprintFailure,
    deduplicate_checkpoint_roots,
    fingerprint_from_checkpoint,
    fingerprints_from_checkpoints,
)
from .pyscf_backend import (
    DEFAULT_GUESSES,
    DFTMethodSpec,
    DiatomicSpec,
    SCFSettings,
    build_molecule,
    run_guess_panel,
    run_scf_attempt,
)
from .root_record import (
    SCFRootRecord,
    SCFRunStatus,
)
from .state_identity import (
    DeduplicationResult,
    IdentityThresholds,
    RootCluster,
    StateComparison,
    StateFingerprint,
    StateRelation,
    classify_metrics,
    compare_states,
    deduplicate_roots,
    fingerprint_from_orthonormal_density,
)

__all__ = [
    "CheckpointAuditSettings",
    "CheckpointDeduplicationResult",
    "CheckpointFingerprintAudit",
    "CheckpointFingerprintFailure",
    "DEFAULT_GUESSES",
    "DFTMethodSpec",
    "DeduplicationResult",
    "DiatomicSpec",
    "IdentityThresholds",
    "RootCluster",
    "SCFRootRecord",
    "SCFRunStatus",
    "SCFSettings",
    "StateComparison",
    "StateFingerprint",
    "StateRelation",
    "build_molecule",
    "classify_metrics",
    "compare_states",
    "deduplicate_checkpoint_roots",
    "deduplicate_roots",
    "fingerprint_from_checkpoint",
    "fingerprints_from_checkpoints",
    "fingerprint_from_orthonormal_density",
    "run_guess_panel",
    "run_scf_attempt",
]
