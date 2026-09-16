from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .context import (
    CHECKPOINT_CONTEXT_VERSION,
    calculation_context,
    parse_context_json,
)
from .root_record import (
    SCFRootRecord,
    SCFRunStatus,
)
from .state_identity import (
    DeduplicationResult,
    IdentityThresholds,
    StateFingerprint,
    deduplicate_roots,
    fingerprint_from_orthonormal_density,
)


@dataclass(frozen=True)
class CheckpointAuditSettings:
    """
    Numerical consistency tolerances for checkpoint reconstruction.

    These are file/numerical integrity tolerances, not electronic-state
    identity thresholds and not physical accuracy criteria.
    """

    energy_tol_hartree: float = 1.0e-8
    geometry_tol_angstrom: float = 1.0e-10
    electron_trace_tol: float = 1.0e-7
    spin_trace_tol: float = 1.0e-7
    overlap_eigenvalue_floor: float = 1.0e-10

    def __post_init__(self) -> None:
        values = (
            self.energy_tol_hartree,
            self.geometry_tol_angstrom,
            self.electron_trace_tol,
            self.spin_trace_tol,
            self.overlap_eigenvalue_floor,
        )

        if any(
            not isfinite(float(x))
            or float(x) <= 0.0
            for x in values
        ):
            raise ValueError(
                "checkpoint audit tolerances must be finite and > 0"
            )


@dataclass(frozen=True)
class CheckpointFingerprintAudit:
    root_id: str
    checkpoint_path: str

    checkpoint_energy_hartree: float
    record_energy_hartree: float
    energy_delta_hartree: float

    checkpoint_r_angstrom: float
    record_r_angstrom: float
    geometry_delta_angstrom: float

    electron_trace: float
    expected_electron_count: int
    electron_trace_error: float

    spin_trace: float
    expected_spin_2s: int
    spin_trace_error: float

    overlap_min_eigenvalue: float
    overlap_max_eigenvalue: float

    reference: str


@dataclass(frozen=True)
class CheckpointFingerprintFailure:
    root_id: str
    message: str


@dataclass(frozen=True)
class CheckpointDeduplicationResult:
    deduplication: DeduplicationResult
    audits: tuple[
        CheckpointFingerprintAudit,
        ...,
    ]
    fingerprint_failures: tuple[
        CheckpointFingerprintFailure,
        ...,
    ]


def _real_array(
    value,
    *,
    name: str,
):
    array = np.asarray(value)

    if np.iscomplexobj(array):
        imag_max = float(
            np.max(
                np.abs(
                    array.imag
                )
            )
        )

        if imag_max > 1.0e-12:
            raise ValueError(
                f"{name} contains non-negligible complex values"
            )

        array = array.real

    array = np.asarray(
        array,
        dtype=float,
    )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            f"{name} contains non-finite values"
        )

    return array


def _spin_resolved_density(
    root: SCFRootRecord,
    scf_data: Mapping,
):
    """
    Reconstruct alpha/beta AO density matrices from checkpoint MOs.
    """
    if "mo_coeff" not in scf_data:
        raise ValueError(
            "checkpoint has no mo_coeff"
        )

    if "mo_occ" not in scf_data:
        raise ValueError(
            "checkpoint has no mo_occ"
        )

    coeff = _real_array(
        scf_data["mo_coeff"],
        name="mo_coeff",
    )

    occ = _real_array(
        scf_data["mo_occ"],
        name="mo_occ",
    )

    if root.reference == "RKS":
        if (
            coeff.ndim != 2
            or occ.ndim != 1
        ):
            raise ValueError(
                "RKS checkpoint has unexpected MO dimensions"
            )

        if coeff.shape[1] != occ.shape[0]:
            raise ValueError(
                "RKS coefficient/occupation dimensions disagree"
            )

        total = (
            coeff
            * occ[np.newaxis, :]
        ) @ coeff.T

        return np.stack(
            (
                0.5 * total,
                0.5 * total,
            )
        )

    if root.reference == "UKS":
        if (
            coeff.ndim != 3
            or coeff.shape[0] != 2
            or occ.ndim != 2
            or occ.shape[0] != 2
        ):
            raise ValueError(
                "UKS checkpoint has unexpected MO dimensions"
            )

        if (
            coeff.shape[2]
            != occ.shape[1]
        ):
            raise ValueError(
                "UKS coefficient/occupation dimensions disagree"
            )

        density = []

        for spin in range(2):
            density.append(
                (
                    coeff[spin]
                    * occ[
                        spin
                    ][np.newaxis, :]
                )
                @ coeff[spin].T
            )

        return np.stack(
            density
        )

    raise ValueError(
        f"unsupported reference {root.reference!r}"
    )


def _lowdin_orthonormal_density(
    mol,
    dm_ao,
    *,
    eigenvalue_floor: float,
):
    """
    Transform AO density to a symmetric Löwdin-orthonormal representation.

        P_orth = S^(1/2) P_AO S^(1/2)

    This preserves Tr(P S) as Tr(P_orth).
    """
    overlap = _real_array(
        mol.intor_symmetric(
            "int1e_ovlp"
        ),
        name="AO overlap",
    )

    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            overlap
        )
    )

    minimum = float(
        eigenvalues.min()
    )

    maximum = float(
        eigenvalues.max()
    )

    if minimum <= eigenvalue_floor:
        raise ValueError(
            "AO overlap is singular or below the configured "
            f"eigenvalue floor: {minimum:.6e}"
        )

    sqrt_overlap = (
        eigenvectors
        @ np.diag(
            np.sqrt(
                eigenvalues
            )
        )
        @ eigenvectors.T
    )

    transformed = np.stack(
        (
            sqrt_overlap
            @ dm_ao[0]
            @ sqrt_overlap,
            sqrt_overlap
            @ dm_ao[1]
            @ sqrt_overlap,
        )
    )

    transformed = 0.5 * (
        transformed
        + transformed.transpose(
            0,
            2,
            1,
        )
    )

    return (
        transformed,
        minimum,
        maximum,
    )


def fingerprint_from_checkpoint(
    root: SCFRootRecord,
    *,
    settings: CheckpointAuditSettings | None = None,
) -> tuple[
    StateFingerprint,
    CheckpointFingerprintAudit,
]:
    """
    Reconstruct and audit a StateFingerprint from one final PySCF checkpoint.

    The checkpoint must agree with the SCFRootRecord in:

    - charge;
    - requested 2S sector;
    - final energy;
    - electron count;
    - alpha-minus-beta electron count.

    These checks protect state-identity logic from stale or inconsistent
    checkpoint files.
    """
    from pyscf.scf import chkfile

    if settings is None:
        settings = (
            CheckpointAuditSettings()
        )

    if (
        root.status
        != SCFRunStatus.CANONICALIZED
    ):
        raise ValueError(
            "checkpoint fingerprint requires a canonicalized root"
        )

    if root.energy_hartree is None:
        raise ValueError(
            "canonicalized root has no energy"
        )

    if not root.checkpoint_path:
        raise ValueError(
            "root has no checkpoint_path"
        )

    path = Path(
        root.checkpoint_path
    )

    if not path.is_file():
        raise ValueError(
            f"checkpoint does not exist: {path}"
        )

    mol, scf_data = (
        chkfile.load_scf(
            str(path)
        )
    )

    if not scf_data:
        raise ValueError(
            "checkpoint contains no SCF data"
        )

    if int(mol.charge) != int(
        root.charge
    ):
        raise ValueError(
            "checkpoint charge does not match root record"
        )

    if int(mol.spin) != int(
        root.spin_2s
    ):
        raise ValueError(
            "checkpoint spin does not match root record"
        )

    coordinates = np.asarray(
        mol.atom_coords(
            unit="Angstrom"
        ),
        dtype=float,
    )

    if coordinates.shape != (2, 3):
        raise ValueError(
            "checkpoint molecule is not a diatomic geometry"
        )

    checkpoint_r = float(
        np.linalg.norm(
            coordinates[1]
            - coordinates[0]
        )
    )

    geometry_delta = (
        checkpoint_r
        - float(
            root.r_angstrom
        )
    )

    if (
        abs(geometry_delta)
        > settings.geometry_tol_angstrom
    ):
        raise ValueError(
            "checkpoint geometry does not match root record: "
            f"delta={geometry_delta:+.6e} Angstrom"
        )

    from pyscf import lib

    try:
        raw_context = (
            lib.chkfile.load(
                str(path),
                "openea/context_json",
            )
        )
    except Exception as exc:
        raise ValueError(
            "checkpoint has no valid OpenEA metadata"
        ) from exc

    metadata = parse_context_json(
        raw_context
    )

    schema_version = metadata.get(
        "schema_version"
    )

    if (
        schema_version
        != CHECKPOINT_CONTEXT_VERSION
    ):
        raise ValueError(
            "checkpoint OpenEA metadata has unsupported schema version"
        )

    checkpoint_root_id = metadata.get(
        "root_id"
    )

    if checkpoint_root_id != root.root_id:
        raise ValueError(
            "checkpoint OpenEA context mismatch for root_id"
        )

    expected_context = (
        calculation_context(
            molecule=root.molecule,
            atom_a=root.atom_a,
            atom_b=root.atom_b,
            charge=root.charge,
            spin_2s=root.spin_2s,
            r_angstrom=root.r_angstrom,
            functional=root.functional,
            basis=root.basis,
            reference=root.reference,
            origin_guess=(
                root.origin_guess
            ),
            ecp_assignments=(
                root.ecp_assignments
            ),
        )
    )

    stored_context = metadata.get(
        "calculation"
    )

    if not isinstance(
        stored_context,
        dict,
    ):
        raise ValueError(
            "checkpoint OpenEA metadata has no calculation context"
        )

    for field, expected in (
        expected_context.items()
    ):
        actual = stored_context.get(
            field
        )

        if actual != expected:
            raise ValueError(
                "checkpoint OpenEA context mismatch for "
                f"{field}: {actual!r} != {expected!r}"
            )

    if "e_tot" not in scf_data:
        raise ValueError(
            "checkpoint has no e_tot"
        )

    checkpoint_energy = float(
        scf_data["e_tot"]
    )

    if not isfinite(
        checkpoint_energy
    ):
        raise ValueError(
            "checkpoint energy is non-finite"
        )

    energy_delta = (
        checkpoint_energy
        - float(
            root.energy_hartree
        )
    )

    if (
        abs(energy_delta)
        > settings.energy_tol_hartree
    ):
        raise ValueError(
            "checkpoint energy does not match root record: "
            f"delta={energy_delta:+.6e} Eh"
        )

    dm_ao = _spin_resolved_density(
        root,
        scf_data,
    )

    (
        dm_orth,
        overlap_min,
        overlap_max,
    ) = _lowdin_orthonormal_density(
        mol,
        dm_ao,
        eigenvalue_floor=(
            settings.overlap_eigenvalue_floor
        ),
    )

    fingerprint = (
        fingerprint_from_orthonormal_density(
            dm_orth
        )
    )

    electron_trace = float(
        fingerprint.total_trace
    )

    spin_trace = float(
        fingerprint.spin_trace
    )

    expected_electrons = int(
        mol.nelectron
    )

    expected_spin = int(
        root.spin_2s
    )

    electron_error = (
        electron_trace
        - expected_electrons
    )

    spin_error = (
        spin_trace
        - expected_spin
    )

    if (
        abs(electron_error)
        > settings.electron_trace_tol
    ):
        raise ValueError(
            "orthonormal density has inconsistent electron count: "
            f"error={electron_error:+.6e}"
        )

    if (
        abs(spin_error)
        > settings.spin_trace_tol
    ):
        raise ValueError(
            "orthonormal density has inconsistent spin trace: "
            f"error={spin_error:+.6e}"
        )

    audit = CheckpointFingerprintAudit(
        root_id=root.root_id,
        checkpoint_path=str(path),
        checkpoint_energy_hartree=(
            checkpoint_energy
        ),
        record_energy_hartree=float(
            root.energy_hartree
        ),
        energy_delta_hartree=(
            energy_delta
        ),
        checkpoint_r_angstrom=(
            checkpoint_r
        ),
        record_r_angstrom=float(
            root.r_angstrom
        ),
        geometry_delta_angstrom=(
            geometry_delta
        ),
        electron_trace=(
            electron_trace
        ),
        expected_electron_count=(
            expected_electrons
        ),
        electron_trace_error=(
            electron_error
        ),
        spin_trace=(
            spin_trace
        ),
        expected_spin_2s=(
            expected_spin
        ),
        spin_trace_error=(
            spin_error
        ),
        overlap_min_eigenvalue=(
            overlap_min
        ),
        overlap_max_eigenvalue=(
            overlap_max
        ),
        reference=root.reference,
    )

    return (
        fingerprint,
        audit,
    )


def fingerprints_from_checkpoints(
    roots: Sequence[
        SCFRootRecord
    ],
    *,
    settings: CheckpointAuditSettings | None = None,
) -> tuple[
    dict[str, StateFingerprint],
    tuple[
        CheckpointFingerprintAudit,
        ...,
    ],
    tuple[
        CheckpointFingerprintFailure,
        ...,
    ],
]:
    """
    Load all usable canonicalized-root fingerprints.

    Failure to reconstruct one root is preserved explicitly. It does not
    prevent successful roots from being fingerprinted, and it does not cause
    the failed root to disappear from later deduplication.
    """
    fingerprints: dict[
        str,
        StateFingerprint,
    ] = {}

    audits = []
    failures = []

    for root in roots:
        if (
            root.status
            != SCFRunStatus.CANONICALIZED
        ):
            failures.append(
                CheckpointFingerprintFailure(
                    root_id=root.root_id,
                    message=(
                        "root is not CANONICALIZED"
                    ),
                )
            )

            continue

        try:
            (
                fingerprint,
                audit,
            ) = fingerprint_from_checkpoint(
                root,
                settings=settings,
            )

        except Exception as exc:
            failures.append(
                CheckpointFingerprintFailure(
                    root_id=root.root_id,
                    message=(
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )

            continue

        fingerprints[
            root.root_id
        ] = fingerprint

        audits.append(
            audit
        )

    return (
        fingerprints,
        tuple(
            sorted(
                audits,
                key=lambda item:
                    item.root_id,
            )
        ),
        tuple(
            sorted(
                failures,
                key=lambda item:
                    item.root_id,
            )
        ),
    )


def deduplicate_checkpoint_roots(
    roots: Sequence[
        SCFRootRecord
    ],
    *,
    thresholds: IdentityThresholds,
    audit_settings: CheckpointAuditSettings | None = None,
) -> CheckpointDeduplicationResult:
    """
    End-to-end same-geometry root deduplication from final checkpoints.
    """
    (
        fingerprints,
        audits,
        failures,
    ) = fingerprints_from_checkpoints(
        roots,
        settings=audit_settings,
    )

    deduplication = (
        deduplicate_roots(
            roots,
            fingerprints,
            thresholds=thresholds,
        )
    )

    return CheckpointDeduplicationResult(
        deduplication=deduplication,
        audits=audits,
        fingerprint_failures=(
            failures
        ),
    )
