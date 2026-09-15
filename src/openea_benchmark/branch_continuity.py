from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

import numpy as np

from .checkpoint_fingerprint import (
    CheckpointAuditSettings,
    fingerprint_from_checkpoint,
)
from .root_record import (
    SCFRootRecord,
    SCFRunStatus,
)


HARTREE_TO_EV = 27.211386245988


class BranchRelation(str, Enum):
    """
    Pairwise continuity relation between roots at different R.

    CONTINUOUS:
        Available occupied-subspace and spin evidence supports following
        the same electronic branch.

    DISCONTINUOUS:
        Available evidence clearly indicates a branch change.

    AMBIGUOUS:
        Evidence is intermediate or the geometry spacing is too large for
        the configured continuity policy.

    AMBIGUOUS is a scientific result and must not silently be converted
    into either of the other relations.
    """

    CONTINUOUS = "CONTINUOUS"
    DISCONTINUOUS = "DISCONTINUOUS"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class BranchThresholds:
    """
    Explicit branch-continuity thresholds.

    No production defaults are provided.

    The workflow must validate these values before prediction calculations.

    continuous_occ_min
        Minimum singular value of every occupied spin subspace required for
        a positive continuity assignment.

    discontinuous_occ_min
        If any occupied-subspace minimum singular value falls at or below
        this value, the pair may be classified as discontinuous.

    continuous_delta_s2 / discontinuous_delta_s2
        Corresponding bounds for changes in <S^2>.

    max_step_angstrom
        Maximum R spacing at which an automatic continuity assignment is
        allowed. Larger gaps remain AMBIGUOUS even if orbital overlap happens
        to be high.
    """

    continuous_occ_min: float
    discontinuous_occ_min: float

    continuous_delta_s2: float
    discontinuous_delta_s2: float

    max_step_angstrom: float

    def __post_init__(self) -> None:
        values = (
            self.continuous_occ_min,
            self.discontinuous_occ_min,
            self.continuous_delta_s2,
            self.discontinuous_delta_s2,
            self.max_step_angstrom,
        )

        if any(
            not isfinite(float(x))
            for x in values
        ):
            raise ValueError(
                "branch thresholds must be finite"
            )

        if not (
            0.0
            <= self.discontinuous_occ_min
            < self.continuous_occ_min
            <= 1.0
        ):
            raise ValueError(
                "occupied-overlap thresholds must satisfy "
                "0 <= discontinuous < continuous <= 1"
            )

        if not (
            0.0
            <= self.continuous_delta_s2
            < self.discontinuous_delta_s2
        ):
            raise ValueError(
                "delta-S2 thresholds must satisfy "
                "0 <= continuous < discontinuous"
            )

        if self.max_step_angstrom <= 0.0:
            raise ValueError(
                "max_step_angstrom must be > 0"
            )


@dataclass(frozen=True)
class BranchComparison:
    root_a: str
    root_b: str

    r_a_angstrom: float
    r_b_angstrom: float
    delta_r_angstrom: float

    delta_energy_mev: float
    delta_s2: float

    alpha_singular_values: tuple[
        float,
        ...,
    ]
    beta_singular_values: tuple[
        float,
        ...,
    ]

    alpha_occ_overlap_min: float
    alpha_occ_overlap_mean: float

    beta_occ_overlap_min: float
    beta_occ_overlap_mean: float

    relation: BranchRelation


def classify_branch_metrics(
    *,
    delta_r_angstrom: float,
    delta_s2: float,
    alpha_occ_overlap_min: float,
    beta_occ_overlap_min: float,
    thresholds: BranchThresholds,
) -> BranchRelation:
    """
    Conservative branch-continuity classification.

    Energy is deliberately absent from this decision.

    Along a physical PEC a branch energy can move strongly with R, so energy
    proximity is stored as a diagnostic but is not used as primary branch
    identity evidence.
    """
    values = (
        delta_r_angstrom,
        delta_s2,
        alpha_occ_overlap_min,
        beta_occ_overlap_min,
    )

    if any(
        not isfinite(float(x))
        for x in values
    ):
        raise ValueError(
            "branch metrics must be finite"
        )

    dr = abs(
        float(delta_r_angstrom)
    )

    ds2 = abs(
        float(delta_s2)
    )

    alpha = float(
        alpha_occ_overlap_min
    )

    beta = float(
        beta_occ_overlap_min
    )

    if not (
        0.0 <= alpha <= 1.0
        and 0.0 <= beta <= 1.0
    ):
        raise ValueError(
            "occupied-subspace overlaps must lie in [0, 1]"
        )

    if dr <= 0.0:
        raise ValueError(
            "branch comparison requires different geometries"
        )

    if dr > thresholds.max_step_angstrom:
        return BranchRelation.AMBIGUOUS

    worst_overlap = min(
        alpha,
        beta,
    )

    if (
        worst_overlap
        >= thresholds.continuous_occ_min
        and ds2
        <= thresholds.continuous_delta_s2
    ):
        return BranchRelation.CONTINUOUS

    if (
        worst_overlap
        <= thresholds.discontinuous_occ_min
        or ds2
        >= thresholds.discontinuous_delta_s2
    ):
        return BranchRelation.DISCONTINUOUS

    return BranchRelation.AMBIGUOUS


def _validate_context(
    a: SCFRootRecord,
    b: SCFRootRecord,
) -> None:
    """
    Branch following currently compares the same method/spin sector at
    different geometries.

    Basis changes are handled separately by basis-projection logic.
    Spin changes are separate candidate states, not a continuous branch.
    """
    fields = (
        (
            "molecule",
            a.molecule,
            b.molecule,
        ),
        (
            "charge",
            a.charge,
            b.charge,
        ),
        (
            "spin_2s",
            a.spin_2s,
            b.spin_2s,
        ),
        (
            "functional",
            a.functional,
            b.functional,
        ),
        (
            "basis",
            a.basis,
            b.basis,
        ),
        (
            "reference",
            a.reference,
            b.reference,
        ),
        (
            "ecp_assignments",
            a.ecp_assignments,
            b.ecp_assignments,
        ),
    )

    for name, x, y in fields:
        if x != y:
            raise ValueError(
                "cannot compare branch continuity across different "
                f"{name}: {x!r} != {y!r}"
            )

    if abs(
        a.r_angstrom
        - b.r_angstrom
    ) <= 1.0e-12:
        raise ValueError(
            "branch continuity requires different geometries"
        )

    for root in (
        a,
        b,
    ):
        if (
            root.status
            != SCFRunStatus.CANONICALIZED
        ):
            raise ValueError(
                "branch continuity requires canonicalized roots"
            )

        if root.energy_hartree is None:
            raise ValueError(
                "branch continuity requires root energies"
            )

        if root.s2 is None:
            raise ValueError(
                "branch continuity requires <S^2>"
            )

        if not root.checkpoint_path:
            raise ValueError(
                "branch continuity requires final checkpoints"
            )


def _load_checkpoint_mos(
    root: SCFRootRecord,
):
    from pyscf.scf import chkfile

    mol, data = chkfile.load_scf(
        root.checkpoint_path
    )

    if not data:
        raise ValueError(
            "checkpoint contains no SCF data"
        )

    if (
        "mo_coeff" not in data
        or "mo_occ" not in data
    ):
        raise ValueError(
            "checkpoint lacks MO coefficients or occupations"
        )

    coeff = np.asarray(
        data["mo_coeff"]
    )

    occ = np.asarray(
        data["mo_occ"]
    )

    if root.reference == "RKS":
        if (
            coeff.ndim != 2
            or occ.ndim != 1
        ):
            raise ValueError(
                "RKS checkpoint has unexpected MO dimensions"
            )

        occupied = coeff[
            :,
            occ > 0.0,
        ]

        return (
            mol,
            occupied,
            occupied,
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

        #
        # Select the spin block first, then apply the MO occupation mask.
        #
        # Writing coeff[0, :, mask] mixes basic and advanced NumPy
        # indexing and moves the masked MO axis to the front, producing
        # (nocc, nao) instead of the required (nao, nocc).
        #
        # The two-step indexing below also preserves the correct
        # (nao, 0) shape for an empty occupied spin channel.
        #
        alpha = coeff[0][
            :,
            occ[0] > 0.0,
        ]

        beta = coeff[1][
            :,
            occ[1] > 0.0,
        ]

        return (
            mol,
            alpha,
            beta,
        )

    raise ValueError(
        f"unsupported reference {root.reference!r}"
    )


def _subspace_singular_values(
    coeff_a,
    coeff_b,
    cross_overlap,
) -> tuple[float, ...]:
    """
    Principal occupied-subspace overlaps between two geometries.

    Each occupied MO set is internally orthonormal in its own AO metric.
    The singular values of

        C_A^H S_AB C_B

    are therefore the cosines of the principal angles between the occupied
    subspaces represented at the two geometries.
    """
    ca = np.asarray(
        coeff_a
    )

    cb = np.asarray(
        coeff_b
    )

    sab = np.asarray(
        cross_overlap
    )

    if (
        ca.ndim != 2
        or cb.ndim != 2
        or sab.ndim != 2
    ):
        raise ValueError(
            "MO coefficients and cross overlap must be matrices"
        )

    if ca.shape[0] != sab.shape[0]:
        raise ValueError(
            "left AO dimension does not match cross overlap"
        )

    if cb.shape[0] != sab.shape[1]:
        raise ValueError(
            "right AO dimension does not match cross overlap"
        )

    if ca.shape[1] != cb.shape[1]:
        raise ValueError(
            "occupied-subspace dimensions differ"
        )

    if ca.shape[1] == 0:
        #
        # Empty beta spaces are possible for one-electron systems.
        # There is no beta occupied subspace to lose continuity.
        #
        return ()

    overlap = (
        ca.conj().T
        @ sab
        @ cb
    )

    singular = np.linalg.svd(
        overlap,
        compute_uv=False,
    )

    cleaned = []

    for value in singular:
        x = float(
            np.real_if_close(
                value
            )
        )

        if x < -1.0e-10:
            raise ValueError(
                "negative singular value encountered"
            )

        if x > 1.0 + 1.0e-7:
            raise ValueError(
                "occupied-subspace singular value exceeds 1 "
                f"beyond numerical tolerance: {x}"
            )

        cleaned.append(
            min(
                1.0,
                max(
                    0.0,
                    x,
                ),
            )
        )

    return tuple(
        sorted(
            cleaned,
            reverse=True,
        )
    )


def _summary_overlap(
    values: tuple[
        float,
        ...,
    ],
) -> tuple[
    float,
    float,
]:
    if not values:
        #
        # Empty spin channel: no occupied subspace exists in either root.
        # Treat this channel as trivially continuous.
        #
        return (
            1.0,
            1.0,
        )

    array = np.asarray(
        values,
        dtype=float,
    )

    return (
        float(
            array.min()
        ),
        float(
            array.mean()
        ),
    )


def compare_branch_roots(
    a: SCFRootRecord,
    b: SCFRootRecord,
    *,
    thresholds: BranchThresholds,
    audit_settings: CheckpointAuditSettings | None = None,
) -> BranchComparison:
    """
    Compare two adjacent-geometry roots using occupied-subspace continuity.

    Both checkpoints are first passed through the existing checkpoint audit.
    """
    from pyscf import gto

    _validate_context(
        a,
        b,
    )

    #
    # This audits charge, spin, energy, electron trace, and checkpoint
    # consistency before any branch metric is trusted.
    #
    fingerprint_from_checkpoint(
        a,
        settings=audit_settings,
    )

    fingerprint_from_checkpoint(
        b,
        settings=audit_settings,
    )

    (
        mol_a,
        alpha_a,
        beta_a,
    ) = _load_checkpoint_mos(
        a
    )

    (
        mol_b,
        alpha_b,
        beta_b,
    ) = _load_checkpoint_mos(
        b
    )

    if mol_a.natm != mol_b.natm:
        raise ValueError(
            "checkpoint molecules have different atom counts"
        )

    symbols_a = tuple(
        mol_a.atom_pure_symbol(i)
        for i in range(
            mol_a.natm
        )
    )

    symbols_b = tuple(
        mol_b.atom_pure_symbol(i)
        for i in range(
            mol_b.natm
        )
    )

    if symbols_a != symbols_b:
        raise ValueError(
            "checkpoint atom ordering differs between geometries"
        )

    cross_overlap = (
        gto.intor_cross(
            "int1e_ovlp",
            mol_a,
            mol_b,
        )
    )

    alpha_sv = (
        _subspace_singular_values(
            alpha_a,
            alpha_b,
            cross_overlap,
        )
    )

    beta_sv = (
        _subspace_singular_values(
            beta_a,
            beta_b,
            cross_overlap,
        )
    )

    (
        alpha_min,
        alpha_mean,
    ) = _summary_overlap(
        alpha_sv
    )

    (
        beta_min,
        beta_mean,
    ) = _summary_overlap(
        beta_sv
    )

    delta_r = abs(
        b.r_angstrom
        - a.r_angstrom
    )

    delta_s2 = abs(
        float(
            b.s2
            - a.s2
        )
    )

    delta_energy_mev = (
        float(
            b.energy_hartree
            - a.energy_hartree
        )
        * HARTREE_TO_EV
        * 1000.0
    )

    relation = (
        classify_branch_metrics(
            delta_r_angstrom=(
                delta_r
            ),
            delta_s2=(
                delta_s2
            ),
            alpha_occ_overlap_min=(
                alpha_min
            ),
            beta_occ_overlap_min=(
                beta_min
            ),
            thresholds=thresholds,
        )
    )

    return BranchComparison(
        root_a=a.root_id,
        root_b=b.root_id,
        r_a_angstrom=a.r_angstrom,
        r_b_angstrom=b.r_angstrom,
        delta_r_angstrom=delta_r,
        delta_energy_mev=(
            delta_energy_mev
        ),
        delta_s2=delta_s2,
        alpha_singular_values=(
            alpha_sv
        ),
        beta_singular_values=(
            beta_sv
        ),
        alpha_occ_overlap_min=(
            alpha_min
        ),
        alpha_occ_overlap_mean=(
            alpha_mean
        ),
        beta_occ_overlap_min=(
            beta_min
        ),
        beta_occ_overlap_mean=(
            beta_mean
        ),
        relation=relation,
    )
