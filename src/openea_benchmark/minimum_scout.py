from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .local_pec import (
    LocalPEC,
)


class MinimumScoutStatus(str, Enum):
    """
    Conservative classification of the discrete energetic shape of a
    LocalPEC.

    These statuses describe only what is supported by the sampled points.
    They do not assert an equilibrium geometry or an unsampled minimum.
    """

    BRACKETED_SINGLE_MINIMUM = (
        "bracketed_single_minimum"
    )

    MULTIPLE_MINIMUM_CANDIDATES = (
        "multiple_minimum_candidates"
    )

    DECREASES_TOWARD_HIGHER_R = (
        "decreases_toward_higher_r"
    )

    DECREASES_TOWARD_LOWER_R = (
        "decreases_toward_lower_r"
    )

    FLAT_OR_UNRESOLVED = (
        "flat_or_unresolved"
    )

    COMPLEX_SHAPE = (
        "complex_shape"
    )


@dataclass(frozen=True)
class MinimumScoutThresholds:
    """
    Numerical resolution used only for discrete PEC-shape interpretation.

    No production default is provided deliberately.
    """

    energy_tolerance_hartree: float

    def __post_init__(self) -> None:
        value = float(
            self.energy_tolerance_hartree
        )

        if (
            not isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(
                "energy_tolerance_hartree must be finite and > 0"
            )

        object.__setattr__(
            self,
            "energy_tolerance_hartree",
            value,
        )


@dataclass(frozen=True)
class MinimumCandidate:
    """
    One strictly bracketed discrete local-minimum candidate.

    The candidate is the sampled point itself. No interpolation, fitting,
    or equilibrium-geometry estimate is performed.
    """

    point_index: int

    root_id: str
    r_angstrom: float
    energy_hartree: float

    left_root_id: str
    left_r_angstrom: float
    left_energy_hartree: float

    right_root_id: str
    right_r_angstrom: float
    right_energy_hartree: float


@dataclass(frozen=True)
class MinimumScoutResult:
    """
    Result of conservative discrete LocalPEC inspection.
    """

    component_id: str

    status: MinimumScoutStatus

    candidates: tuple[
        MinimumCandidate,
        ...,
    ]

    adjacent_delta_e_hartree: tuple[
        float,
        ...,
    ]


def _step_sign(
    delta_e_hartree: float,
    *,
    tolerance: float,
) -> int:
    """
    Classify E(R[i+1]) - E(R[i]).

    -1: energy decreases toward higher R
     0: unresolved within numerical tolerance
    +1: energy increases toward higher R
    """
    if (
        delta_e_hartree
        < -tolerance
    ):
        return -1

    if (
        delta_e_hartree
        > tolerance
    ):
        return 1

    return 0


def _candidate_at(
    pec: LocalPEC,
    index: int,
) -> MinimumCandidate:
    left = pec.points[
        index - 1
    ]

    point = pec.points[
        index
    ]

    right = pec.points[
        index + 1
    ]

    return MinimumCandidate(
        point_index=index,

        root_id=point.root_id,
        r_angstrom=(
            point.r_angstrom
        ),
        energy_hartree=(
            point.energy_hartree
        ),

        left_root_id=(
            left.root_id
        ),
        left_r_angstrom=(
            left.r_angstrom
        ),
        left_energy_hartree=(
            left.energy_hartree
        ),

        right_root_id=(
            right.root_id
        ),
        right_r_angstrom=(
            right.r_angstrom
        ),
        right_energy_hartree=(
            right.energy_hartree
        ),
    )


def scout_local_pec_minimum(
    pec: LocalPEC,
    *,
    thresholds: MinimumScoutThresholds,
) -> MinimumScoutResult:
    """
    Inspect the discrete energetic shape of one LocalPEC.

    This function deliberately does not:

    - interpolate the PEC;
    - fit a polynomial, spline, Morse potential, or other model;
    - estimate an equilibrium geometry;
    - assert that an unsampled minimum exists;
    - rank multiple local minima by energy;
    - compare separate electronic branches or spin sectors;
    - calculate an electron affinity.

    A sampled point is a strict discrete local-minimum candidate only when
    both direct neighbors are higher in energy by more than the explicit
    numerical tolerance.
    """
    if not isinstance(
        pec,
        LocalPEC,
    ):
        raise TypeError(
            "pec must be a LocalPEC"
        )

    tolerance = (
        thresholds.energy_tolerance_hartree
    )

    deltas = tuple(
        pec.points[index + 1].energy_hartree
        - pec.points[index].energy_hartree
        for index in range(
            len(pec.points) - 1
        )
    )

    signs = tuple(
        _step_sign(
            delta,
            tolerance=tolerance,
        )
        for delta in deltas
    )

    candidates = tuple(
        _candidate_at(
            pec,
            index,
        )
        for index in range(
            1,
            len(pec.points) - 1,
        )
        if (
            signs[index - 1] == -1
            and signs[index] == 1
        )
    )

    #
    # Numerical non-resolution takes priority over shape interpretation.
    #
    # Strictly bracketed candidates are still retained for provenance, but
    # the PEC as a whole is not classified as resolved while any adjacent
    # energy step remains within the explicit numerical tolerance.
    #
    if any(
        sign == 0
        for sign in signs
    ):
        status = (
            MinimumScoutStatus
            .FLAT_OR_UNRESOLVED
        )

    #
    # More than one strict local minimum is retained explicitly.
    # No energetic ranking or winner selection is allowed here.
    #
    elif len(candidates) > 1:
        status = (
            MinimumScoutStatus
            .MULTIPLE_MINIMUM_CANDIDATES
        )

    elif len(candidates) == 1:
        candidate = (
            candidates[0]
        )

        index = (
            candidate.point_index
        )

        clean_left = all(
            sign == -1
            for sign in signs[
                :index
            ]
        )

        clean_right = all(
            sign == 1
            for sign in signs[
                index:
            ]
        )

        if (
            clean_left
            and clean_right
        ):
            status = (
                MinimumScoutStatus
                .BRACKETED_SINGLE_MINIMUM
            )
        else:
            status = (
                MinimumScoutStatus
                .COMPLEX_SHAPE
            )

    elif all(
        sign == -1
        for sign in signs
    ):
        status = (
            MinimumScoutStatus
            .DECREASES_TOWARD_HIGHER_R
        )

    elif all(
        sign == 1
        for sign in signs
    ):
        status = (
            MinimumScoutStatus
            .DECREASES_TOWARD_LOWER_R
        )

    else:
        status = (
            MinimumScoutStatus
            .COMPLEX_SHAPE
        )

    return MinimumScoutResult(
        component_id=(
            pec.component_id
        ),
        status=status,
        candidates=candidates,
        adjacent_delta_e_hartree=(
            deltas
        ),
    )
