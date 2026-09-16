from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Sequence

from .branch_graph import (
    BranchGraph,
)
from .root_record import (
    SCFRootRecord,
)


class LocalPECRejectionReason(str, Enum):
    """
    Reason why a BranchComponent cannot yet be promoted to a local PEC.

    These are scientific workflow outcomes, not malformed-input errors.
    """

    BRANCHING_TOPOLOGY = "BRANCHING_TOPOLOGY"
    AMBIGUOUS_BOUNDARY = "AMBIGUOUS_BOUNDARY"
    INSUFFICIENT_POINTS = "INSUFFICIENT_POINTS"


@dataclass(frozen=True)
class LocalPECPoint:
    """
    One geometry/energy point on a uniquely traced local electronic branch.
    """

    root_id: str
    r_angstrom: float
    energy_hartree: float

    def __post_init__(self) -> None:
        if not self.root_id.strip():
            raise ValueError(
                "root_id must be non-empty"
            )

        if (
            not isfinite(self.r_angstrom)
            or self.r_angstrom <= 0.0
        ):
            raise ValueError(
                "r_angstrom must be finite and > 0"
            )

        if not isfinite(
            self.energy_hartree
        ):
            raise ValueError(
                "energy_hartree must be finite"
            )


@dataclass(frozen=True)
class LocalPEC:
    """
    Ordered local potential-energy curve segment.

    A LocalPEC is only a uniquely traced electronic branch over at least two
    geometries. It does not identify a minimum, equilibrium geometry, ground
    state, or final physical electronic state.
    """

    component_id: str
    points: tuple[
        LocalPECPoint,
        ...,
    ]

    def __post_init__(self) -> None:
        if not self.component_id.strip():
            raise ValueError(
                "component_id must be non-empty"
            )

        if len(self.points) < 2:
            raise ValueError(
                "local PEC requires at least two points"
            )

        for left, right in zip(
            self.points,
            self.points[1:],
        ):
            if (
                right.r_angstrom
                <= left.r_angstrom
            ):
                raise ValueError(
                    "local PEC points must be strictly ordered by geometry"
                )


@dataclass(frozen=True)
class RejectedLocalPECComponent:
    """
    Branch component retained explicitly because it cannot become a LocalPEC.
    """

    component_id: str
    member_root_ids: tuple[
        str,
        ...,
    ]
    geometry_values: tuple[
        float,
        ...,
    ]
    reasons: tuple[
        LocalPECRejectionReason,
        ...,
    ]

    def __post_init__(self) -> None:
        if not self.component_id.strip():
            raise ValueError(
                "component_id must be non-empty"
            )

        if not self.member_root_ids:
            raise ValueError(
                "rejected component must contain at least one root"
            )

        if not self.geometry_values:
            raise ValueError(
                "rejected component must contain at least one geometry"
            )

        if not self.reasons:
            raise ValueError(
                "rejected component must contain at least one reason"
            )


@dataclass(frozen=True)
class LocalPECConstructionResult:
    """
    Complete result of promoting BranchGraph components to local PECs.

    Components are never silently discarded.
    """

    pecs: tuple[
        LocalPEC,
        ...,
    ]
    rejected_components: tuple[
        RejectedLocalPECComponent,
        ...,
    ]


def _graph_root_ids(
    graph: BranchGraph,
) -> tuple[
    str,
    ...,
]:
    root_ids = []

    for _, ids in graph.root_ids_by_geometry:
        root_ids.extend(
            ids
        )

    return tuple(
        root_ids
    )


def _validate_inputs(
    roots: Sequence[
        SCFRootRecord
    ],
    graph: BranchGraph,
) -> dict[
    str,
    SCFRootRecord,
]:
    root_by_id: dict[
        str,
        SCFRootRecord,
    ] = {}

    for root in roots:
        if root.root_id in root_by_id:
            raise ValueError(
                f"duplicate root_id: {root.root_id}"
            )

        root_by_id[
            root.root_id
        ] = root

    supplied_ids = set(
        root_by_id
    )
    graph_ids_tuple = (
        _graph_root_ids(
            graph
        )
    )
    graph_ids = set(
        graph_ids_tuple
    )

    if (
        len(graph_ids)
        != len(graph_ids_tuple)
    ):
        raise ValueError(
            "branch graph contains duplicate root ids"
        )

    if supplied_ids != graph_ids:
        missing = sorted(
            graph_ids
            - supplied_ids
        )
        extra = sorted(
            supplied_ids
            - graph_ids
        )

        raise ValueError(
            "root set does not match branch graph; "
            f"missing={missing}, extra={extra}"
        )

    for r_angstrom, root_ids in (
        graph.root_ids_by_geometry
    ):
        for root_id in root_ids:
            root = root_by_id[
                root_id
            ]

            if abs(
                root.r_angstrom
                - r_angstrom
            ) > 1.0e-10:
                raise ValueError(
                    "root geometry does not match branch graph layer"
                )

    component_members = []

    for component in graph.components:
        for root_id in (
            component.member_root_ids
        ):
            if root_id not in graph_ids:
                raise ValueError(
                    "branch component references a root outside the graph"
                )

            component_members.append(
                root_id
            )

    if (
        len(component_members)
        != len(set(component_members))
    ):
        raise ValueError(
            "branch graph components contain duplicate root membership"
        )

    if set(component_members) != graph_ids:
        raise ValueError(
            "branch graph components do not cover the root set exactly"
        )

    return root_by_id


def _component_rejection_reasons(
    component,
) -> tuple[
    LocalPECRejectionReason,
    ...,
]:
    reasons = []

    if component.has_branching:
        reasons.append(
            LocalPECRejectionReason.BRANCHING_TOPOLOGY
        )

    if component.has_ambiguous_boundary:
        reasons.append(
            LocalPECRejectionReason.AMBIGUOUS_BOUNDARY
        )

    if len(
        component.geometry_values
    ) < 2:
        reasons.append(
            LocalPECRejectionReason.INSUFFICIENT_POINTS
        )

    return tuple(
        reasons
    )


def construct_local_pecs(
    roots: Sequence[
        SCFRootRecord
    ],
    graph: BranchGraph,
) -> LocalPECConstructionResult:
    """
    Promote unambiguous BranchGraph components to ordered local PECs.

    No energy-based branch selection is performed.

    No minimum search, interpolation, fitting, equilibrium-geometry
    assignment, cross-spin comparison, or ground-state assignment is
    performed here.
    """

    root_by_id = _validate_inputs(
        roots,
        graph,
    )

    pecs = []
    rejected = []

    for component in graph.components:
        reasons = (
            _component_rejection_reasons(
                component
            )
        )

        if reasons:
            rejected.append(
                RejectedLocalPECComponent(
                    component_id=(
                        component.component_id
                    ),
                    member_root_ids=(
                        component.member_root_ids
                    ),
                    geometry_values=(
                        component.geometry_values
                    ),
                    reasons=reasons,
                )
            )
            continue

        points = []

        for root_id in (
            component.member_root_ids
        ):
            root = root_by_id[
                root_id
            ]

            if root.energy_hartree is None:
                raise ValueError(
                    "local PEC construction requires root energies"
                )

            energy = float(
                root.energy_hartree
            )

            if not isfinite(
                energy
            ):
                raise ValueError(
                    "local PEC construction requires finite root energies"
                )

            points.append(
                LocalPECPoint(
                    root_id=root.root_id,
                    r_angstrom=float(
                        root.r_angstrom
                    ),
                    energy_hartree=energy,
                )
            )

        points.sort(
            key=lambda point: (
                point.r_angstrom,
                point.root_id,
            )
        )

        pecs.append(
            LocalPEC(
                component_id=(
                    component.component_id
                ),
                points=tuple(
                    points
                ),
            )
        )

    return LocalPECConstructionResult(
        pecs=tuple(
            pecs
        ),
        rejected_components=tuple(
            rejected
        ),
    )
