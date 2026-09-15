from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

from .branch_continuity import (
    BranchComparison,
    BranchRelation,
    BranchThresholds,
    compare_branch_roots,
)
from .checkpoint_fingerprint import (
    CheckpointAuditSettings,
)
from .root_record import (
    SCFRootRecord,
)


@dataclass(frozen=True)
class BranchComponent:
    """
    Connected component formed only from CONTINUOUS branch edges.

    A component is not automatically a physical electronic state.

    has_branching
        At least one geometry contains more than one root inside the same
        CONTINUOUS component. This corresponds to a split/merge topology and
        prevents automatic unique-path assignment.

    has_ambiguous_boundary
        At least one member participates in an AMBIGUOUS comparison with
        another root. The component therefore cannot yet be regarded as
        uniquely isolated from its alternatives.

    is_unambiguous
        Convenience flag requiring neither of the above conditions.
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

    has_branching: bool
    has_ambiguous_boundary: bool
    is_unambiguous: bool

    def __post_init__(self) -> None:
        if not self.component_id.strip():
            raise ValueError(
                "component_id must be non-empty"
            )

        if not self.member_root_ids:
            raise ValueError(
                "branch component must contain at least one root"
            )

        if not self.geometry_values:
            raise ValueError(
                "branch component must contain at least one geometry"
            )

        expected = not (
            self.has_branching
            or self.has_ambiguous_boundary
        )

        if self.is_unambiguous != expected:
            raise ValueError(
                "is_unambiguous is inconsistent with component flags"
            )


@dataclass(frozen=True)
class BranchGraph:
    """
    Layered electronic-state continuity graph.

    The graph contains every supplied root. Edges exist only between
    neighboring geometry layers.

    No energy minimization and no ground-state assignment is performed.
    """

    geometry_values: tuple[
        float,
        ...,
    ]

    root_ids_by_geometry: tuple[
        tuple[
            float,
            tuple[
                str,
                ...,
            ],
        ],
        ...,
    ]

    comparisons: tuple[
        BranchComparison,
        ...,
    ]

    continuous_edges: tuple[
        tuple[
            str,
            str,
        ],
        ...,
    ]

    ambiguous_edges: tuple[
        tuple[
            str,
            str,
        ],
        ...,
    ]

    discontinuous_edges: tuple[
        tuple[
            str,
            str,
        ],
        ...,
    ]

    components: tuple[
        BranchComponent,
        ...,
    ]

    topology_ambiguous_root_ids: tuple[
        str,
        ...,
    ]

    ambiguous_edge_root_ids: tuple[
        str,
        ...,
    ]

    @property
    def is_fully_unambiguous(self) -> bool:
        return (
            not self.ambiguous_edges
            and not self.topology_ambiguous_root_ids
            and all(
                component.is_unambiguous
                for component in self.components
            )
        )


def _root_context(
    root: SCFRootRecord,
) -> tuple:
    """
    Electronic-method context that must remain fixed while following one
    branch graph.

    Separate spin sectors form separate graphs. Their energetic competition
    is assessed at a higher workflow layer.
    """
    return (
        root.molecule,
        root.charge,
        root.spin_2s,
        root.functional,
        root.basis,
        root.reference,
        root.ecp_assignments,
    )


def _validate_roots(
    roots: Sequence[
        SCFRootRecord
    ],
) -> dict[
    str,
    SCFRootRecord,
]:
    if not roots:
        raise ValueError(
            "branch graph requires at least one root"
        )

    root_by_id = {}

    for root in roots:
        if root.root_id in root_by_id:
            raise ValueError(
                f"duplicate root_id: {root.root_id}"
            )

        root_by_id[
            root.root_id
        ] = root

    contexts = {
        _root_context(root)
        for root in roots
    }

    if len(contexts) != 1:
        raise ValueError(
            "all roots in one branch graph must share molecule, charge, "
            "spin sector, functional, basis, reference, and ECP assignment"
        )

    return root_by_id


def _layers(
    roots: Sequence[
        SCFRootRecord
    ],
) -> tuple[
    tuple[
        float,
        tuple[
            SCFRootRecord,
            ...,
        ],
    ],
    ...,
]:
    by_r: dict[
        float,
        list[
            SCFRootRecord
        ],
    ] = {}

    for root in roots:
        by_r.setdefault(
            root.r_angstrom,
            [],
        ).append(root)

    return tuple(
        (
            r,
            tuple(
                sorted(
                    by_r[r],
                    key=lambda item:
                        item.root_id,
                )
            ),
        )
        for r in sorted(
            by_r
        )
    )


def _pair_key(
    root_a: str,
    root_b: str,
) -> frozenset[str]:
    if root_a == root_b:
        raise ValueError(
            "comparison cannot connect a root to itself"
        )

    return frozenset(
        (
            root_a,
            root_b,
        )
    )


def _canonical_edge(
    root_a: SCFRootRecord,
    root_b: SCFRootRecord,
) -> tuple[
    str,
    str,
]:
    """
    Deterministically orient an edge from lower R to higher R.
    """
    if root_a.r_angstrom < root_b.r_angstrom:
        return (
            root_a.root_id,
            root_b.root_id,
        )

    if root_b.r_angstrom < root_a.r_angstrom:
        return (
            root_b.root_id,
            root_a.root_id,
        )

    raise ValueError(
        "branch edge endpoints must be at different geometries"
    )


def _expected_adjacent_pairs(
    layers,
) -> dict[
    frozenset[str],
    tuple[
        SCFRootRecord,
        SCFRootRecord,
    ],
]:
    expected = {}

    for (
        _,
        left_roots,
    ), (
        _,
        right_roots,
    ) in zip(
        layers,
        layers[1:],
    ):
        for left, right in product(
            left_roots,
            right_roots,
        ):
            key = _pair_key(
                left.root_id,
                right.root_id,
            )

            expected[key] = (
                left,
                right,
            )

    return expected


def _validate_comparisons(
    roots: Sequence[
        SCFRootRecord
    ],
    comparisons: Sequence[
        BranchComparison
    ],
):
    root_by_id = _validate_roots(
        roots
    )

    layers = _layers(
        roots
    )

    expected = (
        _expected_adjacent_pairs(
            layers
        )
    )

    supplied = {}

    for comparison in comparisons:
        if (
            comparison.root_a
            not in root_by_id
            or comparison.root_b
            not in root_by_id
        ):
            raise ValueError(
                "branch comparison references an unknown root"
            )

        key = _pair_key(
            comparison.root_a,
            comparison.root_b,
        )

        if key in supplied:
            raise ValueError(
                "duplicate branch comparison"
            )

        if key not in expected:
            raise ValueError(
                "branch comparison is not between adjacent geometry layers"
            )

        a = root_by_id[
            comparison.root_a
        ]

        b = root_by_id[
            comparison.root_b
        ]

        if abs(
            comparison.r_a_angstrom
            - a.r_angstrom
        ) > 1.0e-10:
            raise ValueError(
                "comparison r_a does not match its root record"
            )

        if abs(
            comparison.r_b_angstrom
            - b.r_angstrom
        ) > 1.0e-10:
            raise ValueError(
                "comparison r_b does not match its root record"
            )

        supplied[
            key
        ] = comparison

    missing = (
        set(expected)
        - set(supplied)
    )

    extra = (
        set(supplied)
        - set(expected)
    )

    if missing:
        pairs = sorted(
            tuple(
                sorted(key)
            )
            for key in missing
        )

        raise ValueError(
            "branch comparison matrix is incomplete; "
            f"missing adjacent pairs: {pairs}"
        )

    if extra:
        raise ValueError(
            "unexpected branch comparisons are present"
        )

    return (
        root_by_id,
        layers,
    )


def _continuous_components(
    root_by_id: Mapping[
        str,
        SCFRootRecord,
    ],
    comparisons: Sequence[
        BranchComparison
    ],
):
    adjacency: dict[
        str,
        set[str],
    ] = {
        root_id: set()
        for root_id
        in root_by_id
    }

    ambiguous_touch: set[
        str
    ] = set()

    continuous_edges = []
    ambiguous_edges = []
    discontinuous_edges = []

    for comparison in comparisons:
        a = root_by_id[
            comparison.root_a
        ]

        b = root_by_id[
            comparison.root_b
        ]

        edge = _canonical_edge(
            a,
            b,
        )

        if (
            comparison.relation
            == BranchRelation.CONTINUOUS
        ):
            continuous_edges.append(
                edge
            )

            adjacency[
                a.root_id
            ].add(
                b.root_id
            )

            adjacency[
                b.root_id
            ].add(
                a.root_id
            )

        elif (
            comparison.relation
            == BranchRelation.AMBIGUOUS
        ):
            ambiguous_edges.append(
                edge
            )

            ambiguous_touch.update(
                edge
            )

        elif (
            comparison.relation
            == BranchRelation.DISCONTINUOUS
        ):
            discontinuous_edges.append(
                edge
            )

        else:
            raise ValueError(
                f"unknown BranchRelation: {comparison.relation}"
            )

    unseen = set(
        root_by_id
    )

    raw_components = []

    while unseen:
        start = min(
            unseen
        )

        stack = [
            start
        ]

        members = set()

        while stack:
            current = stack.pop()

            if current in members:
                continue

            members.add(
                current
            )

            unseen.discard(
                current
            )

            stack.extend(
                sorted(
                    adjacency[
                        current
                    ]
                    - members
                )
            )

        raw_components.append(
            tuple(
                sorted(
                    members,
                    key=lambda root_id: (
                        root_by_id[
                            root_id
                        ].r_angstrom,
                        root_id,
                    ),
                )
            )
        )

    raw_components.sort(
        key=lambda members: (
            root_by_id[
                members[0]
            ].r_angstrom,
            members,
        )
    )

    topology_ambiguous = set()
    components = []

    for index, members in enumerate(
        raw_components,
        start=1,
    ):
        counts: dict[
            float,
            int,
        ] = {}

        for root_id in members:
            r = root_by_id[
                root_id
            ].r_angstrom

            counts[r] = (
                counts.get(
                    r,
                    0,
                )
                + 1
            )

        duplicate_geometries = {
            r
            for r, count
            in counts.items()
            if count > 1
        }

        has_branching = bool(
            duplicate_geometries
        )

        if has_branching:
            for root_id in members:
                if (
                    root_by_id[
                        root_id
                    ].r_angstrom
                    in duplicate_geometries
                ):
                    topology_ambiguous.add(
                        root_id
                    )

            #
            # The entire connected component participates in a split/merge
            # topology, so no member can yet be treated as a uniquely traced
            # physical branch.
            #
            topology_ambiguous.update(
                members
            )

        has_ambiguous_boundary = any(
            root_id
            in ambiguous_touch
            for root_id in members
        )

        geometries = tuple(
            sorted(
                counts
            )
        )

        components.append(
            BranchComponent(
                component_id=(
                    f"branch_component_{index:04d}"
                ),
                member_root_ids=members,
                geometry_values=(
                    geometries
                ),
                has_branching=(
                    has_branching
                ),
                has_ambiguous_boundary=(
                    has_ambiguous_boundary
                ),
                is_unambiguous=(
                    not has_branching
                    and not has_ambiguous_boundary
                ),
            )
        )

    return (
        tuple(
            sorted(
                continuous_edges
            )
        ),
        tuple(
            sorted(
                ambiguous_edges
            )
        ),
        tuple(
            sorted(
                discontinuous_edges
            )
        ),
        tuple(
            components
        ),
        tuple(
            sorted(
                topology_ambiguous
            )
        ),
        tuple(
            sorted(
                ambiguous_touch
            )
        ),
    )


def build_branch_graph_from_comparisons(
    roots: Sequence[
        SCFRootRecord
    ],
    comparisons: Sequence[
        BranchComparison
    ],
) -> BranchGraph:
    """
    Assemble a layered graph from a complete adjacent-layer comparison set.

    This pure graph-building layer is intentionally separate from PySCF so
    graph topology and scientific policy can be unit-tested independently.
    """
    (
        root_by_id,
        layers,
    ) = _validate_comparisons(
        roots,
        comparisons,
    )

    (
        continuous_edges,
        ambiguous_edges,
        discontinuous_edges,
        components,
        topology_ambiguous,
        ambiguous_touch,
    ) = _continuous_components(
        root_by_id,
        comparisons,
    )

    geometry_values = tuple(
        r
        for r, _
        in layers
    )

    root_ids_by_geometry = tuple(
        (
            r,
            tuple(
                root.root_id
                for root
                in layer_roots
            ),
        )
        for r, layer_roots
        in layers
    )

    return BranchGraph(
        geometry_values=(
            geometry_values
        ),
        root_ids_by_geometry=(
            root_ids_by_geometry
        ),
        comparisons=tuple(
            sorted(
                comparisons,
                key=lambda comparison: (
                    min(
                        comparison.r_a_angstrom,
                        comparison.r_b_angstrom,
                    ),
                    max(
                        comparison.r_a_angstrom,
                        comparison.r_b_angstrom,
                    ),
                    min(
                        comparison.root_a,
                        comparison.root_b,
                    ),
                    max(
                        comparison.root_a,
                        comparison.root_b,
                    ),
                ),
            )
        ),
        continuous_edges=(
            continuous_edges
        ),
        ambiguous_edges=(
            ambiguous_edges
        ),
        discontinuous_edges=(
            discontinuous_edges
        ),
        components=(
            components
        ),
        topology_ambiguous_root_ids=(
            topology_ambiguous
        ),
        ambiguous_edge_root_ids=(
            ambiguous_touch
        ),
    )


def build_branch_graph(
    roots: Sequence[
        SCFRootRecord
    ],
    *,
    thresholds: BranchThresholds,
    audit_settings: CheckpointAuditSettings | None = None,
) -> BranchGraph:
    """
    Calculate every adjacent-layer root comparison and assemble the graph.

    The caller should normally supply roots after same-geometry
    deduplication. If duplicate numerical realizations remain, graph
    ambiguity is preserved rather than silently resolved.

    No root is selected because of its energy.
    """
    _validate_roots(
        roots
    )

    layers = _layers(
        roots
    )

    comparisons = []

    for (
        _,
        left_roots,
    ), (
        _,
        right_roots,
    ) in zip(
        layers,
        layers[1:],
    ):
        for left, right in product(
            left_roots,
            right_roots,
        ):
            comparisons.append(
                compare_branch_roots(
                    left,
                    right,
                    thresholds=thresholds,
                    audit_settings=(
                        audit_settings
                    ),
                )
            )

    return (
        build_branch_graph_from_comparisons(
            roots,
            comparisons,
        )
    )
