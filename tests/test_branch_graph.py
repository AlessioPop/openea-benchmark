from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openea_benchmark import (
    BranchComparison,
    BranchRelation,
    BranchThresholds,
    DFTMethodSpec,
    DiatomicSpec,
    SCFRootRecord,
    SCFRunStatus,
    SCFSettings,
    build_branch_graph,
    build_branch_graph_from_comparisons,
    run_scf_attempt,
)


TEST_THRESHOLDS = BranchThresholds(
    continuous_occ_min=0.95,
    discontinuous_occ_min=0.50,
    continuous_delta_s2=0.02,
    discontinuous_delta_s2=0.20,
    max_step_angstrom=0.10,
)


FAST_SETTINGS = SCFSettings(
    grid_level=1,
    num_threads=1,
)


def make_root(
    root_id: str,
    r: float,
    *,
    energy: float = -10.0,
) -> SCFRootRecord:
    return SCFRootRecord(
        root_id=root_id,
        molecule="XY",
        charge=0,
        spin_2s=1,
        r_angstrom=r,
        functional="r2scan",
        basis="ma-def2-tzvpp",
        origin_guess=root_id,
        scf_path="standard",
        reference="UKS",
        status=SCFRunStatus.CANONICALIZED,
        energy_hartree=energy,
        internal_stable=True,
        s2=0.75,
        observed_multiplicity=2.0,
    )


def make_comparison(
    a: SCFRootRecord,
    b: SCFRootRecord,
    relation: BranchRelation,
) -> BranchComparison:
    return BranchComparison(
        root_a=a.root_id,
        root_b=b.root_id,
        r_a_angstrom=a.r_angstrom,
        r_b_angstrom=b.r_angstrom,
        delta_r_angstrom=abs(
            b.r_angstrom
            - a.r_angstrom
        ),
        delta_energy_mev=0.0,
        delta_s2=0.0,
        alpha_singular_values=(
            0.99,
        ),
        beta_singular_values=(
            0.99,
        ),
        alpha_occ_overlap_min=0.99,
        alpha_occ_overlap_mean=0.99,
        beta_occ_overlap_min=0.99,
        beta_occ_overlap_mean=0.99,
        relation=relation,
    )


class BranchGraphPolicyTests(
    unittest.TestCase
):

    def test_simple_continuous_chain_is_one_unambiguous_component(self):
        a = make_root(
            "a",
            1.00,
        )

        b = make_root(
            "b",
            1.05,
        )

        c = make_root(
            "c",
            1.10,
        )

        graph = (
            build_branch_graph_from_comparisons(
                (
                    a,
                    b,
                    c,
                ),
                (
                    make_comparison(
                        a,
                        b,
                        BranchRelation.CONTINUOUS,
                    ),
                    make_comparison(
                        b,
                        c,
                        BranchRelation.CONTINUOUS,
                    ),
                ),
            )
        )

        self.assertEqual(
            graph.geometry_values,
            (
                1.00,
                1.05,
                1.10,
            ),
        )

        self.assertEqual(
            len(
                graph.components
            ),
            1,
        )

        component = (
            graph.components[0]
        )

        self.assertEqual(
            component.member_root_ids,
            (
                "a",
                "b",
                "c",
            ),
        )

        self.assertFalse(
            component.has_branching
        )

        self.assertFalse(
            component.has_ambiguous_boundary
        )

        self.assertTrue(
            component.is_unambiguous
        )

        self.assertTrue(
            graph.is_fully_unambiguous
        )


    def test_split_merge_topology_is_not_resolved(self):
        a = make_root(
            "a",
            1.00,
        )

        b = make_root(
            "b",
            1.05,
        )

        c = make_root(
            "c",
            1.05,
        )

        d = make_root(
            "d",
            1.10,
        )

        graph = (
            build_branch_graph_from_comparisons(
                (
                    a,
                    b,
                    c,
                    d,
                ),
                (
                    make_comparison(
                        a,
                        b,
                        BranchRelation.CONTINUOUS,
                    ),
                    make_comparison(
                        a,
                        c,
                        BranchRelation.CONTINUOUS,
                    ),
                    make_comparison(
                        b,
                        d,
                        BranchRelation.CONTINUOUS,
                    ),
                    make_comparison(
                        c,
                        d,
                        BranchRelation.CONTINUOUS,
                    ),
                ),
            )
        )

        self.assertEqual(
            len(
                graph.components
            ),
            1,
        )

        component = (
            graph.components[0]
        )

        self.assertTrue(
            component.has_branching
        )

        self.assertFalse(
            component.is_unambiguous
        )

        self.assertEqual(
            set(
                graph.topology_ambiguous_root_ids
            ),
            {
                "a",
                "b",
                "c",
                "d",
            },
        )

        self.assertFalse(
            graph.is_fully_unambiguous
        )


    def test_ambiguous_edge_prevents_unique_assignment(self):
        a = make_root(
            "a",
            1.00,
        )

        b = make_root(
            "b",
            1.05,
        )

        c = make_root(
            "c",
            1.05,
        )

        graph = (
            build_branch_graph_from_comparisons(
                (
                    a,
                    b,
                    c,
                ),
                (
                    make_comparison(
                        a,
                        b,
                        BranchRelation.CONTINUOUS,
                    ),
                    make_comparison(
                        a,
                        c,
                        BranchRelation.AMBIGUOUS,
                    ),
                ),
            )
        )

        component_ab = next(
            component
            for component
            in graph.components
            if "a"
            in component.member_root_ids
        )

        self.assertTrue(
            component_ab.has_ambiguous_boundary
        )

        self.assertFalse(
            component_ab.is_unambiguous
        )

        self.assertEqual(
            graph.ambiguous_edges,
            (
                (
                    "a",
                    "c",
                ),
            ),
        )

        self.assertEqual(
            set(
                graph.ambiguous_edge_root_ids
            ),
            {
                "a",
                "c",
            },
        )


    def test_discontinuous_edges_do_not_connect_components(self):
        a = make_root(
            "a",
            1.00,
        )

        b = make_root(
            "b",
            1.05,
        )

        graph = (
            build_branch_graph_from_comparisons(
                (
                    a,
                    b,
                ),
                (
                    make_comparison(
                        a,
                        b,
                        BranchRelation.DISCONTINUOUS,
                    ),
                ),
            )
        )

        self.assertEqual(
            len(
                graph.components
            ),
            2,
        )

        self.assertEqual(
            graph.continuous_edges,
            (),
        )

        self.assertEqual(
            graph.discontinuous_edges,
            (
                (
                    "a",
                    "b",
                ),
            ),
        )


    def test_incomplete_adjacent_comparison_matrix_is_rejected(self):
        a = make_root(
            "a",
            1.00,
        )

        b = make_root(
            "b",
            1.05,
        )

        c = make_root(
            "c",
            1.05,
        )

        with self.assertRaisesRegex(
            ValueError,
            "incomplete",
        ):
            build_branch_graph_from_comparisons(
                (
                    a,
                    b,
                    c,
                ),
                (
                    make_comparison(
                        a,
                        b,
                        BranchRelation.CONTINUOUS,
                    ),
                ),
            )


    def test_nonadjacent_comparison_is_rejected(self):
        a = make_root(
            "a",
            1.00,
        )

        b = make_root(
            "b",
            1.05,
        )

        c = make_root(
            "c",
            1.10,
        )

        with self.assertRaisesRegex(
            ValueError,
            "not between adjacent",
        ):
            build_branch_graph_from_comparisons(
                (
                    a,
                    b,
                    c,
                ),
                (
                    make_comparison(
                        a,
                        b,
                        BranchRelation.CONTINUOUS,
                    ),
                    make_comparison(
                        b,
                        c,
                        BranchRelation.CONTINUOUS,
                    ),
                    make_comparison(
                        a,
                        c,
                        BranchRelation.CONTINUOUS,
                    ),
                ),
            )


class RealBranchGraphTests(
    unittest.TestCase
):

    def test_three_point_h2_graph_is_continuous(self):
        method = DFTMethodSpec(
            functional="pbe",
            basis="def2-svp",
        )

        with TemporaryDirectory() as tmp:
            roots = []

            for r in (
                0.72,
                0.74,
                0.76,
            ):
                roots.append(
                    run_scf_attempt(
                        DiatomicSpec(
                            label="H2",
                            atom_a="H",
                            atom_b="H",
                            r_angstrom=r,
                            charge=0,
                            spin_2s=0,
                        ),
                        method,
                        "minao",
                        settings=(
                            FAST_SETTINGS
                        ),
                        checkpoint_dir=(
                            Path(tmp)
                        ),
                    )
                )

            graph = build_branch_graph(
                roots,
                thresholds=(
                    TEST_THRESHOLDS
                ),
            )

        self.assertEqual(
            len(
                graph.comparisons
            ),
            2,
        )

        self.assertEqual(
            len(
                graph.continuous_edges
            ),
            2,
        )

        self.assertEqual(
            graph.ambiguous_edges,
            (),
        )

        self.assertEqual(
            len(
                graph.components
            ),
            1,
        )

        self.assertEqual(
            len(
                graph.components[
                    0
                ].member_root_ids
            ),
            3,
        )

        self.assertTrue(
            graph.components[
                0
            ].is_unambiguous
        )

        self.assertTrue(
            graph.is_fully_unambiguous
        )


if __name__ == "__main__":
    unittest.main()
