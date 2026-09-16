import unittest

from openea_benchmark import (
    BranchComparison,
    BranchRelation,
    SCFRootRecord,
    SCFRunStatus,
    build_branch_graph_from_comparisons,
)

from openea_benchmark.local_pec import (
    LocalPECRejectionReason,
    construct_local_pecs,
)


def make_root(
    root_id: str,
    r: float,
    *,
    energy: float,
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
            b.r_angstrom - a.r_angstrom
        ),
        delta_energy_mev=0.0,
        delta_s2=0.0,
        alpha_singular_values=(0.99,),
        beta_singular_values=(0.99,),
        alpha_occ_overlap_min=0.99,
        alpha_occ_overlap_mean=0.99,
        beta_occ_overlap_min=0.99,
        beta_occ_overlap_mean=0.99,
        relation=relation,
    )


class LocalPECPolicyTests(unittest.TestCase):

    def test_simple_continuous_chain_becomes_local_pec(self):
        a = make_root(
            "a",
            1.00,
            energy=-10.00,
        )
        b = make_root(
            "b",
            1.05,
            energy=-10.10,
        )
        c = make_root(
            "c",
            1.10,
            energy=-10.05,
        )

        roots = (a, b, c)

        graph = build_branch_graph_from_comparisons(
            roots,
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

        result = construct_local_pecs(
            roots,
            graph,
        )

        self.assertEqual(
            len(result.pecs),
            1,
        )
        self.assertEqual(
            result.rejected_components,
            (),
        )

        pec = result.pecs[0]

        self.assertEqual(
            pec.component_id,
            "branch_component_0001",
        )
        self.assertEqual(
            tuple(
                point.root_id
                for point in pec.points
            ),
            ("a", "b", "c"),
        )
        self.assertEqual(
            tuple(
                point.r_angstrom
                for point in pec.points
            ),
            (1.00, 1.05, 1.10),
        )
        self.assertEqual(
            tuple(
                point.energy_hartree
                for point in pec.points
            ),
            (-10.00, -10.10, -10.05),
        )

    def test_energy_order_does_not_reorder_pec_points(self):
        a = make_root(
            "a",
            1.00,
            energy=-10.00,
        )
        b = make_root(
            "b",
            1.05,
            energy=-9.00,
        )
        c = make_root(
            "c",
            1.10,
            energy=-11.00,
        )

        roots = (a, b, c)

        graph = build_branch_graph_from_comparisons(
            roots,
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

        result = construct_local_pecs(
            roots,
            graph,
        )

        pec = result.pecs[0]

        self.assertEqual(
            tuple(
                point.root_id
                for point in pec.points
            ),
            ("a", "b", "c"),
        )

    def test_branching_component_is_not_promoted_to_pec(self):
        a = make_root(
            "a",
            1.00,
            energy=-10.00,
        )
        b = make_root(
            "b",
            1.05,
            energy=-10.10,
        )
        c = make_root(
            "c",
            1.05,
            energy=-10.08,
        )
        d = make_root(
            "d",
            1.10,
            energy=-10.03,
        )

        roots = (a, b, c, d)

        graph = build_branch_graph_from_comparisons(
            roots,
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

        result = construct_local_pecs(
            roots,
            graph,
        )

        self.assertEqual(
            result.pecs,
            (),
        )
        self.assertEqual(
            len(result.rejected_components),
            1,
        )
        self.assertIn(
            LocalPECRejectionReason.BRANCHING_TOPOLOGY,
            result.rejected_components[0].reasons,
        )

    def test_ambiguous_boundary_is_not_promoted_to_pec(self):
        a = make_root(
            "a",
            1.00,
            energy=-10.00,
        )
        b = make_root(
            "b",
            1.05,
            energy=-10.10,
        )
        c = make_root(
            "c",
            1.05,
            energy=-10.08,
        )

        roots = (a, b, c)

        graph = build_branch_graph_from_comparisons(
            roots,
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

        result = construct_local_pecs(
            roots,
            graph,
        )

        self.assertEqual(
            result.pecs,
            (),
        )

        reasons = {
            reason
            for rejected
            in result.rejected_components
            for reason
            in rejected.reasons
        }

        self.assertIn(
            LocalPECRejectionReason.AMBIGUOUS_BOUNDARY,
            reasons,
        )

    def test_discontinuity_can_create_two_local_pec_segments(self):
        a = make_root(
            "a",
            1.00,
            energy=-10.00,
        )
        b = make_root(
            "b",
            1.05,
            energy=-10.10,
        )
        c = make_root(
            "c",
            1.10,
            energy=-9.95,
        )
        d = make_root(
            "d",
            1.15,
            energy=-10.02,
        )

        roots = (a, b, c, d)

        graph = build_branch_graph_from_comparisons(
            roots,
            (
                make_comparison(
                    a,
                    b,
                    BranchRelation.CONTINUOUS,
                ),
                make_comparison(
                    b,
                    c,
                    BranchRelation.DISCONTINUOUS,
                ),
                make_comparison(
                    c,
                    d,
                    BranchRelation.CONTINUOUS,
                ),
            ),
        )

        result = construct_local_pecs(
            roots,
            graph,
        )

        self.assertEqual(
            len(result.pecs),
            2,
        )

        self.assertEqual(
            tuple(
                tuple(
                    point.root_id
                    for point in pec.points
                )
                for pec in result.pecs
            ),
            (
                ("a", "b"),
                ("c", "d"),
            ),
        )

    def test_singleton_component_is_reported_not_called_a_pec(self):
        a = make_root(
            "a",
            1.00,
            energy=-10.00,
        )
        b = make_root(
            "b",
            1.05,
            energy=-9.90,
        )

        roots = (a, b)

        graph = build_branch_graph_from_comparisons(
            roots,
            (
                make_comparison(
                    a,
                    b,
                    BranchRelation.DISCONTINUOUS,
                ),
            ),
        )

        result = construct_local_pecs(
            roots,
            graph,
        )

        self.assertEqual(
            result.pecs,
            (),
        )
        self.assertEqual(
            len(result.rejected_components),
            2,
        )

        for rejected in result.rejected_components:
            self.assertIn(
                LocalPECRejectionReason.INSUFFICIENT_POINTS,
                rejected.reasons,
            )

    def test_root_set_must_match_branch_graph_exactly(self):
        a = make_root(
            "a",
            1.00,
            energy=-10.00,
        )
        b = make_root(
            "b",
            1.05,
            energy=-10.10,
        )

        roots = (a, b)

        graph = build_branch_graph_from_comparisons(
            roots,
            (
                make_comparison(
                    a,
                    b,
                    BranchRelation.CONTINUOUS,
                ),
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "root set",
        ):
            construct_local_pecs(
                (a,),
                graph,
            )


if __name__ == "__main__":
    unittest.main()
