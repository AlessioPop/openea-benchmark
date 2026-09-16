import inspect
import unittest

from openea_benchmark import (
    LocalPEC,
    LocalPECPoint,
)

from openea_benchmark.minimum_scout import (
    MinimumScoutStatus,
    MinimumScoutThresholds,
    scout_local_pec_minimum,
)


ENERGY_TOL = 1.0e-5


def make_pec(
    energies,
    *,
    r0=1.0,
    step=0.1,
) -> LocalPEC:
    points = []

    for index, energy in enumerate(
        energies
    ):
        points.append(
            LocalPECPoint(
                root_id=(
                    f"root_{index}"
                ),
                r_angstrom=(
                    r0
                    + index * step
                ),
                energy_hartree=float(
                    energy
                ),
            )
        )

    return LocalPEC(
        component_id="component_0",
        points=tuple(
            points
        ),
    )


def thresholds():
    return MinimumScoutThresholds(
        energy_tolerance_hartree=(
            ENERGY_TOL
        ),
    )


class MinimumScoutThresholdTests(
    unittest.TestCase
):

    def test_energy_tolerance_has_no_default(self):
        signature = inspect.signature(
            MinimumScoutThresholds
        )

        parameter = (
            signature.parameters[
                "energy_tolerance_hartree"
            ]
        )

        self.assertIs(
            parameter.default,
            inspect.Parameter.empty,
        )

    def test_energy_tolerance_must_be_positive(self):
        with self.assertRaises(
            ValueError
        ):
            MinimumScoutThresholds(
                energy_tolerance_hartree=0.0,
            )

        with self.assertRaises(
            ValueError
        ):
            MinimumScoutThresholds(
                energy_tolerance_hartree=-1.0e-6,
            )


class MinimumScoutPolicyTests(
    unittest.TestCase
):

    def test_two_points_can_only_indicate_decrease_toward_high_r(self):
        pec = make_pec(
            (
                -1.00,
                -1.01,
            )
        )

        result = scout_local_pec_minimum(
            pec,
            thresholds=thresholds(),
        )

        self.assertEqual(
            result.status,
            MinimumScoutStatus.DECREASES_TOWARD_HIGHER_R,
        )

        self.assertEqual(
            result.candidates,
            (),
        )

    def test_two_points_can_only_indicate_decrease_toward_low_r(self):
        pec = make_pec(
            (
                -1.01,
                -1.00,
            )
        )

        result = scout_local_pec_minimum(
            pec,
            thresholds=thresholds(),
        )

        self.assertEqual(
            result.status,
            MinimumScoutStatus.DECREASES_TOWARD_LOWER_R,
        )

        self.assertEqual(
            result.candidates,
            (),
        )

    def test_endpoint_lowest_point_is_not_called_a_minimum(self):
        pec = make_pec(
            (
                -1.00,
                -1.01,
                -1.02,
            )
        )

        result = scout_local_pec_minimum(
            pec,
            thresholds=thresholds(),
        )

        self.assertEqual(
            result.status,
            MinimumScoutStatus.DECREASES_TOWARD_HIGHER_R,
        )

        self.assertEqual(
            result.candidates,
            (),
        )

    def test_clean_three_point_bowl_has_one_bracketed_candidate(self):
        pec = make_pec(
            (
                -1.00,
                -1.02,
                -1.00,
            )
        )

        result = scout_local_pec_minimum(
            pec,
            thresholds=thresholds(),
        )

        self.assertEqual(
            result.status,
            MinimumScoutStatus.BRACKETED_SINGLE_MINIMUM,
        )

        self.assertEqual(
            len(
                result.candidates
            ),
            1,
        )

        candidate = (
            result.candidates[0]
        )

        self.assertEqual(
            candidate.point_index,
            1,
        )

        self.assertEqual(
            candidate.root_id,
            "root_1",
        )

        self.assertEqual(
            candidate.left_root_id,
            "root_0",
        )

        self.assertEqual(
            candidate.right_root_id,
            "root_2",
        )

        self.assertAlmostEqual(
            candidate.r_angstrom,
            1.1,
        )

        self.assertAlmostEqual(
            candidate.left_r_angstrom,
            1.0,
        )

        self.assertAlmostEqual(
            candidate.right_r_angstrom,
            1.2,
        )

    def test_shallow_apparent_minimum_within_tolerance_is_unresolved(self):
        pec = make_pec(
            (
                -1.000000,
                -1.000005,
                -1.000000,
            )
        )

        result = scout_local_pec_minimum(
            pec,
            thresholds=thresholds(),
        )

        self.assertEqual(
            result.status,
            MinimumScoutStatus.FLAT_OR_UNRESOLVED,
        )

        self.assertEqual(
            result.candidates,
            (),
        )

    def test_flat_two_point_curve_is_unresolved(self):
        pec = make_pec(
            (
                -1.000000,
                -1.000005,
            )
        )

        result = scout_local_pec_minimum(
            pec,
            thresholds=thresholds(),
        )

        self.assertEqual(
            result.status,
            MinimumScoutStatus.FLAT_OR_UNRESOLVED,
        )

    def test_multiple_minimum_candidates_are_all_retained(self):
        pec = make_pec(
            (
                -1.00,
                -1.02,
                -1.00,
                -1.03,
                -1.00,
            )
        )

        result = scout_local_pec_minimum(
            pec,
            thresholds=thresholds(),
        )

        self.assertEqual(
            result.status,
            MinimumScoutStatus.MULTIPLE_MINIMUM_CANDIDATES,
        )

        self.assertEqual(
            tuple(
                candidate.root_id
                for candidate
                in result.candidates
            ),
            (
                "root_1",
                "root_3",
            ),
        )

    def test_multiple_candidates_are_not_energy_ranked(self):
        pec = make_pec(
            (
                -1.00,
                -1.01,
                -1.00,
                -1.05,
                -1.00,
            )
        )

        result = scout_local_pec_minimum(
            pec,
            thresholds=thresholds(),
        )

        self.assertEqual(
            tuple(
                candidate.root_id
                for candidate
                in result.candidates
            ),
            (
                "root_1",
                "root_3",
            ),
        )

    def test_local_maximum_without_minimum_is_complex(self):
        pec = make_pec(
            (
                -1.02,
                -1.00,
                -1.01,
            )
        )

        result = scout_local_pec_minimum(
            pec,
            thresholds=thresholds(),
        )

        self.assertEqual(
            result.status,
            MinimumScoutStatus.COMPLEX_SHAPE,
        )

        self.assertEqual(
            result.candidates,
            (),
        )

    def test_clean_wider_bowl_is_bracketed_single_minimum(self):
        pec = make_pec(
            (
                -1.00,
                -1.01,
                -1.03,
                -1.01,
                -1.00,
            )
        )

        result = scout_local_pec_minimum(
            pec,
            thresholds=thresholds(),
        )

        self.assertEqual(
            result.status,
            MinimumScoutStatus.BRACKETED_SINGLE_MINIMUM,
        )

        self.assertEqual(
            tuple(
                candidate.point_index
                for candidate
                in result.candidates
            ),
            (
                2,
            ),
        )

    def test_energy_steps_are_reported_in_geometry_order(self):
        pec = make_pec(
            (
                -1.00,
                -1.02,
                -1.01,
            )
        )

        result = scout_local_pec_minimum(
            pec,
            thresholds=thresholds(),
        )

        self.assertEqual(
            len(
                result.adjacent_delta_e_hartree
            ),
            2,
        )

        self.assertAlmostEqual(
            result.adjacent_delta_e_hartree[0],
            -0.02,
        )

        self.assertAlmostEqual(
            result.adjacent_delta_e_hartree[1],
            0.01,
        )


class MinimumScoutResolutionPriorityTests(
    unittest.TestCase
):

    def test_unresolved_step_takes_priority_over_single_candidate(self):
        pec = make_pec(
            (
                -1.000000,
                -1.020000,
                -1.000000,
                -1.000005,
            )
        )

        result = scout_local_pec_minimum(
            pec,
            thresholds=thresholds(),
        )

        self.assertEqual(
            result.status,
            MinimumScoutStatus.FLAT_OR_UNRESOLVED,
        )

        self.assertEqual(
            tuple(
                candidate.root_id
                for candidate
                in result.candidates
            ),
            (
                "root_1",
            ),
        )

    def test_unresolved_step_takes_priority_over_multiple_candidates(self):
        pec = make_pec(
            (
                -1.000000,
                -1.020000,
                -1.000000,
                -1.030000,
                -1.000000,
                -1.000005,
            )
        )

        result = scout_local_pec_minimum(
            pec,
            thresholds=thresholds(),
        )

        self.assertEqual(
            result.status,
            MinimumScoutStatus.FLAT_OR_UNRESOLVED,
        )

        self.assertEqual(
            tuple(
                candidate.root_id
                for candidate
                in result.candidates
            ),
            (
                "root_1",
                "root_3",
            ),
        )



if __name__ == "__main__":
    unittest.main()
