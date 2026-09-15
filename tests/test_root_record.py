import math
import unittest

from openea_benchmark import (
    SCFRootRecord,
    SCFRunStatus,
)


def make_record(**overrides):
    values = dict(
        root_id="FeH_N_S3_minao",
        molecule="FeH",
        charge=0,
        spin_2s=3,
        r_angstrom=1.5674,
        functional="r2scan",
        basis="ma-def2-tzvpp",
        origin_guess="minao",
        scf_path="standard",
        reference="UKS",
        status=SCFRunStatus.CANONICALIZED,
        energy_hartree=-1264.0,
        internal_stable=True,
        external_stable=None,
        stability_shift_ev=0.0,
        s2=3.75,
        observed_multiplicity=4.0,
        checkpoint_path="example.chk",
    )

    values.update(overrides)
    return SCFRootRecord(**values)


class RootRecordTests(unittest.TestCase):

    def test_nominal_spin_quantities_use_2s_convention(self):
        root = make_record(
            spin_2s=3,
            s2=3.75,
        )

        self.assertEqual(
            root.nominal_spin_s,
            1.5,
        )
        self.assertEqual(
            root.nominal_multiplicity,
            4,
        )
        self.assertAlmostEqual(
            root.expected_s2,
            3.75,
        )
        self.assertAlmostEqual(
            root.spin_contamination,
            0.0,
        )

    def test_spin_contamination_is_recorded_not_rejected(self):
        root = make_record(
            root_id="NbC_N_S1",
            molecule="NbC",
            spin_2s=1,
            s2=1.433979,
            observed_multiplicity=2.595,
        )

        self.assertEqual(
            root.status,
            SCFRunStatus.CANONICALIZED,
        )
        self.assertTrue(
            root.internal_stable,
        )
        self.assertAlmostEqual(
            root.expected_s2,
            0.75,
        )
        self.assertAlmostEqual(
            root.spin_contamination,
            0.683979,
        )

    def test_external_instability_is_independent(self):
        root = make_record(
            root_id="NbC_A_S0_RKS",
            molecule="NbC",
            spin_2s=0,
            s2=0.0,
            internal_stable=True,
            external_stable=False,
        )

        self.assertTrue(
            root.internal_stable,
        )
        self.assertFalse(
            root.external_stable,
        )

    def test_canonicalized_requires_internal_stability(self):
        with self.assertRaisesRegex(
            ValueError,
            "internal_stable",
        ):
            make_record(
                internal_stable=False,
            )

    def test_converged_requires_finite_energy(self):
        with self.assertRaisesRegex(
            ValueError,
            "finite energy",
        ):
            make_record(
                status=SCFRunStatus.CONVERGED,
                internal_stable=None,
                energy_hartree=None,
            )

        with self.assertRaisesRegex(
            ValueError,
            "finite energy",
        ):
            make_record(
                status=SCFRunStatus.CONVERGED,
                internal_stable=None,
                energy_hartree=math.nan,
            )

    def test_failed_attempt_can_preserve_diagnostic(self):
        root = make_record(
            root_id="failed_root",
            status=SCFRunStatus.FAILED,
            energy_hartree=None,
            internal_stable=None,
            stability_shift_ev=None,
            s2=None,
            observed_multiplicity=None,
            diagnostic_message="SCF did not converge",
        )

        self.assertEqual(
            root.status,
            SCFRunStatus.FAILED,
        )
        self.assertIsNone(
            root.energy_hartree,
        )
        self.assertIn(
            "did not converge",
            root.diagnostic_message,
        )

    def test_reference_and_ecp_provenance_are_preserved(self):
        root = make_record(
            reference="UKS",
            ecp_assignments=(
                ("Nb", "def2-tzvpp"),
            ),
        )

        self.assertEqual(
            root.reference,
            "UKS",
        )

        self.assertEqual(
            root.ecp_assignments,
            (
                ("Nb", "def2-tzvpp"),
            ),
        )

    def test_geometry_must_be_positive_and_finite(self):
        for bad_r in (
            0.0,
            -1.0,
            math.nan,
            math.inf,
        ):
            with self.subTest(
                r_angstrom=bad_r
            ):
                with self.assertRaises(
                    ValueError
                ):
                    make_record(
                        r_angstrom=bad_r
                    )


if __name__ == "__main__":
    unittest.main()
