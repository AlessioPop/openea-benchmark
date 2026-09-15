from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openea_benchmark import (
    DEFAULT_GUESSES,
    DFTMethodSpec,
    DiatomicSpec,
    SCFRunStatus,
    SCFSettings,
    build_molecule,
    run_guess_panel,
    run_scf_attempt,
)


FAST_SETTINGS = SCFSettings(
    grid_level=1,
    num_threads=1,
)


class PySCFBackendContractTests(
    unittest.TestCase
):

    def test_default_guess_panel_is_generic(self):
        self.assertEqual(
            DEFAULT_GUESSES,
            (
                "minao",
                "atom",
                "hcore",
            ),
        )

    def test_duplicate_ecp_assignment_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "duplicate ECP",
        ):
            DFTMethodSpec(
                functional="r2scan",
                basis="def2-tzvpp",
                ecp_assignments=(
                    (
                        "Nb",
                        "def2-tzvpp",
                    ),
                    (
                        "Nb",
                        "def2-qzvppd",
                    ),
                ),
            )

    def test_ecp_assignment_is_element_specific(self):
        spec = DiatomicSpec(
            label="NbC",
            atom_a="Nb",
            atom_b="C",
            r_angstrom=1.70,
            charge=0,
            spin_2s=1,
        )

        method = DFTMethodSpec(
            functional="r2scan",
            basis="def2-tzvpp",
            ecp_assignments=(
                (
                    "Nb",
                    "def2-tzvpp",
                ),
            ),
        )

        mol = build_molecule(
            spec,
            method,
        )

        self.assertIn(
            "Nb",
            mol._ecp,
        )

        self.assertNotIn(
            "C",
            mol._ecp,
        )

    def test_duplicate_guesses_are_rejected(self):
        spec = DiatomicSpec(
            label="H2",
            atom_a="H",
            atom_b="H",
            r_angstrom=0.74,
            charge=0,
            spin_2s=0,
        )

        method = DFTMethodSpec(
            functional="pbe",
            basis="def2-svp",
        )

        with self.assertRaisesRegex(
            ValueError,
            "duplicate guesses",
        ):
            run_guess_panel(
                spec,
                method,
                guesses=(
                    "minao",
                    "minao",
                ),
                settings=FAST_SETTINGS,
            )


class PySCFBackendSmokeTests(
    unittest.TestCase
):

    def test_rks_guess_panel_preserves_attempts(self):
        spec = DiatomicSpec(
            label="H2",
            atom_a="H",
            atom_b="H",
            r_angstrom=0.74,
            charge=0,
            spin_2s=0,
        )

        method = DFTMethodSpec(
            functional="pbe",
            basis="def2-svp",
        )

        with TemporaryDirectory() as tmp:
            roots = run_guess_panel(
                spec,
                method,
                guesses=(
                    "minao",
                    "hcore",
                ),
                settings=FAST_SETTINGS,
                checkpoint_dir=Path(tmp),
            )

        self.assertEqual(
            len(roots),
            2,
        )

        self.assertEqual(
            len(
                {
                    root.root_id
                    for root in roots
                }
            ),
            2,
        )

        for root in roots:
            self.assertEqual(
                root.reference,
                "RKS",
            )

            self.assertEqual(
                root.status,
                SCFRunStatus.CANONICALIZED,
            )

            self.assertTrue(
                root.internal_stable,
            )

            self.assertIsInstance(
                root.external_stable,
                bool,
            )

            self.assertAlmostEqual(
                root.s2,
                0.0,
            )

            self.assertEqual(
                root.observed_multiplicity,
                1.0,
            )

    def test_uks_open_shell_smoke(self):
        spec = DiatomicSpec(
            label="H2plus",
            atom_a="H",
            atom_b="H",
            r_angstrom=1.00,
            charge=1,
            spin_2s=1,
        )

        method = DFTMethodSpec(
            functional="pbe0",
            basis="def2-svp",
        )

        with TemporaryDirectory() as tmp:
            root = run_scf_attempt(
                spec,
                method,
                "minao",
                settings=FAST_SETTINGS,
                checkpoint_dir=Path(tmp),
            )

        self.assertEqual(
            root.reference,
            "UKS",
        )

        self.assertEqual(
            root.status,
            SCFRunStatus.CANONICALIZED,
        )

        self.assertTrue(
            root.internal_stable,
        )

        self.assertIsNone(
            root.external_stable,
        )

        self.assertIsNotNone(
            root.s2,
        )

        self.assertAlmostEqual(
            root.s2,
            0.75,
            places=6,
        )

        self.assertIsNotNone(
            root.energy_hartree,
        )


if __name__ == "__main__":
    unittest.main()
