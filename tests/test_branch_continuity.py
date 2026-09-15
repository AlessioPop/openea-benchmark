from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openea_benchmark import (
    BranchRelation,
    BranchThresholds,
    DFTMethodSpec,
    DiatomicSpec,
    SCFSettings,
    classify_branch_metrics,
    compare_branch_roots,
    run_scf_attempt,
)


#
# Synthetic test policy only.
# These are not frozen OpenEA production thresholds.
#
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


class BranchPolicyTests(
    unittest.TestCase
):

    def test_no_production_defaults_exist(self):
        with self.assertRaises(
            TypeError
        ):
            BranchThresholds()

    def test_continuous_metrics(self):
        relation = (
            classify_branch_metrics(
                delta_r_angstrom=0.02,
                delta_s2=0.002,
                alpha_occ_overlap_min=0.99,
                beta_occ_overlap_min=0.98,
                thresholds=(
                    TEST_THRESHOLDS
                ),
            )
        )

        self.assertEqual(
            relation,
            BranchRelation.CONTINUOUS,
        )

    def test_clear_subspace_loss_is_discontinuous(self):
        relation = (
            classify_branch_metrics(
                delta_r_angstrom=0.02,
                delta_s2=0.002,
                alpha_occ_overlap_min=0.20,
                beta_occ_overlap_min=0.99,
                thresholds=(
                    TEST_THRESHOLDS
                ),
            )
        )

        self.assertEqual(
            relation,
            BranchRelation.DISCONTINUOUS,
        )

    def test_intermediate_evidence_is_ambiguous(self):
        relation = (
            classify_branch_metrics(
                delta_r_angstrom=0.02,
                delta_s2=0.05,
                alpha_occ_overlap_min=0.80,
                beta_occ_overlap_min=0.85,
                thresholds=(
                    TEST_THRESHOLDS
                ),
            )
        )

        self.assertEqual(
            relation,
            BranchRelation.AMBIGUOUS,
        )

    def test_large_geometry_step_is_ambiguous(self):
        relation = (
            classify_branch_metrics(
                delta_r_angstrom=0.20,
                delta_s2=0.0,
                alpha_occ_overlap_min=1.0,
                beta_occ_overlap_min=1.0,
                thresholds=(
                    TEST_THRESHOLDS
                ),
            )
        )

        self.assertEqual(
            relation,
            BranchRelation.AMBIGUOUS,
        )


class RealCheckpointBranchTests(
    unittest.TestCase
):

    def test_h2_rks_neighbor_is_continuous(self):
        method = DFTMethodSpec(
            functional="pbe",
            basis="def2-svp",
        )

        with TemporaryDirectory() as tmp:
            root_a = run_scf_attempt(
                DiatomicSpec(
                    label="H2",
                    atom_a="H",
                    atom_b="H",
                    r_angstrom=0.74,
                    charge=0,
                    spin_2s=0,
                ),
                method,
                "minao",
                settings=FAST_SETTINGS,
                checkpoint_dir=Path(tmp),
            )

            root_b = run_scf_attempt(
                DiatomicSpec(
                    label="H2",
                    atom_a="H",
                    atom_b="H",
                    r_angstrom=0.76,
                    charge=0,
                    spin_2s=0,
                ),
                method,
                "minao",
                settings=FAST_SETTINGS,
                checkpoint_dir=Path(tmp),
            )

            comparison = (
                compare_branch_roots(
                    root_a,
                    root_b,
                    thresholds=(
                        TEST_THRESHOLDS
                    ),
                )
            )

        self.assertEqual(
            comparison.relation,
            BranchRelation.CONTINUOUS,
        )

        self.assertGreater(
            comparison.alpha_occ_overlap_min,
            0.95,
        )

        self.assertGreater(
            comparison.beta_occ_overlap_min,
            0.95,
        )

        self.assertAlmostEqual(
            comparison.delta_r_angstrom,
            0.02,
            places=12,
        )


    def test_h2plus_uks_neighbor_is_continuous(self):
        method = DFTMethodSpec(
            functional="pbe0",
            basis="def2-svp",
        )

        with TemporaryDirectory() as tmp:
            root_a = run_scf_attempt(
                DiatomicSpec(
                    label="H2plus",
                    atom_a="H",
                    atom_b="H",
                    r_angstrom=1.00,
                    charge=1,
                    spin_2s=1,
                ),
                method,
                "minao",
                settings=FAST_SETTINGS,
                checkpoint_dir=Path(tmp),
            )

            root_b = run_scf_attempt(
                DiatomicSpec(
                    label="H2plus",
                    atom_a="H",
                    atom_b="H",
                    r_angstrom=1.02,
                    charge=1,
                    spin_2s=1,
                ),
                method,
                "minao",
                settings=FAST_SETTINGS,
                checkpoint_dir=Path(tmp),
            )

            comparison = (
                compare_branch_roots(
                    root_a,
                    root_b,
                    thresholds=(
                        TEST_THRESHOLDS
                    ),
                )
            )

        self.assertEqual(
            comparison.relation,
            BranchRelation.CONTINUOUS,
        )

        self.assertGreater(
            comparison.alpha_occ_overlap_min,
            0.95,
        )

        #
        # H2+ has no occupied beta orbital. The empty beta channel is
        # intentionally treated as trivially continuous.
        #
        self.assertEqual(
            comparison.beta_singular_values,
            (),
        )

        self.assertEqual(
            comparison.beta_occ_overlap_min,
            1.0,
        )


    def test_different_spin_sector_is_rejected(self):
        method = DFTMethodSpec(
            functional="pbe",
            basis="def2-svp",
        )

        with TemporaryDirectory() as tmp:
            singlet = run_scf_attempt(
                DiatomicSpec(
                    label="H2",
                    atom_a="H",
                    atom_b="H",
                    r_angstrom=0.74,
                    charge=0,
                    spin_2s=0,
                ),
                method,
                "minao",
                settings=FAST_SETTINGS,
                checkpoint_dir=Path(tmp),
            )

            triplet = run_scf_attempt(
                DiatomicSpec(
                    label="H2",
                    atom_a="H",
                    atom_b="H",
                    r_angstrom=0.76,
                    charge=0,
                    spin_2s=2,
                ),
                method,
                "minao",
                settings=FAST_SETTINGS,
                checkpoint_dir=Path(tmp),
            )

            with self.assertRaisesRegex(
                ValueError,
                "spin_2s",
            ):
                compare_branch_roots(
                    singlet,
                    triplet,
                    thresholds=(
                        TEST_THRESHOLDS
                    ),
                )


if __name__ == "__main__":
    unittest.main()
