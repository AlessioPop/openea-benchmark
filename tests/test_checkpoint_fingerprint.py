from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openea_benchmark import (
    CheckpointAuditSettings,
    DFTMethodSpec,
    DiatomicSpec,
    IdentityThresholds,
    SCFRunStatus,
    SCFSettings,
    deduplicate_checkpoint_roots,
    fingerprint_from_checkpoint,
    fingerprints_from_checkpoints,
    run_guess_panel,
    run_scf_attempt,
)


FAST_SETTINGS = SCFSettings(
    grid_level=1,
    num_threads=1,
)


#
# Synthetic test thresholds only.
# These are not production OpenEA state-identity policy.
#
TEST_THRESHOLDS = IdentityThresholds(
    same_energy_mev=0.5,
    same_delta_s2=1.0e-5,
    same_total_spectrum_max=1.0e-5,
    same_spin_spectrum_max=1.0e-5,
    same_total_density_rel_fro=1.0e-5,
    same_spin_density_rel_fro=1.0e-5,

    distinct_energy_mev=10.0,
    distinct_delta_s2=0.10,
    distinct_total_spectrum_max=0.10,
    distinct_spin_spectrum_max=0.10,
    distinct_total_density_rel_fro=0.10,
    distinct_spin_density_rel_fro=0.10,
)


class CheckpointFingerprintTests(
    unittest.TestCase
):

    def test_rks_checkpoint_reconstructs_electron_and_spin_traces(self):
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
            root = run_scf_attempt(
                spec,
                method,
                "minao",
                settings=FAST_SETTINGS,
                checkpoint_dir=Path(tmp),
            )

            self.assertEqual(
                root.status,
                SCFRunStatus.CANONICALIZED,
            )

            self.assertIsNotNone(
                root.checkpoint_path,
            )

            fingerprint, audit = (
                fingerprint_from_checkpoint(
                    root
                )
            )

        self.assertAlmostEqual(
            fingerprint.total_trace,
            2.0,
            places=7,
        )

        self.assertAlmostEqual(
            fingerprint.spin_trace,
            0.0,
            places=7,
        )

        self.assertAlmostEqual(
            audit.energy_delta_hartree,
            0.0,
            places=10,
        )

        self.assertGreater(
            audit.overlap_min_eigenvalue,
            0.0,
        )


    def test_uks_checkpoint_reconstructs_spin_trace(self):
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

            fingerprint, audit = (
                fingerprint_from_checkpoint(
                    root
                )
            )

        self.assertAlmostEqual(
            fingerprint.total_trace,
            1.0,
            places=7,
        )

        self.assertAlmostEqual(
            fingerprint.spin_trace,
            1.0,
            places=7,
        )

        self.assertEqual(
            audit.expected_spin_2s,
            1,
        )


    def test_stale_checkpoint_energy_is_rejected(self):
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
            root = run_scf_attempt(
                spec,
                method,
                "minao",
                settings=FAST_SETTINGS,
                checkpoint_dir=Path(tmp),
            )

            bad = replace(
                root,
                energy_hartree=(
                    root.energy_hartree
                    + 1.0e-4
                ),
            )

            with self.assertRaisesRegex(
                ValueError,
                "energy does not match",
            ):
                fingerprint_from_checkpoint(
                    bad
                )


    def test_batch_preserves_unusable_root_as_failure(self):
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
            good = run_scf_attempt(
                spec,
                method,
                "minao",
                settings=FAST_SETTINGS,
                checkpoint_dir=Path(tmp),
            )

            incomplete = replace(
                good,
                root_id="not_canonicalized",
                status=SCFRunStatus.CONVERGED,
                internal_stable=False,
            )

            (
                fingerprints,
                audits,
                failures,
            ) = fingerprints_from_checkpoints(
                (
                    good,
                    incomplete,
                )
            )

        self.assertIn(
            good.root_id,
            fingerprints,
        )

        self.assertEqual(
            len(audits),
            1,
        )

        self.assertEqual(
            tuple(
                failure.root_id
                for failure in failures
            ),
            (
                "not_canonicalized",
            ),
        )


    def test_end_to_end_guess_panel_deduplicates_identical_h2_root(self):
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

            self.assertTrue(
                all(
                    root.status
                    == SCFRunStatus.CANONICALIZED
                    for root in roots
                )
            )

            result = (
                deduplicate_checkpoint_roots(
                    roots,
                    thresholds=(
                        TEST_THRESHOLDS
                    ),
                )
            )

        self.assertEqual(
            result.fingerprint_failures,
            (),
        )

        self.assertEqual(
            len(result.audits),
            2,
        )

        self.assertEqual(
            len(
                result.deduplication.clusters
            ),
            1,
        )

        self.assertEqual(
            set(
                result.deduplication
                .clusters[0]
                .member_root_ids
            ),
            {
                roots[0].root_id,
                roots[1].root_id,
            },
        )

        self.assertEqual(
            result.deduplication
            .ambiguous_components,
            (),
        )


    def test_numerical_audit_thresholds_are_explicitly_separate(self):
        settings = (
            CheckpointAuditSettings()
        )

        self.assertGreater(
            settings.energy_tol_hartree,
            0.0,
        )

        self.assertGreater(
            settings.electron_trace_tol,
            0.0,
        )

        #
        # These are checkpoint-integrity tolerances only. The scientific
        # identity thresholds remain separately required by the deduplication
        # API.
        #
        self.assertGreater(
            settings.spin_trace_tol,
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
