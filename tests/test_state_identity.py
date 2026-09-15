import unittest

import numpy as np

from openea_benchmark import (
    IdentityThresholds,
    SCFRootRecord,
    SCFRunStatus,
    StateFingerprint,
    StateRelation,
    classify_metrics,
    compare_states,
    deduplicate_roots,
    fingerprint_from_orthonormal_density,
)


HARTREE_TO_EV = 27.211386245988


#
# Synthetic unit-test values only.
#
# These are deliberately local to the tests and are not OpenEA production
# thresholds.
#
TEST_THRESHOLDS = IdentityThresholds(
    same_energy_mev=1.0,
    same_delta_s2=0.01,
    same_total_spectrum_max=0.01,
    same_spin_spectrum_max=0.01,
    same_total_density_rel_fro=0.01,
    same_spin_density_rel_fro=0.01,

    distinct_energy_mev=10.0,
    distinct_delta_s2=0.10,
    distinct_total_spectrum_max=0.10,
    distinct_spin_spectrum_max=0.10,
    distinct_total_density_rel_fro=0.10,
    distinct_spin_density_rel_fro=0.10,
)


def energy_at_mev(
    mev: float,
) -> float:
    return (
        -100.0
        + mev
        / (
            HARTREE_TO_EV
            * 1000.0
        )
    )


def make_root(
    root_id: str,
    *,
    energy_mev: float = 0.0,
    spin_2s: int = 1,
    s2: float = 0.75,
    status: SCFRunStatus = (
        SCFRunStatus.CANONICALIZED
    ),
    molecule: str = "XY",
    charge: int = 0,
    r_angstrom: float = 1.5,
    functional: str = "r2scan",
    basis: str = "ma-def2-tzvpp",
) -> SCFRootRecord:
    return SCFRootRecord(
        root_id=root_id,
        molecule=molecule,
        charge=charge,
        spin_2s=spin_2s,
        r_angstrom=r_angstrom,
        functional=functional,
        basis=basis,
        origin_guess=root_id,
        scf_path="standard",
        reference=(
            "RKS"
            if spin_2s == 0
            else "UKS"
        ),
        status=status,
        energy_hartree=(
            None
            if status
            == SCFRunStatus.FAILED
            else energy_at_mev(
                energy_mev
            )
        ),
        internal_stable=(
            True
            if status
            == SCFRunStatus.CANONICALIZED
            else None
        ),
        s2=(
            None
            if status
            == SCFRunStatus.FAILED
            else s2
        ),
    )


def make_fingerprint(
    shift: float = 0.0,
) -> StateFingerprint:
    return StateFingerprint(
        total_spectrum=(
            1.8 + shift,
            0.2,
            0.0,
        ),
        spin_spectrum=(
            0.8 + shift,
            0.0,
            -0.8,
        ),
        total_density_orth=(
            (
                1.8 + shift,
                0.0,
                0.0,
            ),
            (
                0.0,
                0.2,
                0.0,
            ),
            (
                0.0,
                0.0,
                0.0,
            ),
        ),
        spin_density_orth=(
            (
                0.8 + shift,
                0.0,
                0.0,
            ),
            (
                0.0,
                0.0,
                0.0,
            ),
            (
                0.0,
                0.0,
                -0.8,
            ),
        ),
        total_trace=2.0 + shift,
        spin_trace=shift,
    )


class IdentityThresholdTests(
    unittest.TestCase
):

    def test_no_implicit_production_defaults_exist(self):
        with self.assertRaises(
            TypeError
        ):
            IdentityThresholds()

    def test_same_threshold_must_be_below_distinct_threshold(self):
        with self.assertRaisesRegex(
            ValueError,
            "strictly below",
        ):
            IdentityThresholds(
                same_energy_mev=10.0,
                same_delta_s2=0.01,
                same_total_spectrum_max=0.01,
                same_spin_spectrum_max=0.01,
                same_total_density_rel_fro=0.01,
                same_spin_density_rel_fro=0.01,
                distinct_energy_mev=10.0,
                distinct_delta_s2=0.10,
                distinct_total_spectrum_max=0.10,
                distinct_spin_spectrum_max=0.10,
                distinct_total_density_rel_fro=0.10,
                distinct_spin_density_rel_fro=0.10,
            )


class DensityFingerprintTests(
    unittest.TestCase
):

    def test_density_spectra_are_rotation_invariant(self):
        dm = np.zeros(
            (
                2,
                3,
                3,
            )
        )

        dm[0] = np.diag(
            (
                1.0,
                0.4,
                0.0,
            )
        )

        dm[1] = np.diag(
            (
                0.8,
                0.1,
                0.0,
            )
        )

        angle = 0.37

        q = np.array(
            [
                [
                    np.cos(angle),
                    -np.sin(angle),
                    0.0,
                ],
                [
                    np.sin(angle),
                    np.cos(angle),
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                ],
            ]
        )

        rotated = np.stack(
            (
                q @ dm[0] @ q.T,
                q @ dm[1] @ q.T,
            )
        )

        a = (
            fingerprint_from_orthonormal_density(
                dm
            )
        )

        b = (
            fingerprint_from_orthonormal_density(
                rotated
            )
        )

        np.testing.assert_allclose(
            a.total_spectrum,
            b.total_spectrum,
            atol=1.0e-12,
        )

        np.testing.assert_allclose(
            a.spin_spectrum,
            b.spin_spectrum,
            atol=1.0e-12,
        )

    def test_invalid_density_shape_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "shape",
        ):
            fingerprint_from_orthonormal_density(
                np.eye(3)
            )


class PairwiseIdentityTests(
    unittest.TestCase
):

    def test_all_same_metrics_give_same_state(self):
        relation = classify_metrics(
            same_spin=True,
            delta_energy_mev=0.5,
            delta_s2=0.002,
            total_spectrum_max=0.003,
            spin_spectrum_max=0.004,
            total_density_rel_fro=0.004,
            spin_density_rel_fro=0.004,
            thresholds=TEST_THRESHOLDS,
        )

        self.assertEqual(
            relation,
            StateRelation.SAME_STATE,
        )

    def test_different_spin_is_always_distinct(self):
        relation = classify_metrics(
            same_spin=False,
            delta_energy_mev=0.0,
            delta_s2=0.0,
            total_spectrum_max=0.0,
            spin_spectrum_max=0.0,
            total_density_rel_fro=0.0,
            spin_density_rel_fro=0.0,
            thresholds=TEST_THRESHOLDS,
        )

        self.assertEqual(
            relation,
            StateRelation.DISTINCT_STATE,
        )

    def test_energy_separation_alone_is_not_enough(self):
        relation = classify_metrics(
            same_spin=True,
            delta_energy_mev=20.0,
            delta_s2=0.0,
            total_spectrum_max=0.0,
            spin_spectrum_max=0.0,
            total_density_rel_fro=0.0,
            spin_density_rel_fro=0.0,
            thresholds=TEST_THRESHOLDS,
        )

        self.assertEqual(
            relation,
            StateRelation.AMBIGUOUS,
        )

    def test_energy_plus_density_difference_is_distinct(self):
        relation = classify_metrics(
            same_spin=True,
            delta_energy_mev=20.0,
            delta_s2=0.0,
            total_spectrum_max=0.20,
            spin_spectrum_max=0.0,
            total_density_rel_fro=0.0,
            spin_density_rel_fro=0.0,
            thresholds=TEST_THRESHOLDS,
        )

        self.assertEqual(
            relation,
            StateRelation.DISTINCT_STATE,
        )

    def test_isospectral_rotated_density_is_not_same_state(self):
        #
        # These two densities have exactly the same eigenvalues but their
        # occupied density subspaces point in different directions.
        #
        dm_a = np.zeros(
            (
                2,
                3,
                3,
            )
        )

        dm_a[0] = np.diag(
            (
                1.0,
                0.0,
                0.0,
            )
        )

        dm_a[1] = np.diag(
            (
                1.0,
                0.0,
                0.0,
            )
        )

        dm_b = np.zeros_like(
            dm_a
        )

        dm_b[0] = np.diag(
            (
                0.0,
                1.0,
                0.0,
            )
        )

        dm_b[1] = np.diag(
            (
                0.0,
                1.0,
                0.0,
            )
        )

        fp_a = (
            fingerprint_from_orthonormal_density(
                dm_a
            )
        )

        fp_b = (
            fingerprint_from_orthonormal_density(
                dm_b
            )
        )

        self.assertEqual(
            fp_a.total_spectrum,
            fp_b.total_spectrum,
        )

        a = make_root(
            "a",
            energy_mev=0.0,
            spin_2s=0,
            s2=0.0,
        )

        b = make_root(
            "b",
            energy_mev=20.0,
            spin_2s=0,
            s2=0.0,
        )

        comparison = compare_states(
            a,
            fp_a,
            b,
            fp_b,
            thresholds=TEST_THRESHOLDS,
        )

        self.assertAlmostEqual(
            comparison.total_spectrum_max,
            0.0,
        )

        self.assertGreater(
            comparison.total_density_rel_fro,
            TEST_THRESHOLDS.distinct_total_density_rel_fro,
        )

        self.assertEqual(
            comparison.relation,
            StateRelation.DISTINCT_STATE,
        )

    def test_common_ao_rotation_preserves_density_distance(self):
        dm_a = np.zeros(
            (
                2,
                3,
                3,
            )
        )

        dm_b = np.zeros_like(
            dm_a
        )

        dm_a[0] = np.diag(
            (
                1.0,
                0.3,
                0.0,
            )
        )

        dm_a[1] = np.diag(
            (
                0.8,
                0.1,
                0.0,
            )
        )

        dm_b[0] = np.diag(
            (
                0.9,
                0.4,
                0.0,
            )
        )

        dm_b[1] = np.diag(
            (
                0.7,
                0.2,
                0.0,
            )
        )

        angle = 0.43

        q = np.array(
            [
                [
                    np.cos(angle),
                    -np.sin(angle),
                    0.0,
                ],
                [
                    np.sin(angle),
                    np.cos(angle),
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                ],
            ]
        )

        def rotate(dm):
            return np.stack(
                (
                    q @ dm[0] @ q.T,
                    q @ dm[1] @ q.T,
                )
            )

        a1 = make_root(
            "a1"
        )

        b1 = make_root(
            "b1",
            energy_mev=3.0,
        )

        before = compare_states(
            a1,
            fingerprint_from_orthonormal_density(
                dm_a
            ),
            b1,
            fingerprint_from_orthonormal_density(
                dm_b
            ),
            thresholds=TEST_THRESHOLDS,
        )

        after = compare_states(
            a1,
            fingerprint_from_orthonormal_density(
                rotate(dm_a)
            ),
            b1,
            fingerprint_from_orthonormal_density(
                rotate(dm_b)
            ),
            thresholds=TEST_THRESHOLDS,
        )

        self.assertAlmostEqual(
            before.total_density_rel_fro,
            after.total_density_rel_fro,
            places=12,
        )

        self.assertAlmostEqual(
            before.spin_density_rel_fro,
            after.spin_density_rel_fro,
            places=12,
        )

    def test_compare_states_rejects_different_geometry(self):
        a = make_root(
            "a",
            r_angstrom=1.5,
        )

        b = make_root(
            "b",
            r_angstrom=1.6,
        )

        with self.assertRaisesRegex(
            ValueError,
            "different geometries",
        ):
            compare_states(
                a,
                make_fingerprint(),
                b,
                make_fingerprint(),
                thresholds=(
                    TEST_THRESHOLDS
                ),
            )


class DeduplicationTests(
    unittest.TestCase
):

    def test_two_mutually_same_roots_are_deduplicated(self):
        a = make_root(
            "a",
            energy_mev=0.0,
        )

        b = make_root(
            "b",
            energy_mev=0.4,
        )

        result = deduplicate_roots(
            (
                a,
                b,
            ),
            {
                "a": make_fingerprint(
                    0.000
                ),
                "b": make_fingerprint(
                    0.004
                ),
            },
            thresholds=TEST_THRESHOLDS,
        )

        self.assertEqual(
            len(result.clusters),
            1,
        )

        self.assertEqual(
            result.clusters[
                0
            ].member_root_ids,
            (
                "a",
                "b",
            ),
        )

        self.assertEqual(
            result.clusters[
                0
            ].representative_root_id,
            "a",
        )

        self.assertEqual(
            result.ambiguous_components,
            (),
        )

    def test_nontransitive_same_links_are_not_merged(self):
        #
        # A-B: same
        # B-C: same
        # A-C: spectrum difference exceeds same threshold
        #      but remains below distinct threshold -> ambiguous
        #
        a = make_root(
            "a",
            energy_mev=0.0,
        )

        b = make_root(
            "b",
            energy_mev=0.4,
        )

        c = make_root(
            "c",
            energy_mev=0.8,
        )

        result = deduplicate_roots(
            (
                a,
                b,
                c,
            ),
            {
                "a": make_fingerprint(
                    0.000
                ),
                "b": make_fingerprint(
                    0.006
                ),
                "c": make_fingerprint(
                    0.012
                ),
            },
            thresholds=TEST_THRESHOLDS,
        )

        self.assertEqual(
            result.ambiguous_components,
            (
                (
                    "a",
                    "b",
                    "c",
                ),
            ),
        )

        self.assertEqual(
            sorted(
                cluster.member_root_ids
                for cluster
                in result.clusters
            ),
            [
                ("a",),
                ("b",),
                ("c",),
            ],
        )

    def test_ambiguous_pair_is_retained_separately(self):
        a = make_root(
            "a",
            energy_mev=0.0,
        )

        b = make_root(
            "b",
            energy_mev=3.0,
        )

        result = deduplicate_roots(
            (
                a,
                b,
            ),
            {
                "a": make_fingerprint(
                    0.000
                ),
                "b": make_fingerprint(
                    0.020
                ),
            },
            thresholds=TEST_THRESHOLDS,
        )

        self.assertEqual(
            result.ambiguous_components,
            (
                (
                    "a",
                    "b",
                ),
            ),
        )

        self.assertEqual(
            len(result.clusters),
            2,
        )

    def test_ineligible_root_is_preserved_as_singleton(self):
        good = make_root(
            "good",
        )

        failed = make_root(
            "failed",
            status=(
                SCFRunStatus.FAILED
            ),
        )

        result = deduplicate_roots(
            (
                good,
                failed,
            ),
            {
                "good":
                    make_fingerprint(),
            },
            thresholds=TEST_THRESHOLDS,
        )

        self.assertEqual(
            result.ineligible_root_ids,
            (
                "failed",
            ),
        )

        member_sets = {
            cluster.member_root_ids
            for cluster
            in result.clusters
        }

        self.assertEqual(
            member_sets,
            {
                ("good",),
                ("failed",),
            },
        )

    def test_different_spin_sectors_do_not_merge(self):
        a = make_root(
            "doublet",
            spin_2s=1,
            s2=0.75,
        )

        b = make_root(
            "quartet",
            spin_2s=3,
            s2=3.75,
        )

        result = deduplicate_roots(
            (
                a,
                b,
            ),
            {
                "doublet":
                    make_fingerprint(),
                "quartet":
                    make_fingerprint(),
            },
            thresholds=TEST_THRESHOLDS,
        )

        self.assertEqual(
            len(result.clusters),
            2,
        )

        self.assertEqual(
            result.comparisons[
                0
            ].relation,
            StateRelation.DISTINCT_STATE,
        )


if __name__ == "__main__":
    unittest.main()
