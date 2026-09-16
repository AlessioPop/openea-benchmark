from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openea_benchmark import (
    DFTMethodSpec,
    DiatomicSpec,
    IdentityThresholds,
    SCFRootRecord,
    SCFRunStatus,
    SCFSettings,
    StateFingerprint,
    StateRelation,
    compare_states,
    deduplicate_roots,
    fingerprint_from_checkpoint,
    run_scf_attempt,
)

from openea_benchmark.pyscf_backend import (
    _root_id,
)


FAST_SETTINGS = SCFSettings(
    grid_level=1,
    num_threads=1,
)


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


def make_root(
    root_id: str,
    *,
    r_angstrom: float = 1.50,
    ecp_name: str = "ecp_a",
) -> SCFRootRecord:
    return SCFRootRecord(
        root_id=root_id,
        molecule="XY",
        charge=0,
        spin_2s=0,
        r_angstrom=r_angstrom,
        functional="pbe0",
        basis="def2-svp",
        origin_guess=root_id,
        scf_path="standard",
        reference="RKS",
        status=SCFRunStatus.CANONICALIZED,
        energy_hartree=-10.0,
        internal_stable=True,
        s2=0.0,
        observed_multiplicity=1.0,
        ecp_assignments=(
            ("X", ecp_name),
        ),
    )


def make_fingerprint() -> StateFingerprint:
    return StateFingerprint(
        total_spectrum=(2.0,),
        spin_spectrum=(0.0,),
        total_density_orth=(
            (2.0,),
        ),
        spin_density_orth=(
            (0.0,),
        ),
        total_trace=2.0,
        spin_trace=0.0,
    )


class ScientificContextHardeningTests(
    unittest.TestCase
):

    def test_compare_states_rejects_different_ecp_assignments(self):
        a = make_root(
            "a",
            ecp_name="ecp_a",
        )
        b = make_root(
            "b",
            ecp_name="ecp_b",
        )

        fingerprint = make_fingerprint()

        with self.assertRaisesRegex(
            ValueError,
            "ecp",
        ):
            compare_states(
                a,
                fingerprint,
                b,
                fingerprint,
                thresholds=TEST_THRESHOLDS,
            )

    def test_deduplication_does_not_merge_different_ecp_contexts(self):
        a = make_root(
            "a",
            ecp_name="ecp_a",
        )
        b = make_root(
            "b",
            ecp_name="ecp_b",
        )

        fingerprint = make_fingerprint()

        result = deduplicate_roots(
            (a, b),
            {
                "a": fingerprint,
                "b": fingerprint,
            },
            thresholds=TEST_THRESHOLDS,
        )

        self.assertEqual(
            len(result.clusters),
            2,
        )

        self.assertEqual(
            {
                cluster.member_root_ids
                for cluster in result.clusters
            },
            {
                ("a",),
                ("b",),
            },
        )

    def test_root_id_distinguishes_ecp_assignments(self):
        spec = DiatomicSpec(
            label="XY",
            atom_a="X",
            atom_b="Y",
            r_angstrom=1.50,
            charge=0,
            spin_2s=0,
        )

        method_a = DFTMethodSpec(
            functional="pbe0",
            basis="def2-svp",
            ecp_assignments=(
                ("X", "ecp_a"),
            ),
        )

        method_b = DFTMethodSpec(
            functional="pbe0",
            basis="def2-svp",
            ecp_assignments=(
                ("X", "ecp_b"),
            ),
        )

        self.assertNotEqual(
            _root_id(
                spec,
                method_a,
                "minao",
            ),
            _root_id(
                spec,
                method_b,
                "minao",
            ),
        )

    def test_root_id_does_not_collapse_distinct_geometries(self):
        method = DFTMethodSpec(
            functional="pbe0",
            basis="def2-svp",
        )

        a = DiatomicSpec(
            label="XY",
            atom_a="X",
            atom_b="Y",
            r_angstrom=1.0000001,
            charge=0,
            spin_2s=0,
        )

        b = DiatomicSpec(
            label="XY",
            atom_a="X",
            atom_b="Y",
            r_angstrom=1.0000004,
            charge=0,
            spin_2s=0,
        )

        self.assertNotEqual(
            _root_id(
                a,
                method,
                "minao",
            ),
            _root_id(
                b,
                method,
                "minao",
            ),
        )

    def test_equivalent_float_geometries_are_canonicalized(self):
        a = DiatomicSpec(
            label="XY",
            atom_a="X",
            atom_b="Y",
            r_angstrom=1.2,
            charge=0,
            spin_2s=0,
        )

        b = DiatomicSpec(
            label="XY",
            atom_a="X",
            atom_b="Y",
            r_angstrom=0.4 * 3.0,
            charge=0,
            spin_2s=0,
        )

        self.assertEqual(
            a.r_angstrom,
            b.r_angstrom,
        )

        root_a = make_root(
            "root_a",
            r_angstrom=1.2,
        )

        root_b = make_root(
            "root_b",
            r_angstrom=0.4 * 3.0,
        )

        self.assertEqual(
            root_a.r_angstrom,
            root_b.r_angstrom,
        )

    def test_checkpoint_geometry_mismatch_is_rejected(self):
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

            wrong_geometry = replace(
                root,
                root_id="wrong_geometry",
                r_angstrom=0.75,
            )

            with self.assertRaisesRegex(
                ValueError,
                "geometry|bond|distance",
            ):
                fingerprint_from_checkpoint(
                    wrong_geometry
                )


if __name__ == "__main__":
    unittest.main()
