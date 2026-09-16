from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openea_benchmark import (
    DFTMethodSpec,
    DiatomicSpec,
    SCFRunStatus,
    SCFSettings,
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


class RootIdentityContextTests(
    unittest.TestCase
):

    def test_root_id_distinguishes_atomic_composition(self):
        method = DFTMethodSpec(
            functional="pbe",
            basis="def2-svp",
        )

        hh = DiatomicSpec(
            label="XY",
            atom_a="H",
            atom_b="H",
            r_angstrom=1.00,
            charge=0,
            spin_2s=0,
        )

        hehe = DiatomicSpec(
            label="XY",
            atom_a="He",
            atom_b="He",
            r_angstrom=1.00,
            charge=0,
            spin_2s=0,
        )

        self.assertNotEqual(
            _root_id(
                hh,
                method,
                "minao",
            ),
            _root_id(
                hehe,
                method,
                "minao",
            ),
        )

    def test_root_id_is_not_vulnerable_to_slug_collision(self):
        method = DFTMethodSpec(
            functional="pbe",
            basis="def2-svp",
        )

        a = DiatomicSpec(
            label="A/B",
            atom_a="H",
            atom_b="H",
            r_angstrom=1.00,
            charge=0,
            spin_2s=0,
        )

        b = DiatomicSpec(
            label="A B",
            atom_a="H",
            atom_b="H",
            r_angstrom=1.00,
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


class CheckpointScientificContextTests(
    unittest.TestCase
):

    @classmethod
    def setUpClass(cls):
        cls._tmp = TemporaryDirectory()

        cls.root = run_scf_attempt(
            DiatomicSpec(
                label="H2",
                atom_a="H",
                atom_b="H",
                r_angstrom=0.74,
                charge=0,
                spin_2s=0,
            ),
            DFTMethodSpec(
                functional="pbe",
                basis="def2-svp",
            ),
            "minao",
            settings=FAST_SETTINGS,
            checkpoint_dir=Path(
                cls._tmp.name
            ),
        )

        if (
            cls.root.status
            != SCFRunStatus.CANONICALIZED
        ):
            raise RuntimeError(
                "H2 validation root did not canonicalize"
            )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_root_record_preserves_atomic_identity(self):
        self.assertEqual(
            (
                self.root.atom_a,
                self.root.atom_b,
            ),
            (
                "H",
                "H",
            ),
        )

    def test_checkpoint_rejects_wrong_molecule_label(self):
        forged = replace(
            self.root,
            molecule="NOT_H2",
        )

        with self.assertRaisesRegex(
            ValueError,
            "context|molecule|metadata",
        ):
            fingerprint_from_checkpoint(
                forged
            )

    def test_checkpoint_rejects_wrong_functional(self):
        forged = replace(
            self.root,
            functional="b3lyp",
        )

        with self.assertRaisesRegex(
            ValueError,
            "context|functional|metadata",
        ):
            fingerprint_from_checkpoint(
                forged
            )

    def test_checkpoint_rejects_wrong_basis(self):
        forged = replace(
            self.root,
            basis="cc-pvdz",
        )

        with self.assertRaisesRegex(
            ValueError,
            "context|basis|metadata",
        ):
            fingerprint_from_checkpoint(
                forged
            )

    def test_checkpoint_rejects_wrong_ecp_context(self):
        forged = replace(
            self.root,
            ecp_assignments=(
                (
                    "H",
                    "bfd",
                ),
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "context|ecp|metadata",
        ):
            fingerprint_from_checkpoint(
                forged
            )


if __name__ == "__main__":
    unittest.main()
