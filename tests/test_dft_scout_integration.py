from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openea_benchmark import (
    BranchThresholds,
    DFTMethodSpec,
    DiatomicSpec,
    IdentityThresholds,
    SCFRunStatus,
    SCFSettings,
    build_branch_graph,
    construct_local_pecs,
    deduplicate_checkpoint_roots,
    run_guess_panel,
)


FAST_SETTINGS = SCFSettings(
    grid_level=1,
    num_threads=1,
)


#
# Validation-only thresholds.
#
# These are deliberately not OpenEA production defaults.
#
IDENTITY_THRESHOLDS = IdentityThresholds(
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


BRANCH_THRESHOLDS = BranchThresholds(
    continuous_occ_min=0.95,
    discontinuous_occ_min=0.50,
    continuous_delta_s2=0.02,
    discontinuous_delta_s2=0.20,
    max_step_angstrom=0.10,
)


class DFTScoutIntegrationTests(
    unittest.TestCase
):

    def test_h2_multi_geometry_pipeline_reaches_one_local_pec(self):
        method = DFTMethodSpec(
            functional="pbe",
            basis="def2-svp",
        )

        geometries = (
            0.72,
            0.74,
            0.76,
        )

        with TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)

            representative_roots = []
            all_roots = []

            for r_angstrom in geometries:
                panel = run_guess_panel(
                    DiatomicSpec(
                        label="H2",
                        atom_a="H",
                        atom_b="H",
                        r_angstrom=r_angstrom,
                        charge=0,
                        spin_2s=0,
                    ),
                    method,
                    guesses=(
                        "minao",
                        "hcore",
                    ),
                    settings=FAST_SETTINGS,
                    checkpoint_dir=checkpoint_dir,
                )

                self.assertEqual(
                    len(panel),
                    2,
                )

                self.assertTrue(
                    all(
                        root.status
                        == SCFRunStatus.CANONICALIZED
                        for root in panel
                    )
                )

                all_roots.extend(
                    panel
                )

                dedup = (
                    deduplicate_checkpoint_roots(
                        panel,
                        thresholds=(
                            IDENTITY_THRESHOLDS
                        ),
                    )
                )

                self.assertEqual(
                    dedup.fingerprint_failures,
                    (),
                )

                self.assertEqual(
                    len(dedup.audits),
                    2,
                )

                self.assertEqual(
                    dedup.deduplication
                    .ambiguous_components,
                    (),
                )

                self.assertEqual(
                    dedup.deduplication
                    .ineligible_root_ids,
                    (),
                )

                self.assertEqual(
                    len(
                        dedup.deduplication
                        .clusters
                    ),
                    1,
                )

                cluster = (
                    dedup.deduplication
                    .clusters[0]
                )

                self.assertEqual(
                    len(
                        cluster.member_root_ids
                    ),
                    2,
                )

                root_by_id = {
                    root.root_id: root
                    for root in panel
                }

                representative_roots.append(
                    root_by_id[
                        cluster.representative_root_id
                    ]
                )

            self.assertEqual(
                len(all_roots),
                6,
            )

            representative_roots = tuple(
                sorted(
                    representative_roots,
                    key=lambda root:
                        root.r_angstrom,
                )
            )

            self.assertEqual(
                tuple(
                    root.r_angstrom
                    for root
                    in representative_roots
                ),
                geometries,
            )

            graph = build_branch_graph(
                representative_roots,
                thresholds=(
                    BRANCH_THRESHOLDS
                ),
            )

            self.assertEqual(
                len(graph.comparisons),
                2,
            )

            self.assertEqual(
                len(graph.continuous_edges),
                2,
            )

            self.assertEqual(
                graph.ambiguous_edges,
                (),
            )

            self.assertEqual(
                graph.discontinuous_edges,
                (),
            )

            self.assertTrue(
                graph.is_fully_unambiguous
            )

            self.assertEqual(
                len(graph.components),
                1,
            )

            pec_result = (
                construct_local_pecs(
                    representative_roots,
                    graph,
                )
            )

            self.assertEqual(
                pec_result.rejected_components,
                (),
            )

            self.assertEqual(
                len(pec_result.pecs),
                1,
            )

            pec = pec_result.pecs[0]

            self.assertEqual(
                tuple(
                    point.r_angstrom
                    for point
                    in pec.points
                ),
                geometries,
            )

            self.assertEqual(
                tuple(
                    point.root_id
                    for point
                    in pec.points
                ),
                tuple(
                    root.root_id
                    for root
                    in representative_roots
                ),
            )

            self.assertEqual(
                tuple(
                    point.energy_hartree
                    for point
                    in pec.points
                ),
                tuple(
                    root.energy_hartree
                    for root
                    in representative_roots
                ),
            )


if __name__ == "__main__":
    unittest.main()
