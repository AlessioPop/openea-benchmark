from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from pyscf import (
    fci,
    gto,
    lib,
    mcscf,
    mrpt,
    scf,
)
from pyscf.mcscf import avas
from pyscf.scf import chkfile


OUT = Path(
    "/srv/storage/homes/analysis/dschmid/photodetachment/"
    "openea-pilot/feh_mr_nevpt2_unified"
)

DFT = Path(
    "/srv/storage/homes/analysis/dschmid/photodetachment/"
    "openea-pilot/feh_fine_diffuse"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


HARTREE_TO_EV = 27.211386245988

BASIS = "def2-tzvppd"

NROOTS = 4

#
# Fe [Ar] = 18 inactive electrons = 9 doubly occupied MOs.
#
NCORE = 9

#
# Literature-motivated FeH active orbital set:
#
# Fe 3d : 5
# Fe 4d : 5
# Fe 4s : 1
# Fe 4p : 3
# H  1s : 1
#
# total  : 15
#
EXPECTED_NCAS = 15

AO_TARGETS = (
    "Fe 3d",
    "Fe 4d",
    "Fe 4s",
    "Fe 4p",
    "H 1s",
)


SYSTEMS = (
    {
        "key": "neutral_quartet",
        "r_angstrom": 1.575,
        "spin_2s": 3,
        "active_electrons": 9,
    },
    {
        "key": "anion_quintet",
        "r_angstrom": 1.675,
        "spin_2s": 4,
        "active_electrons": 10,
    },
)


def find_pbe0_checkpoint(
    system,
):
    directory = (
        DFT
        / "checkpoints"
        / "pbe0"
        / system["key"]
    )

    token = (
        f"R{system['r_angstrom']:.12f}"
    )

    candidates = tuple(
        sorted(
            path
            for path in directory.glob(
                "*.chk"
            )
            if token in path.name
        )
    )

    if not candidates:
        raise RuntimeError(
            f"No PBE0 checkpoint for "
            f"{system['key']} "
            f"R={system['r_angstrom']}"
        )

    #
    # Prefer atom guess purely for deterministic restart.
    # All accepted PBE0 roots here belonged to the same
    # electronic manifold.
    #
    atom = tuple(
        path
        for path in candidates
        if "__atom__" in path.name
    )

    if atom:
        return atom[0]

    return candidates[0]


def build_rohf(
    source_chk,
):
    mol, _ = chkfile.load_scf(
        str(
            source_chk
        )
    )

    mol.verbose = 4
    mol.max_memory = 60000

    #
    # PySCF's ROHF checkpoint initializer explicitly converts
    # UHF/UKS-style checkpoint information to an ROHF density.
    #
    dm0 = (
        scf.rohf
        .init_guess_by_chkfile(
            mol,
            str(
                source_chk
            ),
            project=False,
        )
    )

    mf = scf.ROHF(
        mol
    )

    mf.max_memory = 60000
    mf.max_cycle = 200
    mf.conv_tol = 1.0e-10
    mf.conv_tol_grad = 1.0e-7
    mf.diis_space = 12
    mf.chkfile = str(
        OUT
        / (
            source_chk.stem
            + "__rohf.chk"
        )
    )

    print()
    print(
        "--- ROHF from PBE0 checkpoint density ---"
    )

    mf.kernel(
        dm0=dm0
    )

    if not mf.converged:
        print()
        print(
            "Conventional ROHF did not converge; "
            "switching to Newton ROHF."
        )

        newton = mf.newton()

        newton.max_memory = 60000
        newton.max_cycle = 120
        newton.conv_tol = 1.0e-10
        newton.conv_tol_grad = 1.0e-7
        newton.verbose = 4

        newton.kernel(
            mo_coeff=mf.mo_coeff,
            mo_occ=mf.mo_occ,
        )

        mf = newton

    if not mf.converged:
        raise RuntimeError(
            "ROHF did not converge"
        )

    ss, multiplicity = (
        mf.spin_square()
    )

    print()
    print(
        f"E(ROHF) = {mf.e_tot:.12f} Eh"
    )
    print(
        f"<S^2>   = {ss:.8f}"
    )
    print(
        f"mult.    = {multiplicity:.8f}"
    )

    return (
        mol,
        mf,
    )


def build_active_space(
    mf,
    system,
):
    """
    Construct one charge-balanced physical active space.

    Definition
    ----------
    1. Freeze exactly NCORE spatial orbitals.
    2. Treat every remaining MO on equal footing.
    3. Project the complete post-core MO space onto the
       15-dimensional target AO manifold

           Fe 3d
           Fe 4d
           Fe 4s
           Fe 4p
           H  1s

    4. Diagonalize the projector.
    5. The 15 highest-projector-eigenvalue directions form
       the active orbital space.

    Thus neutral and anion use the same orbital definition.
    Only the number of active electrons changes.
    """

    print()
    print(
        "--- UNIFIED 15-ORBITAL PROJECTOR ACTIVE SPACE ---"
    )

    mol = mf.mol

    mo_coeff = np.asarray(
        mf.mo_coeff
    )

    if mo_coeff.ndim != 2:
        raise RuntimeError(
            "Expected spatial ROHF orbitals."
        )

    nmo = mo_coeff.shape[
        1
    ]

    if NCORE >= nmo:
        raise RuntimeError(
            "NCORE is inconsistent with MO dimension."
        )

    core = (
        mo_coeff[
            :,
            :NCORE
        ]
    )

    post_core = (
        mo_coeff[
            :,
            NCORE:
        ]
    )

    #
    # Build the same reference AO manifold for both charge
    # states.
    #
    pmol = mol.copy()

    pmol.atom = (
        mol._atom
    )

    pmol.unit = "B"
    pmol.symmetry = False
    pmol.basis = BASIS

    pmol.build(
        False,
        False,
    )

    baslst = (
        pmol.search_ao_label(
            AO_TARGETS
        )
    )

    print(
        "reference AO indices:",
        baslst,
    )

    if (
        len(
            baslst
        )
        != EXPECTED_NCAS
    ):
        raise RuntimeError(
            "Expected exactly "
            f"{EXPECTED_NCAS} target AOs, "
            f"found {len(baslst)}"
        )

    reference_overlap = (
        pmol.intor_symmetric(
            "int1e_ovlp"
        )[
            baslst
        ][
            :,
            baslst
        ]
    )

    cross_overlap = (
        gto.intor_cross(
            "int1e_ovlp",
            pmol,
            mol,
        )[
            baslst
        ]
    )

    b = (
        cross_overlap
        @ post_core
    )

    projector = (
        b.T
        @ np.linalg.solve(
            reference_overlap,
            b,
        )
    )

    projector = (
        0.5
        * (
            projector
            + projector.T
        )
    )

    (
        eigvals,
        eigvecs,
    ) = np.linalg.eigh(
        projector
    )

    order = np.argsort(
        eigvals
    )[::-1]

    eigvals = (
        eigvals[
            order
        ]
    )

    eigvecs = (
        eigvecs[
            :,
            order
        ]
    )

    #
    # Because the post-core MOs are orthonormal,
    # this eigenvector rotation preserves orthonormality.
    #
    rotated_post_core = (
        post_core
        @ eigvecs
    )

    active = (
        rotated_post_core[
            :,
            :EXPECTED_NCAS
        ]
    )

    external = (
        rotated_post_core[
            :,
            EXPECTED_NCAS:
        ]
    )

    mo = np.hstack(
        (
            core,
            active,
            external,
        )
    )

    if (
        mo.shape
        != mo_coeff.shape
    ):
        raise RuntimeError(
            "Unified MO matrix has wrong dimensions."
        )

    active_electrons = (
        mol.nelectron
        - 2 * NCORE
    )

    expected_electrons = (
        system[
            "active_electrons"
        ]
    )

    if (
        active_electrons
        != expected_electrons
    ):
        raise RuntimeError(
            "Active-electron count mismatch: "
            f"{active_electrons} vs "
            f"{expected_electrons}"
        )

    spin = (
        system[
            "spin_2s"
        ]
    )

    nalpha = (
        active_electrons
        + spin
    ) // 2

    nbeta = (
        active_electrons
        - spin
    ) // 2

    if (
        nalpha + nbeta
        != active_electrons
        or nalpha - nbeta
        != spin
    ):
        raise RuntimeError(
            "Invalid CAS electron/spin combination."
        )

    nelecas = (
        int(
            nalpha
        ),
        int(
            nbeta
        ),
    )

    #
    # Verify orthonormality explicitly.
    #
    overlap = mol.intor(
        "int1e_ovlp"
    )

    metric = (
        mo.T
        @ overlap
        @ mo
    )

    orth_error = float(
        np.max(
            np.abs(
                metric
                - np.eye(
                    nmo
                )
            )
        )
    )

    print()
    print(
        "fixed inactive core orbitals:",
        NCORE,
    )

    print(
        "post-core orbital dimension:",
        post_core.shape[1],
    )

    print(
        "active-space dimension:",
        EXPECTED_NCAS,
    )

    print(
        "active electrons:",
        active_electrons,
    )

    print(
        "CAS definition:",
        f"CAS({active_electrons},"
        f"{EXPECTED_NCAS})",
        "nelecas=",
        nelecas,
    )

    print()
    print(
        "top 20 projector eigenvalues:"
    )

    print(
        np.array2string(
            eigvals[
                :20
            ],
            precision=10,
            suppress_small=False,
        )
    )

    lambda_15 = float(
        eigvals[
            EXPECTED_NCAS - 1
        ]
    )

    lambda_16 = float(
        eigvals[
            EXPECTED_NCAS
        ]
    )

    print()
    print(
        "lambda_15:",
        f"{lambda_15:.12f}",
    )

    print(
        "lambda_16:",
        f"{lambda_16:.12f}",
    )

    print(
        "projector gap:",
        f"{lambda_15 - lambda_16:.12f}",
    )

    print(
        "MO orthonormality max error:",
        f"{orth_error:.3e}",
    )

    if (
        orth_error
        > 1.0e-8
    ):
        raise RuntimeError(
            "Unified MO basis is not orthonormal."
        )

    #
    # The target reference space has dimension 15.
    # A numerically non-zero 16th projector eigenvalue would
    # indicate that something is inconsistent in the
    # projection construction.
    #
    if (
        abs(
            lambda_16
        )
        > 1.0e-8
    ):
        raise RuntimeError(
            "Unified projector unexpectedly has "
            "rank greater than 15."
        )

    return (
        EXPECTED_NCAS,
        nelecas,
        mo,
    )


def run_casscf_nevpt2(
    mol,
    mf,
    system,
    ncas,
    nelecas,
    mo,
):
    spin = system[
        "spin_2s"
    ]

    s = spin / 2.0
    target_s2 = (
        s
        * (
            s + 1.0
        )
    )

    print()
    print(
        "--- FOUR-ROOT SA-CASSCF ---"
    )

    mc = mcscf.CASSCF(
        mf,
        ncas,
        nelecas,
    )

    mc.max_memory = 60000
    mc.max_cycle_macro = 50
    mc.conv_tol = 1.0e-8
    mc.conv_tol_grad = 1.0e-5

    mc.chkfile = str(
        OUT
        / (
            system["key"]
            + "__sa_casscf.chk"
        )
    )

    #
    # Four roots of the requested spin manifold.
    #
    solver = (
        fci.direct_spin1.FCI(
            mol
        )
    )

    solver.spin = spin
    solver.nroots = NROOTS
    solver.max_cycle = 100
    solver.conv_tol = 1.0e-9

    #
    # Sz alone does not exclude higher-spin solutions.
    # Apply a modest spin penalty and verify S^2 afterwards.
    #
    solver = fci.addons.fix_spin(
        solver,
        shift=0.2,
        ss=target_s2,
    )

    solver.nroots = NROOTS

    mc.fcisolver = solver

    mc.state_average_(
        [
            1.0 / NROOTS
        ]
        * NROOTS
    )

    mc.kernel(
        mo
    )

    if not mc.converged:
        raise RuntimeError(
            "SA-CASSCF did not converge"
        )

    e_states = np.asarray(
        mc.e_states,
        dtype=float,
    )

    if len(
        e_states
    ) != NROOTS:
        raise RuntimeError(
            "Unexpected number of CASSCF roots"
        )

    print()
    print(
        "CASSCF ROOT ENERGIES"
    )

    try:
        ss_values, mult_values = (
            mc.fcisolver
            .states_spin_square(
                mc.ci,
                ncas,
                nelecas,
            )
        )

        ss_values = np.asarray(
            ss_values,
            dtype=float,
        )

        mult_values = np.asarray(
            mult_values,
            dtype=float,
        )

    except Exception:
        ss_values = np.full(
            NROOTS,
            np.nan,
        )

        mult_values = np.full(
            NROOTS,
            np.nan,
        )

    #
    # Root-specific active-space 1-RDMs.
    #
    dm1_roots = (
        mc.fcisolver
        .states_make_rdm1(
            mc.ci,
            ncas,
            nelecas,
        )
    )

    root_data = []

    for root in range(
        NROOTS
    ):
        occupations = np.linalg.eigvalsh(
            np.asarray(
                dm1_roots[
                    root
                ]
            )
        )[::-1]

        max_ci = float(
            np.max(
                np.abs(
                    np.asarray(
                        mc.ci[
                            root
                        ]
                    )
                )
            )
        )

        print()
        print(
            f"ROOT {root}"
        )
        print(
            f"  E(CASSCF) = "
            f"{e_states[root]:.12f} Eh"
        )
        print(
            f"  <S^2>     = "
            f"{ss_values[root]:.8f}"
        )
        print(
            f"  mult.      = "
            f"{mult_values[root]:.8f}"
        )
        print(
            f"  largest |CI| = "
            f"{max_ci:.6f}"
        )
        print(
            f"  largest CI weight = "
            f"{max_ci**2:.6f}"
        )
        print(
            "  natural occupations:"
        )
        print(
            " ",
            np.array2string(
                occupations,
                precision=6,
                suppress_small=True,
            ),
        )

        root_data.append(
            {
                "root": root,
                "casscf_energy_hartree": (
                    float(
                        e_states[
                            root
                        ]
                    )
                ),
                "s2": float(
                    ss_values[
                        root
                    ]
                ),
                "multiplicity": float(
                    mult_values[
                        root
                    ]
                ),
                "largest_ci_coefficient": (
                    max_ci
                ),
                "largest_ci_weight": (
                    max_ci ** 2
                ),
                "natural_occupations": (
                    occupations.tolist()
                ),
            }
        )

    #
    # PySCF cannot apply NEVPT2 directly to the
    # StateAverageFCISolver wrapper.  The official workflow is:
    #
    #   SA-CASSCF orbital optimization
    #       ->
    #   separated multi-root CASCI using the optimized orbitals
    #       ->
    #   root-specific SC-NEVPT2
    #
    print()
    print(
        "--- SEPARATED FOUR-ROOT CASCI "
        "ON SA-CASSCF ORBITALS ---"
    )

    casci = mcscf.CASCI(
        mf,
        ncas,
        nelecas,
    )

    casci.max_memory = 60000

    casci_solver = (
        fci.direct_spin1.FCI(
            mol
        )
    )

    casci_solver.spin = spin
    casci_solver.nroots = NROOTS
    casci_solver.max_cycle = 100
    casci_solver.conv_tol = 1.0e-9

    casci_solver = (
        fci.addons.fix_spin(
            casci_solver,
            shift=0.2,
            ss=target_s2,
        )
    )

    casci_solver.nroots = NROOTS

    casci.fcisolver = (
        casci_solver
    )

    casci.kernel(
        mc.mo_coeff
    )

    casci_energies = np.asarray(
        casci.e_tot,
        dtype=float,
    ).reshape(-1)

    if (
        len(
            casci_energies
        )
        != NROOTS
    ):
        raise RuntimeError(
            "Separated CASCI returned "
            f"{len(casci_energies)} roots; "
            f"expected {NROOTS}"
        )

    print()
    print(
        "SEPARATED CASCI ROOTS"
    )

    try:
        (
            casci_ss,
            casci_mult,
        ) = (
            casci.fcisolver
            .states_spin_square(
                casci.ci,
                ncas,
                nelecas,
            )
        )

        casci_ss = np.asarray(
            casci_ss,
            dtype=float,
        )

        casci_mult = np.asarray(
            casci_mult,
            dtype=float,
        )

    except Exception:
        casci_ss = np.full(
            NROOTS,
            np.nan,
        )

        casci_mult = np.full(
            NROOTS,
            np.nan,
        )

    for root in range(
        NROOTS
    ):
        delta_ev = (
            casci_energies[
                root
            ]
            - e_states[
                root
            ]
        ) * HARTREE_TO_EV

        print()
        print(
            f"CASCI ROOT {root}"
        )

        print(
            "  E(CASCI) =",
            f"{casci_energies[root]:.12f} Eh",
        )

        print(
            "  SA-CASSCF root energy =",
            f"{e_states[root]:.12f} Eh",
        )

        print(
            "  CASCI-SA difference =",
            f"{delta_ev:.9f} eV",
        )

        print(
            "  <S^2> =",
            f"{casci_ss[root]:.8f}",
        )

        print(
            "  mult.  =",
            f"{casci_mult[root]:.8f}",
        )

        #
        # Preserve both energies explicitly.
        #
        root_data[
            root
        ][
            "sa_casscf_energy_hartree"
        ] = root_data[
            root
        ].pop(
            "casscf_energy_hartree"
        )

        root_data[
            root
        ][
            "casci_energy_hartree"
        ] = float(
            casci_energies[
                root
            ]
        )

        root_data[
            root
        ][
            "casci_s2"
        ] = float(
            casci_ss[
                root
            ]
        )

        root_data[
            root
        ][
            "casci_multiplicity"
        ] = float(
            casci_mult[
                root
            ]
        )

    print()
    print(
        "--- ROOT-SPECIFIC SC-NEVPT2 ---"
    )

    nevpt_totals = []

    for root in range(
        NROOTS
    ):
        print()
        print(
            f"NEVPT2 ROOT {root}"
        )

        nevpt = mrpt.NEVPT(
            casci,
            root=root,
        )

        corr = float(
            nevpt.kernel()
        )

        total = float(
            casci_energies[
                root
            ]
            + corr
        )

        nevpt_totals.append(
            total
        )

        root_data[
            root
        ][
            "nevpt2_correlation_hartree"
        ] = corr

        root_data[
            root
        ][
            "casci_nevpt2_energy_hartree"
        ] = total

        print(
            "  E(NEVPT2 corr) =",
            f"{corr:.12f} Eh",
        )

        print(
            "  E(CASCI+NEVPT2) =",
            f"{total:.12f} Eh",
        )

    order = np.argsort(
        np.asarray(
            nevpt_totals
        )
    )

    print()
    print(
        "NEVPT2 ROOT ORDER:"
    )

    reference = (
        nevpt_totals[
            int(
                order[0]
            )
        ]
    )

    for position, root in enumerate(
        order,
        start=1,
    ):
        root = int(
            root
        )

        gap_ev = (
            nevpt_totals[
                root
            ]
            - reference
        ) * HARTREE_TO_EV

        print(
            f"  {position}: "
            f"root {root}  "
            f"E={nevpt_totals[root]:.12f} Eh  "
            f"gap={gap_ev:.6f} eV"
        )

    return {
        "ncas": int(
            ncas
        ),
        "nelecas": list(
            nelecas
        ),
        "nroots": NROOTS,
        "roots": root_data,
        "lowest_nevpt2_root": int(
            order[0]
        ),
        "lowest_nevpt2_energy_hartree": float(
            reference
        ),
    }


def run_system(
    system,
):
    print()
    print(
        "=" * 120
    )
    print(
        system["key"],
        f"R={system['r_angstrom']:.3f} Å",
    )
    print(
        "=" * 120
    )

    source = (
        find_pbe0_checkpoint(
            system
        )
    )

    print(
        "PBE0 source:",
        source,
    )

    mol, mf = build_rohf(
        source
    )

    ncas, nelecas, mo = (
        build_active_space(
            mf,
            system,
        )
    )

    result = (
        run_casscf_nevpt2(
            mol,
            mf,
            system,
            ncas,
            nelecas,
            mo,
        )
    )

    result.update(
        {
            "system": (
                system[
                    "key"
                ]
            ),
            "r_angstrom": (
                system[
                    "r_angstrom"
                ]
            ),
            "spin_2s": (
                system[
                    "spin_2s"
                ]
            ),
            "source_checkpoint": str(
                source
            ),
            "basis": BASIS,
            "rohf_energy_hartree": float(
                mf.e_tot
            ),
        }
    )

    output = (
        OUT
        / (
            system["key"]
            + ".json"
        )
    )

    output.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    return result


def lowest_nevpt2_manifold(
    result,
    *,
    tolerance_mev=1.0,
):
    energies = np.asarray(
        [
            root[
                "casci_nevpt2_energy_hartree"
            ]
            for root
            in result[
                "roots"
            ]
        ],
        dtype=float,
    )

    minimum = float(
        np.min(
            energies
        )
    )

    delta_mev = (
        (
            energies
            - minimum
        )
        * HARTREE_TO_EV
        * 1000.0
    )

    members = np.where(
        delta_mev
        <= tolerance_mev
    )[0]

    if (
        len(
            members
        )
        == 0
    ):
        raise RuntimeError(
            "No lowest NEVPT2 manifold members."
        )

    manifold_energies = (
        energies[
            members
        ]
    )

    energy_min = float(
        np.min(
            manifold_energies
        )
    )

    energy_max = float(
        np.max(
            manifold_energies
        )
    )

    center = (
        0.5
        * (
            energy_min
            + energy_max
        )
    )

    spread_mev = (
        (
            energy_max
            - energy_min
        )
        * HARTREE_TO_EV
        * 1000.0
    )

    return {
        "members": [
            int(
                value
            )
            for value
            in members
        ],
        "center_hartree": (
            center
        ),
        "spread_mev": (
            spread_mev
        ),
        "threshold_mev": (
            tolerance_mev
        ),
    }


def main():
    print(
        "PySCF threads:",
        lib.num_threads(),
    )

    results = {}

    for system in SYSTEMS:
        cache = (
            OUT
            / (
                system[
                    "key"
                ]
                + ".json"
            )
        )

        reusable = False

        if cache.is_file():
            try:
                cached = json.loads(
                    cache.read_text()
                )

                reusable = (
                    cached.get(
                        "system"
                    )
                    == system[
                        "key"
                    ]
                    and abs(
                        float(
                            cached.get(
                                "r_angstrom"
                            )
                        )
                        - system[
                            "r_angstrom"
                        ]
                    )
                    < 1.0e-12
                    and (
                        cached.get(
                            "basis"
                        )
                        == BASIS
                    )
                    and len(
                        cached.get(
                            "roots",
                            [],
                        )
                    )
                    == NROOTS
                    and all(
                        (
                            "casci_nevpt2_energy_hartree"
                            in root
                        )
                        for root
                        in cached.get(
                            "roots",
                            [],
                        )
                    )
                )

            except Exception:
                reusable = False

        if reusable:
            print()
            print(
                "=" * 120
            )
            print(
                "REUSING COMPLETED RESULT:",
                system[
                    "key"
                ],
            )
            print(
                cache
            )
            print(
                "=" * 120
            )

            result = cached

        else:
            result = run_system(
                system
            )

        results[
            system[
                "key"
            ]
        ] = result

    neutral = results[
        "neutral_quartet"
    ]

    anion = results[
        "anion_quintet"
    ]

    neutral_manifold = (
        lowest_nevpt2_manifold(
            neutral
        )
    )

    anion_manifold = (
        lowest_nevpt2_manifold(
            anion
        )
    )

    neutral[
        "lowest_nevpt2_manifold"
    ] = neutral_manifold

    anion[
        "lowest_nevpt2_manifold"
    ] = anion_manifold

    print()
    print(
        "=" * 120
    )
    print(
        "LOWEST SC-NEVPT2 MANIFOLDS"
    )
    print(
        "=" * 120
    )

    for (
        label,
        manifold,
    ) in (
        (
            "neutral",
            neutral_manifold,
        ),
        (
            "anion",
            anion_manifold,
        ),
    ):
        print()
        print(
            label
        )

        print(
            "  roots:",
            manifold[
                "members"
            ],
        )

        print(
            "  center:",
            f"{manifold['center_hartree']:.12f}",
            "Eh",
        )

        print(
            "  spread:",
            f"{manifold['spread_mev']:.6f}",
            "meV",
        )

    ea = (
        neutral_manifold[
            "center_hartree"
        ]
        - anion_manifold[
            "center_hartree"
        ]
    ) * HARTREE_TO_EV

    print()
    print(
        "=" * 120
    )
    print(
        "MULTIREFERENCE ELECTRONIC EA CANDIDATE"
    )
    print(
        "=" * 120
    )

    print(
        f"EA_elec candidate = "
        f"{ea:.6f} eV"
    )

    print()
    print(
        "IMPORTANT: this is not yet the final EA."
    )

    print(
        "The energy of each near-degenerate "
        "ground-state manifold is represented "
        "by its energy-envelope midpoint."
    )

    print(
        "No ZPE, SOC, scalar-relativistic correction, "
        "basis convergence/extrapolation, or "
        "MR-level PEC refinement has yet been applied."
    )

    combined = (
        OUT
        / "combined.json"
    )

    combined.write_text(
        json.dumps(
            results,
            indent=2,
        )
    )

    #
    # Refresh individual files as well so that the
    # manifold summaries are retained.
    #
    for key, result in (
        results.items()
    ):
        (
            OUT
            / f"{key}.json"
        ).write_text(
            json.dumps(
                result,
                indent=2,
            )
        )

    print()
    print(
        "saved:",
        combined,
    )


if __name__ == "__main__":
    main()
