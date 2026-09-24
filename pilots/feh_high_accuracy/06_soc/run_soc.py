from __future__ import annotations

from pathlib import Path
import copy
import json
import sys

import numpy as np

from pyscf import (
    fci,
    gto,
    mcscf,
    scf,
)
from pyscf.scf import chkfile


PEC_DIR = Path(
    "/srv/storage/homes/analysis/dschmid/photodetachment/"
    "openea-pilot/feh_x2c_ccsdt_local_pec"
)

CENTER_DIR = Path(
    "/srv/storage/homes/analysis/dschmid/photodetachment/"
    "openea-pilot/feh_ccpy_x2c_qzvppd"
)

OUT = Path(
    "/srv/storage/homes/analysis/dschmid/photodetachment/"
    "openea-pilot/feh_soc_siso_qzvppd"
)

SISO_DIR = Path(
    "/srv/storage/homes/analysis/dschmid/photodetachment/"
    "openea-pilot/external/fci-siso"
)

sys.path.insert(
    0,
    str(SISO_DIR),
)

from fcisiso import (       # noqa: E402
    FCISISO,
    extract_ci_list,
)


HARTREE_TO_EV = 27.211386245988

BASIS = "def2-qzvppd"

NCORE = 9
NCAS = 15

TARGET_LABELS = [
    "Fe 3d",
    "Fe 4d",
    "Fe 4s",
    "Fe 4p",
    "H 1s",
]


SYSTEMS = (
    {
        "key": "neutral_quartet",
        "charge": 0,
        "spin_2s": 3,
        "nelecas": (6, 3),
        "ground_2s": 3,
        "ground_mult": 4,
        "spin_manifolds": (
            (2, 2),   # 2 roots, doublet
            (4, 4),   # 4 roots, quartet
            (2, 6),   # 2 roots, sextet
        ),
        "center_chk": (
            "neutral_quartet__x2c_rohf.chk"
        ),
    },
    {
        "key": "anion_quintet",
        "charge": -1,
        "spin_2s": 4,
        "nelecas": (7, 3),
        "ground_2s": 4,
        "ground_mult": 5,
        "spin_manifolds": (
            (2, 3),   # 2 roots, triplet
            (4, 5),   # 4 roots, quintet
            (2, 7),   # 2 roots, septet
        ),
        "center_chk": (
            "anion_quintet__x2c_rohf.chk"
        ),
    },
)


def load_pec():
    path = (
        PEC_DIR
        / "combined.json"
    )

    if not path.is_file():
        raise RuntimeError(
            f"Missing PEC result: {path}"
        )

    return json.loads(
        path.read_text()
    )


def build_molecule(
    system,
    R,
):
    return gto.M(
        atom=[
            [
                "Fe",
                (
                    0.0,
                    0.0,
                    0.0,
                ),
            ],
            [
                "H",
                (
                    0.0,
                    0.0,
                    R,
                ),
            ],
        ],
        unit="Angstrom",
        basis=BASIS,
        charge=system["charge"],
        spin=system["spin_2s"],
        symmetry="C1",
        verbose=4,
        max_memory=60000,
    )


def run_x2c_rohf(
    system,
    R,
):
    mol = build_molecule(
        system,
        R,
    )

    mf = scf.ROHF(
        mol
    ).x2c()

    mf.max_memory = 60000
    mf.max_cycle = 100
    mf.conv_tol = 1.0e-11
    mf.conv_tol_grad = 1.0e-7

    mf.chkfile = str(
        OUT
        / (
            f"{system['key']}"
            f"__R{R:.6f}"
            "__x2c_rohf.chk"
        )
    )

    center_chk = (
        CENTER_DIR
        / system[
            "center_chk"
        ]
    )

    if not center_chk.is_file():
        raise RuntimeError(
            f"Missing X2C center checkpoint: {center_chk}"
        )

    dm0 = (
        mf.init_guess_by_chkfile(
            str(center_chk),
            project=True,
        )
    )

    mf.kernel(
        dm0=dm0
    )

    if not mf.converged:
        raise RuntimeError(
            f"{system['key']}: X2C-ROHF did not converge."
        )

    ss, mult = (
        mf.spin_square()
    )

    expected_s2 = (
        system["spin_2s"]
        / 2
        * (
            system["spin_2s"]
            / 2
            + 1
        )
    )

    print()
    print(
        "X2C-ROHF RESULT"
    )
    print(
        f"  R       = {R:.8f} A"
    )
    print(
        f"  E       = {mf.e_tot:.12f} Eh"
    )
    print(
        f"  <S^2>   = {float(ss):.10f}"
    )
    print(
        f"  mult    = {float(mult):.10f}"
    )

    if abs(
        float(ss)
        - expected_s2
    ) > 1.0e-8:
        raise RuntimeError(
            f"{system['key']}: wrong ROHF spin."
        )

    return (
        mol,
        mf,
    )


def unified_active_space(
    mol,
    mf,
):
    C = np.asarray(
        mf.mo_coeff
    )

    S = np.asarray(
        mf.get_ovlp()
    )

    labels = mol.ao_labels()

    target_idx = (
        mol.search_ao_label(
            TARGET_LABELS
        )
    )

    target_idx = np.asarray(
        target_idx,
        dtype=int,
    )

    print()
    print(
        "TARGET AO FUNCTIONS"
    )

    for idx in target_idx:
        print(
            f"  {idx:4d}  {labels[idx]}"
        )

    if len(
        target_idx
    ) != NCAS:
        raise RuntimeError(
            "Expected exactly "
            f"{NCAS} target AOs, "
            f"found {len(target_idx)}."
        )

    Ccore = C[
        :,
        :NCORE
    ]

    Cpost = C[
        :,
        NCORE:
    ]

    Stt = S[
        np.ix_(
            target_idx,
            target_idx,
        )
    ]

    Stt_inv = np.linalg.inv(
        Stt
    )

    cross = (
        Cpost.T
        @ S[
            :,
            target_idx
        ]
    )

    projector = (
        cross
        @ Stt_inv
        @ cross.T
    )

    projector = (
        0.5
        * (
            projector
            + projector.T
        )
    )

    eigvals, eigvecs = (
        np.linalg.eigh(
            projector
        )
    )

    order = np.argsort(
        eigvals
    )[::-1]

    eigvals = eigvals[
        order
    ]

    eigvecs = eigvecs[
        :,
        order
    ]

    Cpost_rot = (
        Cpost
        @ eigvecs
    )

    Cnew = np.hstack(
        (
            Ccore,
            Cpost_rot,
        )
    )

    orth_error = float(
        np.max(
            np.abs(
                Cnew.T
                @ S
                @ Cnew
                - np.eye(
                    Cnew.shape[1]
                )
            )
        )
    )

    print()
    print(
        "UNIFIED ACTIVE PROJECTOR"
    )
    print(
        "  top eigenvalues:"
    )

    print(
        np.array2string(
            eigvals[
                :20
            ],
            precision=10,
        )
    )

    print(
        "  lambda_15 =",
        f"{eigvals[14]:.10f}",
    )

    print(
        "  lambda_16 =",
        f"{eigvals[15]:.10e}",
    )

    print(
        "  MO orthogonality error =",
        f"{orth_error:.3e}",
    )

    if orth_error > 1.0e-8:
        raise RuntimeError(
            "Unified MO basis is not orthonormal."
        )

    if eigvals[14] < 0.02:
        raise RuntimeError(
            "15th active projector direction "
            "is unexpectedly weak."
        )

    return (
        Cnew,
        eigvals,
    )


def ground_sa4_casscf(
    mol,
    mf,
    mo_start,
    system,
):
    nelecas = (
        system[
            "nelecas"
        ]
    )

    spin_2s = (
        system[
            "ground_2s"
        ]
    )

    target_s2 = (
        spin_2s
        / 2
        * (
            spin_2s
            / 2
            + 1
        )
    )

    solver = (
        fci.direct_spin1.FCI(
            mol
        )
    )

    solver.spin = spin_2s
    solver.nroots = 4
    solver.conv_tol = 1.0e-10
    solver.max_cycle = 500
    solver.max_space = 60
    solver.pspace_size = 2000
    solver.davidson_only = True

    solver = (
        fci.addons.fix_spin(
            solver,
            shift=0.2,
            ss=target_s2,
        )
    )

    mc = mcscf.CASSCF(
        mf,
        NCAS,
        nelecas,
        ncore=NCORE,
    )

    mc.fcisolver = solver

    mc.state_average_(
        [
            0.25,
            0.25,
            0.25,
            0.25,
        ]
    )

    mc.max_memory = 60000
    #
    # Keep the original physical convergence thresholds.
    # The first neutral QZ/X2C run approached the stationary
    # point but exhausted 60 macro iterations.  Allow a longer
    # continuation rather than loosening conv_tol_grad.
    #
    mc.max_cycle_macro = 180
    mc.max_cycle_micro = 8
    mc.conv_tol = 1.0e-8
    mc.conv_tol_grad = 2.0e-5

    mc.chkfile = str(
        OUT
        / (
            f"{system['key']}"
            "__sa4_x2c_casscf.chk"
        )
    )

    print()
    print(
        "=" * 120
    )
    print(
        "GROUND-SPIN SA4-CASSCF"
    )
    print(
        "=" * 120
    )

    #
    # Restart directly from the latest optimized CASSCF orbitals
    # when a previous incomplete run left a checkpoint.  This is
    # only an optimizer restart; active space, roots, state weights,
    # Hamiltonian and convergence thresholds remain unchanged.
    #
    mo_initial = mo_start

    restart_path = Path(
        mc.chkfile
    )

    if restart_path.is_file():
        try:
            mo_restart = np.asarray(
                chkfile.load(
                    str(restart_path),
                    "mcscf/mo_coeff",
                )
            )

            if mo_restart.shape != mo_start.shape:
                raise RuntimeError(
                    "Restart MO shape mismatch: "
                    f"{mo_restart.shape} vs {mo_start.shape}"
                )

            S = np.asarray(
                mf.get_ovlp()
            )

            orth_error = float(
                np.max(
                    np.abs(
                        mo_restart.T
                        @ S
                        @ mo_restart
                        - np.eye(
                            mo_restart.shape[1]
                        )
                    )
                )
            )

            print(
                "CASSCF restart checkpoint:",
                restart_path,
            )

            print(
                "restart MO orthogonality error =",
                f"{orth_error:.3e}",
            )

            if orth_error > 1.0e-8:
                raise RuntimeError(
                    "Restart CASSCF orbitals are not "
                    "orthonormal."
                )

            mo_initial = mo_restart

            print(
                "Restarting SA4-CASSCF from saved "
                "mcscf/mo_coeff."
            )

        except Exception as exc:
            raise RuntimeError(
                "Existing CASSCF checkpoint could not "
                "be used safely for restart."
            ) from exc

    else:
        print(
            "No previous CASSCF checkpoint; "
            "starting from unified active-space orbitals."
        )

    mc.kernel(
        mo_initial
    )

    if not mc.converged:
        raise RuntimeError(
            f"{system['key']}: "
            "SA4-CASSCF did not converge."
        )

    energies = np.asarray(
        mc.e_states,
        dtype=float,
    )

    print()
    print(
        "SA4-CASSCF energies:"
    )

    for i, energy in enumerate(
        energies
    ):
        print(
            f"  root {i}: "
            f"{energy:.12f} Eh"
        )

    return mc


def make_spin_solver(
    mol,
    multiplicity,
    nroots,
):
    spin_2s = (
        multiplicity
        - 1
    )

    target_s2 = (
        spin_2s
        / 2
        * (
            spin_2s
            / 2
            + 1
        )
    )

    solver = (
        fci.direct_spin1.FCI(
            mol
        )
    )

    solver.spin = spin_2s
    solver.nroots = nroots
    solver.conv_tol = 1.0e-10
    solver.max_cycle = 500
    solver.max_space = 80
    solver.pspace_size = 2000
    solver.davidson_only = True

    solver = (
        fci.addons.fix_spin(
            solver,
            shift=0.2,
            ss=target_s2,
        )
    )

    solver.spin = spin_2s
    solver.nroots = nroots

    return solver


def mixed_spin_casci(
    mol,
    mf,
    mo,
    system,
):
    solvers = []

    total_states = 0

    print()
    print(
        "=" * 120
    )
    print(
        "MIXED-SPIN CASCI FOR SOC"
    )
    print(
        "=" * 120
    )

    for (
        nroots,
        multiplicity,
    ) in system[
        "spin_manifolds"
    ]:
        print(
            f"  multiplicity {multiplicity}: "
            f"{nroots} roots"
        )

        solvers.append(
            make_spin_solver(
                mol,
                multiplicity,
                nroots,
            )
        )

        total_states += (
            nroots
        )

    weights = (
        np.ones(
            total_states
        )
        / total_states
    )

    mc = mcscf.CASCI(
        mf,
        NCAS,
        system[
            "nelecas"
        ],
        ncore=NCORE,
    )

    mc.max_memory = 60000

    mcscf.state_average_mix_(
        mc,
        solvers,
        weights,
    )

    mc.kernel(
        mo
    )

    ci_list = (
        extract_ci_list(
            mc
        )
    )

    if len(
        ci_list
    ) != total_states:
        raise RuntimeError(
            "Unexpected number of extracted "
            f"CI states: {len(ci_list)} "
            f"vs {total_states}"
        )

    print()
    print(
        "SPIN-FREE CASCI STATES"
    )

    emin = min(
        state[4]
        for state in ci_list
    )

    for i, state in enumerate(
        ci_list
    ):
        print(
            f"  I={i:2d}  "
            f"2S={state[2]:d}  "
            f"2MS={state[3]:d}  "
            f"E={state[4]:.12f} Eh  "
            f"dE={(state[4]-emin)*HARTREE_TO_EV:.6f} eV"
        )

    expected_spin = (
        system[
            "ground_2s"
        ]
    )

    lowest = min(
        ci_list,
        key=lambda x: x[4],
    )

    print()
    print(
        "lowest spin-free 2S =",
        lowest[2],
    )

    if lowest[2] != expected_spin:
        raise RuntimeError(
            "A competing multiplicity is below "
            "the assumed ground-spin state. "
            "Stop before interpreting SOC."
        )

    return (
        mc,
        ci_list,
    )


def ground_manifold_center(
    ci_list,
    ground_2s,
):
    energies = sorted(
        state[4]
        for state in ci_list
        if state[2] == ground_2s
    )

    if len(
        energies
    ) < 2:
        raise RuntimeError(
            "Need at least two ground-spin roots."
        )

    center = (
        0.5
        * (
            energies[0]
            + energies[1]
        )
    )

    spread_ev = (
        energies[1]
        - energies[0]
    ) * HARTREE_TO_EV

    return (
        center,
        spread_ev,
        energies,
    )


def averaged_ground_density(
    mol,
    mo,
    ci_list,
    ground_2s,
):
    states = sorted(
        (
            state
            for state in ci_list
            if state[2]
            == ground_2s
        ),
        key=lambda x: x[4],
    )

    if len(
        states
    ) < 2:
        raise RuntimeError(
            "Need two roots for averaged "
            "ground-manifold density."
        )

    active_dm = np.zeros(
        (
            NCAS,
            NCAS,
        )
    )

    for state in states[:2]:
        na = state[0]
        nb = state[1]

        dm = (
            fci.direct_spin1.make_rdm1(
                state[-1],
                NCAS,
                (
                    na,
                    nb,
                ),
            )
        )

        active_dm += (
            0.5
            * dm
        )

    dm_mo = np.zeros(
        (
            mo.shape[1],
            mo.shape[1],
        )
    )

    dm_mo[
        :NCORE,
        :NCORE,
    ] = (
        2.0
        * np.eye(
            NCORE
        )
    )

    dm_mo[
        NCORE:NCORE + NCAS,
        NCORE:NCORE + NCAS,
    ] = active_dm

    dm_ao = (
        mo
        @ dm_mo
        @ mo.T
    )

    print()
    print(
        "SOC SOMF density trace:",
        np.trace(
            dm_ao
            @ mol.intor_symmetric(
                "int1e_ovlp"
            )
        ),
    )

    return dm_ao


def run_siso(
    mol,
    mf,
    mo,
    nelecas,
    ci_list,
    dmao,
    label,
):
    mf_siso = copy.copy(
        mf
    )

    mf_siso.mo_coeff = mo

    siso = FCISISO(
        mol,
        mf_siso,
        cas=(
            NCAS,
            nelecas,
        ),
    )

    siso.ci = ci_list

    print()
    print(
        "=" * 120
    )
    print(
        "SISO:",
        label
    )
    print(
        "=" * 120
    )

    energies = np.asarray(
        siso.kernel_we(
            dmao=dmao,
            amfi=True,
        ),
        dtype=float,
    )

    return energies


def run_system(
    system,
    R,
):
    print()
    print(
        "#" * 120
    )
    print(
        system["key"],
        f"R = {R:.8f} A"
    )
    print(
        "#" * 120
    )

    mol, mf = (
        run_x2c_rohf(
            system,
            R,
        )
    )

    mo_unified, eigvals = (
        unified_active_space(
            mol,
            mf,
        )
    )

    mc_sa4 = (
        ground_sa4_casscf(
            mol,
            mf,
            mo_unified,
            system,
        )
    )

    mo_final = np.asarray(
        mc_sa4.mo_coeff
    )

    mc_mix, ci_list = (
        mixed_spin_casci(
            mol,
            mf,
            mo_final,
            system,
        )
    )

    center, spread_ev, ground_energies = (
        ground_manifold_center(
            ci_list,
            system[
                "ground_2s"
            ],
        )
    )

    print()
    print(
        "GROUND SPIN-FREE MANIFOLD"
    )

    print(
        f"  center = {center:.12f} Eh"
    )

    print(
        f"  root-pair spread = "
        f"{spread_ev*1000.0:.6f} meV"
    )

    dmao = (
        averaged_ground_density(
            mol,
            mo_final,
            ci_list,
            system[
                "ground_2s"
            ],
        )
    )

    same_spin_ci = [
        state
        for state in ci_list
        if state[2]
        == system[
            "ground_2s"
        ]
    ]

    same_spin_soc = (
        run_siso(
            mol,
            mf,
            mo_final,
            system[
                "nelecas"
            ],
            same_spin_ci,
            dmao,
            "ground multiplicity only",
        )
    )

    all_spin_soc = (
        run_siso(
            mol,
            mf,
            mo_final,
            system[
                "nelecas"
            ],
            ci_list,
            dmao,
            "ground + adjacent multiplicities",
        )
    )

    same_shift_eh = (
        float(
            np.min(
                same_spin_soc
            )
        )
        - center
    )

    all_shift_eh = (
        float(
            np.min(
                all_spin_soc
            )
        )
        - center
    )

    same_shift_ev = (
        same_shift_eh
        * HARTREE_TO_EV
    )

    all_shift_ev = (
        all_shift_eh
        * HARTREE_TO_EV
    )

    adjacent_effect_ev = (
        all_shift_ev
        - same_shift_ev
    )

    print()
    print(
        "=" * 120
    )
    print(
        system["key"],
        "SOC RESULT"
    )
    print(
        "=" * 120
    )

    print(
        f"spin-free manifold center = "
        f"{center:.12f} Eh"
    )

    print(
        "SOC ground shift "
        "(same multiplicity) = "
        f"{same_shift_ev:+.6f} eV"
    )

    print(
        "SOC ground shift "
        "(all included spins) = "
        f"{all_shift_ev:+.6f} eV"
    )

    print(
        "adjacent-spin contribution = "
        f"{adjacent_effect_ev:+.6f} eV"
    )

    result = {
        "system": (
            system[
                "key"
            ]
        ),
        "R_angstrom": (
            R
        ),
        "basis": BASIS,
        "ncore": NCORE,
        "ncas": NCAS,
        "nelecas": list(
            system[
                "nelecas"
            ]
        ),
        "ground_2s": (
            system[
                "ground_2s"
            ]
        ),
        "projector_eigenvalues": (
            eigvals.tolist()
        ),
        "spin_free_ground_center_hartree": (
            center
        ),
        "spin_free_ground_pair_spread_ev": (
            spread_ev
        ),
        "same_spin_soc_shift_ev": (
            same_shift_ev
        ),
        "all_spin_soc_shift_ev": (
            all_shift_ev
        ),
        "adjacent_spin_effect_ev": (
            adjacent_effect_ev
        ),
        "spin_free_states": [
            {
                "na": int(
                    state[0]
                ),
                "nb": int(
                    state[1]
                ),
                "spin_2s": int(
                    state[2]
                ),
                "ms_2": int(
                    state[3]
                ),
                "energy_hartree": float(
                    state[4]
                ),
            }
            for state in ci_list
        ],
        "same_spin_soc_energies_hartree": (
            same_spin_soc.tolist()
        ),
        "all_spin_soc_energies_hartree": (
            all_spin_soc.tolist()
        ),
    }

    output = (
        OUT
        / f"{system['key']}.json"
    )

    output.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print(
        "saved:",
        output,
    )

    return result


def main():
    pec = load_pec()

    ea_before_soc = float(
        pec[
            "ea_v0_before_soc_ev"
        ]
    )

    neutral_R = float(
        pec[
            "systems"
        ][
            "neutral_quartet"
        ][
            "analysis"
        ][
            "Re_angstrom"
        ]
    )

    anion_R = float(
        pec[
            "systems"
        ][
            "anion_quintet"
        ][
            "analysis"
        ][
            "Re_angstrom"
        ]
    )

    print()
    print(
        "=" * 120
    )
    print(
        "INPUT FROM X2C-QZVPPD-CCSD(T) PEC"
    )
    print(
        "=" * 120
    )

    print(
        f"neutral Re = {neutral_R:.8f} A"
    )

    print(
        f"anion   Re = {anion_R:.8f} A"
    )

    print(
        f"EA_0 before SOC = "
        f"{ea_before_soc:.6f} eV"
    )

    def load_or_run(
        system,
        R,
    ):
        result_path = (
            OUT
            / f"{system['key']}.json"
        )

        if result_path.is_file():
            print()
            print(
                "=" * 120
            )
            print(
                "REUSING COMPLETE SOC RESULT:",
                system["key"],
            )
            print(
                result_path
            )
            print(
                "=" * 120
            )

            return json.loads(
                result_path.read_text()
            )

        print()
        print(
            "=" * 120
        )
        print(
            "NO COMPLETE SOC RESULT:",
            system["key"],
        )
        print(
            "Running system; existing CASSCF "
            "checkpoint will be used automatically "
            "when available."
        )
        print(
            "=" * 120
        )

        return run_system(
            system,
            R,
        )

    neutral = load_or_run(
        SYSTEMS[0],
        neutral_R,
    )

    anion = load_or_run(
        SYSTEMS[1],
        anion_R,
    )

    delta_soc_same = (
        neutral[
            "same_spin_soc_shift_ev"
        ]
        - anion[
            "same_spin_soc_shift_ev"
        ]
    )

    delta_soc_all = (
        neutral[
            "all_spin_soc_shift_ev"
        ]
        - anion[
            "all_spin_soc_shift_ev"
        ]
    )

    ea_same = (
        ea_before_soc
        + delta_soc_same
    )

    ea_all = (
        ea_before_soc
        + delta_soc_all
    )

    print()
    print(
        "=" * 120
    )
    print(
        "SOC-CORRECTED ELECTRON AFFINITY"
    )
    print(
        "=" * 120
    )

    print(
        f"EA before SOC = "
        f"{ea_before_soc:.6f} eV"
    )

    print()
    print(
        "Same-ground-multiplicity SOC:"
    )

    print(
        f"  neutral shift = "
        f"{neutral['same_spin_soc_shift_ev']:+.6f} eV"
    )

    print(
        f"  anion shift   = "
        f"{anion['same_spin_soc_shift_ev']:+.6f} eV"
    )

    print(
        f"  delta EA_SOC  = "
        f"{delta_soc_same:+.6f} eV"
    )

    print(
        f"  EA + SOC      = "
        f"{ea_same:.6f} eV"
    )

    print()
    print(
        "All included spin manifolds:"
    )

    print(
        f"  neutral shift = "
        f"{neutral['all_spin_soc_shift_ev']:+.6f} eV"
    )

    print(
        f"  anion shift   = "
        f"{anion['all_spin_soc_shift_ev']:+.6f} eV"
    )

    print(
        f"  delta EA_SOC  = "
        f"{delta_soc_all:+.6f} eV"
    )

    print(
        f"  EA + SOC      = "
        f"{ea_all:.6f} eV"
    )

    print()
    print(
        "Effect of adjacent multiplicities "
        "on delta EA_SOC = "
        f"{delta_soc_all-delta_soc_same:+.6f} eV"
    )

    summary = {
        "ea_before_soc_ev": (
            ea_before_soc
        ),
        "neutral": neutral,
        "anion": anion,
        "delta_soc_same_spin_ev": (
            delta_soc_same
        ),
        "delta_soc_all_spin_ev": (
            delta_soc_all
        ),
        "ea_with_same_spin_soc_ev": (
            ea_same
        ),
        "ea_with_all_spin_soc_ev": (
            ea_all
        ),
        "adjacent_spin_sensitivity_ev": (
            delta_soc_all
            - delta_soc_same
        ),
    }

    output = (
        OUT
        / "combined.json"
    )

    output.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print(
        "saved:",
        output,
    )


if __name__ == "__main__":
    main()
