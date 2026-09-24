from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from pyscf import (
    fci,
    mcscf,
    mrpt,
    scf,
)
from pyscf.mcscf import chkfile as mc_chkfile
from pyscf.scf import chkfile as scf_chkfile


BASE = Path(
    "/srv/storage/homes/analysis/dschmid/photodetachment/"
    "openea-pilot/feh_mr_nevpt2_unified"
)

HARTREE_TO_EV = 27.211386245988
NROOTS = 4


SYSTEMS = (
    {
        "key": "neutral_quartet",
        "spin_2s": 3,
        "target_s2": 3.75,
    },
    {
        "key": "anion_quintet",
        "spin_2s": 4,
        "target_s2": 6.0,
    },
)


def find_rohf_checkpoint(
    key,
):
    prefix = (
        "FeH__"
        if key == "neutral_quartet"
        else "FeH-__"
    )

    hits = sorted(
        path
        for path in BASE.glob(
            "*__rohf.chk"
        )
        if path.name.startswith(
            prefix
        )
    )

    if len(hits) != 1:
        raise RuntimeError(
            f"{key}: expected exactly one ROHF checkpoint; "
            f"found {len(hits)}"
        )

    return hits[0]


def sa_energies_from_json(
    key,
):
    path = (
        BASE
        / f"{key}.json"
    )

    if not path.is_file():
        raise RuntimeError(
            f"Missing previous result JSON: {path}"
        )

    data = json.loads(
        path.read_text()
    )

    energies = []

    for root in data[
        "roots"
    ]:
        if (
            "sa_casscf_energy_hartree"
            in root
        ):
            value = root[
                "sa_casscf_energy_hartree"
            ]

        elif (
            "casscf_energy_hartree"
            in root
        ):
            value = root[
                "casscf_energy_hartree"
            ]

        else:
            raise RuntimeError(
                f"{key}: no SA-CASSCF energy "
                "stored for one root"
            )

        energies.append(
            float(
                value
            )
        )

    if len(energies) != NROOTS:
        raise RuntimeError(
            f"{key}: expected {NROOTS} SA roots; "
            f"found {len(energies)}"
        )

    return np.asarray(
        energies,
        dtype=float,
    )


def lowest_manifold(
    energies,
    *,
    tolerance_mev=1.0,
):
    energies = np.asarray(
        energies,
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

    selected = energies[
        members
    ]

    emin = float(
        np.min(
            selected
        )
    )

    emax = float(
        np.max(
            selected
        )
    )

    return {
        "members": [
            int(index)
            for index
            in members
        ],
        "center_hartree": (
            0.5
            * (
                emin
                + emax
            )
        ),
        "spread_mev": (
            (
                emax
                - emin
            )
            * HARTREE_TO_EV
            * 1000.0
        ),
        "threshold_mev": (
            tolerance_mev
        ),
    }


def run_system(
    system,
):
    key = system[
        "key"
    ]

    print()
    print(
        "=" * 120
    )
    print(
        key
    )
    print(
        "=" * 120
    )

    casscf_chk = (
        BASE
        / (
            key
            + "__sa_casscf.chk"
        )
    )

    if not casscf_chk.is_file():
        raise RuntimeError(
            f"Missing CASSCF checkpoint: "
            f"{casscf_chk}"
        )

    #
    # This checkpoint does NOT contain CI vectors, but it
    # does contain the fully optimized SA-CASSCF orbitals.
    #
    mol_mc, mcdata = (
        mc_chkfile.load_mcscf(
            str(
                casscf_chk
            )
        )
    )

    print(
        "MCSCF checkpoint keys:",
        sorted(
            mcdata.keys()
        ),
    )

    for required in (
        "mo_coeff",
        "ncore",
        "ncas",
        "nelecas",
    ):
        if required not in mcdata:
            raise RuntimeError(
                f"{key}: checkpoint lacks "
                f"mcscf/{required}"
            )

    mo = np.asarray(
        mcdata[
            "mo_coeff"
        ]
    )

    ncore = int(
        mcdata[
            "ncore"
        ]
    )

    ncas = int(
        mcdata[
            "ncas"
        ]
    )

    nelecas = (
        mcdata[
            "nelecas"
        ]
    )

    if isinstance(
        nelecas,
        np.ndarray,
    ):
        nelecas = (
            nelecas.tolist()
        )

    if isinstance(
        nelecas,
        list,
    ):
        nelecas = tuple(
            int(value)
            for value
            in nelecas
        )

    print(
        "ncore:",
        ncore,
    )

    print(
        "ncas:",
        ncas,
    )

    print(
        "nelecas:",
        nelecas,
    )

    expected_sa = (
        sa_energies_from_json(
            key
        )
    )

    print()
    print(
        "SA-CASSCF reference spectrum:"
    )

    for index, energy in enumerate(
        np.sort(
            expected_sa
        )
    ):
        print(
            f"  {index}: "
            f"{energy:.12f} Eh"
        )

    rohf_chk = (
        find_rohf_checkpoint(
            key
        )
    )

    mol_scf, scfdata = (
        scf_chkfile.load_scf(
            str(
                rohf_chk
            )
        )
    )

    mf = scf.ROHF(
        mol_scf
    )

    mf.mo_coeff = np.asarray(
        scfdata[
            "mo_coeff"
        ]
    )

    mf.mo_occ = np.asarray(
        scfdata[
            "mo_occ"
        ]
    )

    mf.mo_energy = np.asarray(
        scfdata[
            "mo_energy"
        ]
    )

    mf.e_tot = float(
        scfdata[
            "e_tot"
        ]
    )

    mf.converged = True
    mf.max_memory = 60000

    casci = mcscf.CASCI(
        mf,
        ncas,
        nelecas,
        ncore=ncore,
    )

    solver = (
        fci.direct_spin1.FCI(
            mol_scf
        )
    )

    solver.spin = system[
        "spin_2s"
    ]

    solver.nroots = NROOTS

    solver.conv_tol = (
        1.0e-10
    )

    #
    # The failed run used:
    #
    #   max_cycle = 100
    #   max_space = 12
    #
    # For these four large and partly near-degenerate roots
    # we deliberately enlarge the Davidson search space.
    #
    solver.max_cycle = 500
    solver.max_space = 80
    solver.pspace_size = 2000
    solver.davidson_only = True

    solver = (
        fci.addons.fix_spin(
            solver,
            shift=0.2,
            ss=system[
                "target_s2"
            ],
        )
    )

    #
    # Re-apply the numerical controls to the wrapped solver.
    #
    solver.spin = system[
        "spin_2s"
    ]

    solver.nroots = NROOTS
    solver.conv_tol = 1.0e-10
    solver.max_cycle = 500
    solver.max_space = 80
    solver.pspace_size = 2000
    solver.davidson_only = True

    casci.fcisolver = solver

    casci.max_memory = 60000

    #
    # Do not perform a first-root-dependent post-CASCI
    # canonicalization.  We want the SA-CASSCF orbital basis
    # exactly as stored in the checkpoint.
    #
    casci.canonicalization = False

    casci.verbose = 4

    print()
    print(
        "--- LARGE-SPACE FOUR-ROOT CASCI ---"
    )

    print(
        "max_cycle =",
        solver.max_cycle,
    )

    print(
        "max_space =",
        solver.max_space,
    )

    print(
        "pspace_size =",
        solver.pspace_size,
    )

    #
    # No CI vectors were stored in the SA checkpoint, so this
    # solve necessarily starts from PySCF's internally generated
    # multi-root initial guesses.
    #
    casci.kernel(
        mo_coeff=mo,
    )

    converged = (
        casci.converged
    )

    print()
    print(
        "CASCI convergence flag:",
        converged,
    )

    if isinstance(
        converged,
        (
            tuple,
            list,
            np.ndarray,
        ),
    ):
        all_converged = all(
            bool(value)
            for value
            in converged
        )

    else:
        all_converged = bool(
            converged
        )

    if not all_converged:
        raise RuntimeError(
            f"{key}: CASCI did not converge. "
            "NEVPT2 WILL NOT BE RUN."
        )

    energies = np.asarray(
        casci.e_tot,
        dtype=float,
    ).reshape(
        -1
    )

    if (
        len(
            energies
        )
        != NROOTS
    ):
        raise RuntimeError(
            f"{key}: expected {NROOTS} CASCI roots; "
            f"obtained {len(energies)}"
        )

    #
    # Degenerate roots may swap labels.  Compare spectra,
    # not root numbers.
    #
    sorted_casci = np.sort(
        energies
    )

    sorted_sa = np.sort(
        expected_sa
    )

    differences = (
        sorted_casci
        - sorted_sa
    )

    print()
    print(
        "--- CASCI / SA-CASSCF SPECTRUM AGREEMENT ---"
    )

    for index in range(
        NROOTS
    ):
        delta_mev = (
            differences[
                index
            ]
            * HARTREE_TO_EV
            * 1000.0
        )

        print(
            f"{index}: "
            f"CASCI={sorted_casci[index]:.12f}  "
            f"SA={sorted_sa[index]:.12f}  "
            f"delta={delta_mev:+.6f} meV"
        )

    max_abs_delta = float(
        np.max(
            np.abs(
                differences
            )
        )
    )

    print()
    print(
        "max |CASCI-SA| =",
        f"{max_abs_delta:.3e} Eh",
    )

    if (
        max_abs_delta
        > 1.0e-7
    ):
        raise RuntimeError(
            f"{key}: converged CASCI does not reproduce "
            "the SA-CASSCF spectrum closely enough. "
            f"max difference = "
            f"{max_abs_delta:.3e} Eh. "
            "NEVPT2 WILL NOT BE RUN."
        )

    print()
    print(
        "CASCI ROOT REPRODUCTION: PASS"
    )

    #
    # Verify target spin directly from the converged CI vectors.
    #
    try:
        ss, mult = (
            casci.fcisolver
            .states_spin_square(
                casci.ci,
                ncas,
                nelecas,
            )
        )

        print()
        print(
            "CASCI SPIN CHECK"
        )

        for root in range(
            NROOTS
        ):
            print(
                f"  root {root}: "
                f"<S^2>={float(ss[root]):.8f}  "
                f"mult={float(mult[root]):.8f}"
            )

    except Exception as error:
        print()
        print(
            "Spin diagnostic unavailable:",
            repr(
                error
            ),
        )

    print()
    print(
        "--- ROOT-SPECIFIC SC-NEVPT2 ---"
    )

    nevpt_totals = []

    root_results = []

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
            energies[
                root
            ]
            + corr
        )

        nevpt_totals.append(
            total
        )

        root_results.append(
            {
                "root": root,
                "casci_energy_hartree": (
                    float(
                        energies[
                            root
                        ]
                    )
                ),
                "nevpt2_correlation_hartree": (
                    corr
                ),
                "casci_nevpt2_energy_hartree": (
                    total
                ),
            }
        )

        print(
            "  E(CASCI) =",
            f"{energies[root]:.12f} Eh",
        )

        print(
            "  E(NEVPT2 corr) =",
            f"{corr:.12f} Eh",
        )

        print(
            "  E(CASCI+NEVPT2) =",
            f"{total:.12f} Eh",
        )

    manifold = (
        lowest_manifold(
            nevpt_totals
        )
    )

    print()
    print(
        "LOWEST SC-NEVPT2 MANIFOLD"
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

    result = {
        "system": key,
        "ncore": ncore,
        "ncas": ncas,
        "nelecas": (
            list(
                nelecas
            )
            if isinstance(
                nelecas,
                tuple,
            )
            else nelecas
        ),
        "casci_converged": True,
        "max_casci_sa_difference_hartree": (
            max_abs_delta
        ),
        "roots": (
            root_results
        ),
        "lowest_nevpt2_manifold": (
            manifold
        ),
    }

    output = (
        BASE
        / (
            key
            + "__nevpt2_repaired_v2.json"
        )
    )

    output.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print()
    print(
        "saved:",
        output,
    )

    return result


def main():
    results = {}

    for system in SYSTEMS:
        result = run_system(
            system
        )

        results[
            system[
                "key"
            ]
        ] = result

    neutral = (
        results[
            "neutral_quartet"
        ][
            "lowest_nevpt2_manifold"
        ]
    )

    anion = (
        results[
            "anion_quintet"
        ][
            "lowest_nevpt2_manifold"
        ]
    )

    ea = (
        neutral[
            "center_hartree"
        ]
        - anion[
            "center_hartree"
        ]
    ) * HARTREE_TO_EV

    print()
    print(
        "=" * 120
    )
    print(
        "VALIDATED UNIFIED SC-NEVPT2 EA"
    )
    print(
        "=" * 120
    )

    print(
        "neutral:",
        neutral,
    )

    print(
        "anion:",
        anion,
    )

    print()
    print(
        f"EA_elec = {ea:.6f} eV"
    )

    combined = (
        BASE
        / "nevpt2_repaired_v2_combined.json"
    )

    combined.write_text(
        json.dumps(
            results,
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
