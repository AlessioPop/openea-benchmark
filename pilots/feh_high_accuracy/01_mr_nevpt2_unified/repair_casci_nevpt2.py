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
        "spin": 3,
        "s2": 3.75,
    },
    {
        "key": "anion_quintet",
        "spin": 4,
        "s2": 6.0,
    },
)


def find_rohf_checkpoint(key):
    if key == "neutral_quartet":
        prefix = "FeH__"
    else:
        prefix = "FeH-__"

    hits = sorted(
        p for p in BASE.glob("*__rohf.chk")
        if p.name.startswith(prefix)
    )

    if len(hits) != 1:
        raise RuntimeError(
            f"{key}: expected one ROHF checkpoint, "
            f"found {len(hits)}"
        )

    return hits[0]


def normalize_ci(ci):
    if isinstance(ci, (list, tuple)):
        roots = list(ci)
    elif isinstance(ci, np.ndarray):
        if ci.dtype == object:
            roots = list(ci)
        elif ci.ndim >= 3 and ci.shape[0] == NROOTS:
            roots = [
                np.asarray(ci[i])
                for i in range(NROOTS)
            ]
        else:
            raise RuntimeError(
                "Stored CI array does not look like "
                f"{NROOTS} state-average roots: "
                f"shape={ci.shape}, dtype={ci.dtype}"
            )
    else:
        raise RuntimeError(
            f"Unsupported stored CI type: {type(ci)}"
        )

    if len(roots) != NROOTS:
        raise RuntimeError(
            f"Expected {NROOTS} CI roots, found {len(roots)}"
        )

    return [
        np.asarray(root)
        for root in roots
    ]


def lowest_manifold(energies, tolerance_mev=1.0):
    energies = np.asarray(
        energies,
        dtype=float,
    )

    emin = float(
        np.min(energies)
    )

    delta_mev = (
        energies - emin
    ) * HARTREE_TO_EV * 1000.0

    members = np.where(
        delta_mev <= tolerance_mev
    )[0]

    selected = energies[
        members
    ]

    center = 0.5 * (
        float(np.min(selected))
        + float(np.max(selected))
    )

    spread_mev = (
        float(np.max(selected))
        - float(np.min(selected))
    ) * HARTREE_TO_EV * 1000.0

    return {
        "members": [
            int(i)
            for i in members
        ],
        "center_hartree": center,
        "spread_mev": spread_mev,
    }


def run_system(system):
    key = system["key"]

    print()
    print("=" * 110)
    print(key)
    print("=" * 110)

    casscf_chk = (
        BASE
        / f"{key}__sa_casscf.chk"
    )

    json_path = (
        BASE
        / f"{key}.json"
    )

    if not casscf_chk.is_file():
        raise RuntimeError(
            f"Missing CASSCF checkpoint: {casscf_chk}"
        )

    if not json_path.is_file():
        raise RuntimeError(
            f"Missing result JSON: {json_path}"
        )

    mol, mcdata = (
        mc_chkfile.load_mcscf(
            str(casscf_chk)
        )
    )

    old_result = json.loads(
        json_path.read_text()
    )

    print(
        "MCSCF checkpoint keys:",
        sorted(mcdata.keys()),
    )

    for required in (
        "mo_coeff",
        "ci",
        "ncore",
        "ncas",
        "nelecas",
    ):
        if required not in mcdata:
            raise RuntimeError(
                f"Checkpoint lacks mcscf/{required}"
            )

    mo = np.asarray(
        mcdata["mo_coeff"]
    )

    ci0 = normalize_ci(
        mcdata["ci"]
    )

    ncore = int(
        mcdata["ncore"]
    )

    ncas = int(
        mcdata["ncas"]
    )

    nelecas_raw = mcdata[
        "nelecas"
    ]

    if isinstance(
        nelecas_raw,
        np.ndarray,
    ):
        nelecas_raw = (
            nelecas_raw.tolist()
        )

    if isinstance(
        nelecas_raw,
        list,
    ):
        nelecas_raw = tuple(
            int(x)
            for x in nelecas_raw
        )

    nelecas = nelecas_raw

    print(
        "ncore:",
        ncore,
        "ncas:",
        ncas,
        "nelecas:",
        nelecas,
    )

    print(
        "stored CI root shapes:",
        [
            root.shape
            for root in ci0
        ],
    )

    rohf_chk = (
        find_rohf_checkpoint(
            key
        )
    )

    mol_scf, scfdata = (
        scf_chkfile.load_scf(
            str(rohf_chk)
        )
    )

    mf = scf.ROHF(
        mol_scf
    )

    mf.mo_coeff = np.asarray(
        scfdata["mo_coeff"]
    )

    mf.mo_occ = np.asarray(
        scfdata["mo_occ"]
    )

    mf.mo_energy = np.asarray(
        scfdata["mo_energy"]
    )

    mf.e_tot = float(
        scfdata["e_tot"]
    )

    mf.converged = True
    mf.max_memory = 60000

    casci = mcscf.CASCI(
        mf,
        ncas,
        nelecas,
        ncore=ncore,
    )

    solver = fci.direct_spin1.FCI(
        mol_scf
    )

    solver.spin = system[
        "spin"
    ]

    solver.nroots = NROOTS

    solver.conv_tol = 1.0e-10
    solver.max_cycle = 300

    #
    # The previous failure used only max_space=12.
    # Four close multi-million-determinant roots need a
    # substantially larger Davidson subspace.
    #
    solver.max_space = 50

    solver = fci.addons.fix_spin(
        solver,
        shift=0.2,
        ss=system["s2"],
    )

    solver.nroots = NROOTS
    solver.conv_tol = 1.0e-10
    solver.max_cycle = 300
    solver.max_space = 50

    casci.fcisolver = solver
    casci.max_memory = 60000
    casci.verbose = 4

    #
    # Do not perform an extra first-root-based canonicalization
    # after the multi-root CI solve.  NEVPT will canonicalize
    # the requested root itself.
    #
    casci.canonicalization = False

    print()
    print(
        "--- RESTARTED MULTI-ROOT CASCI "
        "FROM SA-CASSCF CI VECTORS ---"
    )

    casci.kernel(
        mo_coeff=mo,
        ci0=ci0,
    )

    converged = casci.converged

    if isinstance(
        converged,
        (list, tuple, np.ndarray),
    ):
        all_converged = all(
            bool(x)
            for x in converged
        )
    else:
        all_converged = bool(
            converged
        )

    print()
    print(
        "CASCI converged:",
        converged,
    )

    if not all_converged:
        raise RuntimeError(
            f"{key}: restarted CASCI still did not converge. "
            "NEVPT2 WILL NOT BE RUN."
        )

    casci_energies = np.asarray(
        casci.e_tot,
        dtype=float,
    ).reshape(-1)

    if len(casci_energies) != NROOTS:
        raise RuntimeError(
            f"{key}: expected {NROOTS} CASCI energies, "
            f"got {len(casci_energies)}"
        )

    expected = np.asarray(
        [
            root[
                "sa_casscf_energy_hartree"
            ]
            for root
            in old_result[
                "roots"
            ]
        ],
        dtype=float,
    )

    print()
    print(
        "--- CASCI / SA-CASSCF ROOT AGREEMENT ---"
    )

    differences = []

    for root in range(
        NROOTS
    ):
        diff_eh = (
            casci_energies[root]
            - expected[root]
        )

        diff_mev = (
            diff_eh
            * HARTREE_TO_EV
            * 1000.0
        )

        differences.append(
            abs(diff_eh)
        )

        print(
            f"root {root}:",
            f"CASCI={casci_energies[root]:.12f}",
            f"SA={expected[root]:.12f}",
            f"delta={diff_mev:+.6f} meV",
        )

    max_difference = max(
        differences
    )

    #
    # Same orbitals + converged FCI should reproduce the
    # state-averaged CASSCF root energies essentially exactly.
    #
    if max_difference > 1.0e-7:
        raise RuntimeError(
            f"{key}: separated CASCI does not reproduce "
            "SA-CASSCF roots closely enough; "
            f"max |delta|={max_difference:.3e} Eh. "
            "NEVPT2 WILL NOT BE RUN."
        )

    print()
    print(
        "CASCI ROOT REPRODUCTION: PASS"
    )

    print()
    print(
        "--- ROOT-SPECIFIC SC-NEVPT2 ---"
    )

    totals = []

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
            casci_energies[root]
            + corr
        )

        totals.append(
            total
        )

        root_results.append(
            {
                "root": root,
                "casci_energy_hartree": float(
                    casci_energies[root]
                ),
                "nevpt2_correlation_hartree": corr,
                "casci_nevpt2_energy_hartree": total,
            }
        )

        print(
            f"E(CASCI+NEVPT2) = "
            f"{total:.12f} Eh"
        )

    manifold = lowest_manifold(
        totals
    )

    print()
    print(
        "LOWEST NEVPT2 MANIFOLD"
    )

    print(
        "roots:",
        manifold[
            "members"
        ],
    )

    print(
        "center:",
        f"{manifold['center_hartree']:.12f}",
        "Eh",
    )

    print(
        "spread:",
        f"{manifold['spread_mev']:.6f}",
        "meV",
    )

    result = {
        "system": key,
        "ncore": ncore,
        "ncas": ncas,
        "nelecas": list(
            nelecas
        ),
        "casci_converged": True,
        "max_casci_sa_difference_hartree": float(
            max_difference
        ),
        "roots": root_results,
        "lowest_nevpt2_manifold": manifold,
    }

    (
        BASE
        / f"{key}__nevpt2_repaired.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        )
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

    neutral = results[
        "neutral_quartet"
    ][
        "lowest_nevpt2_manifold"
    ]

    anion = results[
        "anion_quintet"
    ][
        "lowest_nevpt2_manifold"
    ]

    ea = (
        neutral[
            "center_hartree"
        ]
        - anion[
            "center_hartree"
        ]
    ) * HARTREE_TO_EV

    print()
    print("=" * 110)
    print(
        "REPAIRED UNIFIED SC-NEVPT2 EA"
    )
    print("=" * 110)

    print(
        "neutral manifold:",
        neutral,
    )

    print(
        "anion manifold:",
        anion,
    )

    print(
        f"EA_elec = {ea:.6f} eV"
    )

    (
        BASE
        / "nevpt2_repaired_combined.json"
    ).write_text(
        json.dumps(
            results,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
