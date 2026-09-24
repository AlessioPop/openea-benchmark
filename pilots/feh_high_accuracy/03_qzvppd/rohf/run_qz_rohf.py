from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from pyscf import (
    gto,
    scf,
)
from pyscf.scf import chkfile


TZ_DIR = Path(
    "/srv/storage/homes/analysis/dschmid/photodetachment/"
    "openea-pilot/feh_mr_nevpt2_unified"
)

OUT = Path(
    "/srv/storage/homes/analysis/dschmid/photodetachment/"
    "openea-pilot/feh_rohf_qzvppd"
)

BASIS = "def2-qzvppd"

HARTREE_TO_EV = 27.211386245988


SYSTEMS = (
    {
        "key": "neutral_quartet",
        "prefix": "FeH__",
        "expected_s2": 3.75,
        "expected_mult": 4.0,
    },
    {
        "key": "anion_quintet",
        "prefix": "FeH-__",
        "expected_s2": 6.0,
        "expected_mult": 5.0,
    },
)


def find_tz_checkpoint(system):
    hits = sorted(
        path
        for path in TZ_DIR.glob(
            "*__rohf.chk"
        )
        if path.name.startswith(
            system["prefix"]
        )
    )

    if len(hits) != 1:
        raise RuntimeError(
            f"{system['key']}: expected exactly one "
            f"TZVPPD ROHF checkpoint; found {len(hits)}"
        )

    return hits[0]


def orthonormalize(
    coefficients,
    overlap,
):
    metric = (
        coefficients.T
        @ overlap
        @ coefficients
    )

    eigvals, eigvecs = np.linalg.eigh(
        metric
    )

    if np.min(
        eigvals
    ) < 1.0e-10:
        raise RuntimeError(
            "Projected occupied subspace became "
            "linearly dependent."
        )

    transform = (
        eigvecs
        @ np.diag(
            eigvals ** -0.5
        )
        @ eigvecs.T
    )

    return (
        coefficients
        @ transform
    )


def occupied_subspace_overlap(
    mol_old,
    mo_old,
    occ_old,
    mol_new,
    mo_new,
    occ_new,
):
    nocc_old = int(
        np.count_nonzero(
            occ_old > 0
        )
    )

    nocc_new = int(
        np.count_nonzero(
            occ_new > 0
        )
    )

    if nocc_old != nocc_new:
        raise RuntimeError(
            "Occupied-space dimensions changed: "
            f"{nocc_old} -> {nocc_new}"
        )

    old_occ = (
        mo_old[
            :,
            :nocc_old
        ]
    )

    new_occ = (
        mo_new[
            :,
            :nocc_new
        ]
    )

    projected = (
        scf.addons.project_mo_nr2nr(
            mol_old,
            old_occ,
            mol_new,
        )
    )

    overlap = (
        mol_new.intor_symmetric(
            "int1e_ovlp"
        )
    )

    projected = orthonormalize(
        projected,
        overlap,
    )

    matrix = (
        projected.T
        @ overlap
        @ new_occ
    )

    singular_values = (
        np.linalg.svd(
            matrix,
            compute_uv=False,
        )
    )

    return singular_values


def build_qz_molecule(
    old_mol,
):
    #
    # old_mol._atom stores the already-resolved nuclear
    # coordinates in Bohr.  Build a fresh Mole so there is
    # no possibility that the old TZ basis survives internally.
    #
    mol = gto.M(
        atom=old_mol._atom,
        unit="Bohr",
        basis=BASIS,
        charge=old_mol.charge,
        spin=old_mol.spin,
        symmetry="C1",
        verbose=4,
        max_memory=60000,
    )

    return mol


def run_system(
    system,
):
    print()
    print(
        "=" * 120
    )
    print(
        system["key"]
    )
    print(
        "=" * 120
    )

    old_chk = (
        find_tz_checkpoint(
            system
        )
    )

    old_mol, old_data = (
        chkfile.load_scf(
            str(
                old_chk
            )
        )
    )

    old_mo = np.asarray(
        old_data[
            "mo_coeff"
        ]
    )

    old_occ = np.asarray(
        old_data[
            "mo_occ"
        ]
    )

    print(
        "TZ checkpoint:",
        old_chk,
    )

    print(
        "TZ basis functions:",
        old_mol.nao_nr(),
    )

    print(
        "TZ ROHF energy:",
        f"{float(old_data['e_tot']):.12f}",
        "Eh",
    )

    mol = build_qz_molecule(
        old_mol
    )

    print()
    print(
        "QZ basis:",
        BASIS,
    )

    print(
        "QZ basis functions:",
        mol.nao_nr(),
    )

    print(
        "charge:",
        mol.charge,
    )

    print(
        "2S:",
        mol.spin,
    )

    output_chk = (
        OUT
        / f"{system['key']}__rohf_qzvppd.chk"
    )

    mf = scf.ROHF(
        mol
    )

    mf.chkfile = str(
        output_chk
    )

    mf.max_memory = 60000
    mf.max_cycle = 100
    mf.conv_tol = 1.0e-11
    mf.conv_tol_grad = 1.0e-7

    #
    # PySCF projects the converged TZVPPD checkpoint density
    # into the new QZVPPD AO basis.
    #
    dm0 = (
        mf.init_guess_by_chkfile(
            str(
                old_chk
            ),
            project=True,
        )
    )

    print()
    print(
        "--- QZVPPD ROHF FROM PROJECTED TZVPPD STATE ---"
    )

    mf.kernel(
        dm0=dm0
    )

    if not mf.converged:
        print()
        print(
            "Conventional ROHF did not converge; "
            "switching to second-order Newton solver."
        )

        mf = mf.newton()

        mf.chkfile = str(
            output_chk
        )

        mf.max_cycle = 100
        mf.conv_tol = 1.0e-11
        mf.conv_tol_grad = 1.0e-7
        mf.max_memory = 60000

        mf.kernel(
            mo_coeff=mf._scf.mo_coeff,
            mo_occ=mf._scf.mo_occ,
        )

    if not mf.converged:
        raise RuntimeError(
            f"{system['key']}: QZVPPD ROHF "
            "did not converge."
        )

    ss, multiplicity = (
        mf.spin_square()
    )

    print()
    print(
        "QZVPPD ROHF RESULT"
    )

    print(
        "  E =",
        f"{mf.e_tot:.12f}",
        "Eh",
    )

    print(
        "  <S^2> =",
        f"{float(ss):.10f}",
    )

    print(
        "  multiplicity =",
        f"{float(multiplicity):.10f}",
    )

    if abs(
        float(ss)
        - system[
            "expected_s2"
        ]
    ) > 1.0e-8:
        raise RuntimeError(
            f"{system['key']}: wrong QZVPPD spin."
        )

    singular_values = (
        occupied_subspace_overlap(
            old_mol,
            old_mo,
            old_occ,
            mol,
            np.asarray(
                mf.mo_coeff
            ),
            np.asarray(
                mf.mo_occ
            ),
        )
    )

    print()
    print(
        "TZ -> QZ OCCUPIED-SUBSPACE CONTINUITY"
    )

    print(
        np.array2string(
            singular_values,
            precision=10,
        )
    )

    print(
        "minimum singular value:",
        f"{np.min(singular_values):.10f}",
    )

    if np.min(
        singular_values
    ) < 0.90:
        raise RuntimeError(
            f"{system['key']}: projected TZ and "
            "converged QZ occupied spaces differ strongly."
        )

    result = {
        "system": system[
            "key"
        ],
        "basis": BASIS,
        "basis_functions": int(
            mol.nao_nr()
        ),
        "charge": int(
            mol.charge
        ),
        "spin_2s": int(
            mol.spin
        ),
        "rohf_energy_hartree": float(
            mf.e_tot
        ),
        "s2": float(
            ss
        ),
        "multiplicity": float(
            multiplicity
        ),
        "occupied_subspace_singular_values": (
            singular_values.tolist()
        ),
        "occupied_subspace_min_sv": float(
            np.min(
                singular_values
            )
        ),
        "checkpoint": str(
            output_chk
        ),
    }

    path = (
        OUT
        / f"{system['key']}.json"
    )

    path.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print(
        "saved:",
        path,
    )

    return result


def main():
    results = {}

    for system in SYSTEMS:
        results[
            system[
                "key"
            ]
        ] = run_system(
            system
        )

    neutral = results[
        "neutral_quartet"
    ]

    anion = results[
        "anion_quintet"
    ]

    ea = (
        neutral[
            "rohf_energy_hartree"
        ]
        - anion[
            "rohf_energy_hartree"
        ]
    ) * HARTREE_TO_EV

    print()
    print(
        "=" * 120
    )
    print(
        "QZVPPD ROHF ELECTRONIC EA"
    )
    print(
        "=" * 120
    )

    print(
        f"EA_elec(ROHF/QZVPPD) = "
        f"{ea:.6f} eV"
    )

    print()
    print(
        "TZVPPD reference value:"
    )

    print(
        "EA_elec(ROHF/TZVPPD) = "
        "1.719087 eV"
    )

    print(
        "QZ - TZ shift = "
        f"{ea - 1.719087:+.6f} eV"
    )

    (
        OUT
        / "combined.json"
    ).write_text(
        json.dumps(
            results,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
