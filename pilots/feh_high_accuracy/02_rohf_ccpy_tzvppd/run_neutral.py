from __future__ import annotations

from pathlib import Path
import copy
import json

import numpy as np

from pyscf import scf
from pyscf.scf import chkfile

from ccpy.drivers.driver import Driver


OUT = Path(
    "/srv/storage/homes/analysis/dschmid/photodetachment/"
    "openea-pilot/feh_ccpy_rohf_crcc23"
)

ROHF_DIR = Path(
    "/srv/storage/homes/analysis/dschmid/photodetachment/"
    "openea-pilot/feh_mr_nevpt2_unified"
)

HARTREE_TO_EV = 27.211386245988

#
# Freeze Fe 1s, 2s, 2p only:
#
#   1s = 1 spatial MO
#   2s = 1
#   2p = 3
#
# total = 5 spatial core orbitals = 10 electrons.
#
NFROZEN = 5


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): jsonable(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            jsonable(item)
            for item in value
        ]

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ) or value is None:
        return value

    return repr(value)


def find_checkpoint():
    hits = sorted(
        path
        for path in ROHF_DIR.glob(
            "FeH__*__rohf.chk"
        )
    )

    if len(hits) != 1:
        raise RuntimeError(
            "Expected exactly one neutral FeH ROHF "
            f"checkpoint; found {len(hits)}"
        )

    return hits[0]


def load_rohf():
    path = find_checkpoint()

    mol, data = chkfile.load_scf(
        str(path)
    )

    #
    # CCpy's PySCF interface unconditionally calls
    # symm.label_orb_symm().  Our production ROHF checkpoint
    # was generated with symmetry=False, so irrep_name and
    # symm_orb are None.
    #
    # Rebuild only the Mole symmetry metadata in C1.  This does
    # NOT rotate or reoptimize the ROHF orbitals: in C1 every AO
    # and MO belongs to the single totally symmetric irrep.
    #
    mo_coeff_checkpoint = np.asarray(
        data["mo_coeff"]
    )

    overlap_before = mol.intor_symmetric(
        "int1e_ovlp"
    )

    mol_c1 = mol.copy()

    mol_c1.build(
        False,
        False,
        symmetry="C1",
    )

    overlap_after = mol_c1.intor_symmetric(
        "int1e_ovlp"
    )

    if (
        overlap_before.shape
        != overlap_after.shape
    ):
        raise RuntimeError(
            "C1 rebuild changed AO dimension."
        )

    overlap_change = float(
        np.max(
            np.abs(
                overlap_after
                - overlap_before
            )
        )
    )

    print(
        "C1 symmetry metadata:"
    )
    print(
        "  groupname  =",
        mol_c1.groupname,
    )
    print(
        "  irrep_name =",
        mol_c1.irrep_name,
    )
    print(
        "  irrep_id   =",
        mol_c1.irrep_id,
    )
    print(
        "  n(symm_orb)=",
        len(
            mol_c1.symm_orb
        ),
    )
    print(
        "  max |S(C1)-S(original)| =",
        f"{overlap_change:.3e}",
    )

    if mol_c1.groupname != "C1":
        raise RuntimeError(
            f"Expected C1, got {mol_c1.groupname}"
        )

    if (
        mol_c1.irrep_name is None
        or len(
            mol_c1.irrep_name
        ) != 1
    ):
        raise RuntimeError(
            "C1 symmetry metadata not initialized."
        )

    if overlap_change > 1.0e-12:
        raise RuntimeError(
            "C1 rebuild changed the AO overlap matrix."
        )

    orth_error = float(
        np.max(
            np.abs(
                mo_coeff_checkpoint.T
                @ overlap_after
                @ mo_coeff_checkpoint
                - np.eye(
                    mo_coeff_checkpoint.shape[
                        1
                    ]
                )
            )
        )
    )

    print(
        "  ROHF MO orthonormality max error =",
        f"{orth_error:.3e}",
    )

    if orth_error > 1.0e-8:
        raise RuntimeError(
            "Stored ROHF orbitals are not orthonormal "
            "in rebuilt C1 Mole."
        )

    mol = mol_c1

    mf = scf.ROHF(
        mol
    )

    mf.mo_coeff = (
        mo_coeff_checkpoint
    )

    mf.mo_occ = np.asarray(
        data["mo_occ"]
    )

    mf.mo_energy = np.asarray(
        data["mo_energy"]
    )

    mf.e_tot = float(
        data["e_tot"]
    )

    mf.converged = True
    mf.max_memory = 60000

    ss, multiplicity = (
        mf.spin_square()
    )

    print(
        "ROHF checkpoint:",
        path,
    )

    print(
        "electrons:",
        mol.nelectron,
    )

    print(
        "2S:",
        mol.spin,
    )

    print(
        f"E(ROHF) = {mf.e_tot:.12f} Eh"
    )

    print(
        f"<S^2>   = {ss:.10f}"
    )

    print(
        f"mult.    = {multiplicity:.10f}"
    )

    if abs(
        float(ss)
        - 3.75
    ) > 1.0e-8:
        raise RuntimeError(
            "Neutral ROHF reference is not spin-pure quartet."
        )

    return (
        path,
        mf,
    )


def scalar_corr_energy(
    driver,
):
    value = np.asarray(
        driver.correlation_energy,
        dtype=float,
    ).reshape(
        -1
    )

    if value.size != 1:
        raise RuntimeError(
            "Unexpected CCpy correlation_energy shape: "
            f"{value.shape}"
        )

    return float(
        value[0]
    )


def state0_delta_dict(
    deltap3,
):
    obj = deltap3

    if isinstance(
        obj,
        (
            list,
            tuple,
        ),
    ):
        if not obj:
            raise RuntimeError(
                "Empty deltap3 container."
            )
        obj = obj[0]

    if isinstance(
        obj,
        np.ndarray,
    ):
        flat = obj.reshape(
            -1
        )

        if flat.size != 1:
            raise RuntimeError(
                "Unexpected deltap3 ndarray."
            )

        obj = flat[0]

    if not isinstance(
        obj,
        dict,
    ):
        raise RuntimeError(
            "Expected state-0 deltap3 dictionary; "
            f"got {type(obj)}: {obj!r}"
        )

    return {
        str(key): float(value)
        for key, value in obj.items()
        if np.isscalar(
            value
        )
    }


def print_t1_amplitudes(
    driver,
):
    print()
    print(
        "RAW CCSD T1 AMPLITUDE DIAGNOSTIC"
    )
    print(
        "(norms only; NOT a Lee-Taylor T1 diagnostic)"
    )

    for name in (
        "a",
        "b",
    ):
        array = getattr(
            driver.T,
            name,
            None,
        )

        if array is None:
            print(
                f"T1 {name}: unavailable"
            )
            continue

        array = np.asarray(
            array
        )

        print(
            f"T1 {name}: "
            f"shape={array.shape}  "
            f"norm={np.linalg.norm(array):.8f}  "
            f"maxabs={np.max(np.abs(array)):.8f}"
        )


def main():
    checkpoint, mf = (
        load_rohf()
    )

    print()
    print(
        "=" * 120
    )
    print(
        "BUILD CCPY DRIVER FROM SPIN-PURE ROHF"
    )
    print(
        "=" * 120
    )

    driver = Driver.from_pyscf(
        mf,
        nfrozen=NFROZEN,
    )

    driver.system.print_info()

    print()
    print(
        "CCpy occupied counts after frozen-core setup:"
    )

    for attr in (
        "noccupied_alpha",
        "noccupied_beta",
        "nunoccupied_alpha",
        "nunoccupied_beta",
    ):
        print(
            f"  {attr}:",
            getattr(
                driver.system,
                attr,
                "UNAVAILABLE",
            ),
        )

    print()
    print(
        "EXPECTED if nfrozen semantics are spatial-core:"
    )
    print(
        "  correlated occupied alpha = 10"
    )
    print(
        "  correlated occupied beta  = 7"
    )

    driver.options[
        "energy_convergence"
    ] = 1.0e-8

    driver.options[
        "amp_convergence"
    ] = 1.0e-7

    driver.options[
        "maximum_iterations"
    ] = 100

    print()
    print(
        "=" * 120
    )
    print(
        "ROHF-CCSD"
    )
    print(
        "=" * 120
    )

    driver.run_cc(
        method="ccsd"
    )

    reference_energy = float(
        driver.system.reference_energy
    )

    correlation_energy = (
        scalar_corr_energy(
            driver
        )
    )

    ccsd_total = (
        reference_energy
        + correlation_energy
    )

    print()
    print(
        f"CCpy reference energy = "
        f"{reference_energy:.12f} Eh"
    )

    print(
        f"CCSD correlation      = "
        f"{correlation_energy:.12f} Eh"
    )

    print(
        f"CCSD total            = "
        f"{ccsd_total:.12f} Eh"
    )

    print_t1_amplitudes(
        driver
    )

    print()
    print(
        "=" * 120
    )
    print(
        "ROHF-CCSD(T)"
    )
    print(
        "=" * 120
    )

    driver.run_ccp3(
        method="ccsd(t)"
    )

    ccsdt_raw = copy.deepcopy(
        driver.deltap3
    )

    ccsdt_delta = (
        state0_delta_dict(
            ccsdt_raw
        )
    )

    print()
    print(
        "CCSD(T) deltap3:",
        ccsdt_delta,
    )

    #
    # For CCSD(T), CCpy normally reports the correction
    # in the A entry.  Do not silently assume this:
    # store and print the complete returned dictionary.
    #
    ccsdt_key = (
        "A"
        if "A" in ccsdt_delta
        else next(
            iter(
                ccsdt_delta
            )
        )
    )

    ccsdt_total = (
        ccsd_total
        + ccsdt_delta[
            ccsdt_key
        ]
    )

    print(
        f"CCSD(T) selected key = "
        f"{ccsdt_key}"
    )

    print(
        f"CCSD(T) total        = "
        f"{ccsdt_total:.12f} Eh"
    )

    print()
    print(
        "=" * 120
    )
    print(
        "LEFT-CCSD + CR-CC(2,3)"
    )
    print(
        "=" * 120
    )

    driver.run_hbar(
        method="ccsd"
    )

    driver.run_leftcc(
        method="left_ccsd"
    )

    driver.run_ccp3(
        method="crcc23"
    )

    cr_raw = copy.deepcopy(
        driver.deltap3
    )

    cr_delta = (
        state0_delta_dict(
            cr_raw
        )
    )

    print()
    print(
        "CR-CC(2,3) corrections:"
    )

    cr_totals = {}

    for key in (
        "A",
        "B",
        "C",
        "D",
    ):
        if key not in cr_delta:
            continue

        total = (
            ccsd_total
            + cr_delta[
                key
            ]
        )

        cr_totals[
            key
        ] = total

        print(
            f"  {key}: "
            f"delta={cr_delta[key]:+.12f} Eh  "
            f"total={total:.12f} Eh"
        )

    if "D" not in cr_totals:
        raise RuntimeError(
            "CCpy did not return CR-CC(2,3)_D."
        )

    print()
    print(
        "=" * 120
    )
    print(
        "NEUTRAL FeH HIGH-LEVEL CONTROL SUMMARY"
    )
    print(
        "=" * 120
    )

    print(
        f"ROHF        = {reference_energy:.12f} Eh"
    )

    print(
        f"CCSD        = {ccsd_total:.12f} Eh"
    )

    print(
        f"CCSD(T)     = {ccsdt_total:.12f} Eh"
    )

    for key, total in (
        cr_totals.items()
    ):
        print(
            f"CR-CC(2,3){key} = "
            f"{total:.12f} Eh"
        )

    result = {
        "system": "neutral_quartet",
        "geometry_angstrom": 1.575,
        "basis": "def2-tzvppd",
        "reference": "ROHF",
        "checkpoint": str(
            checkpoint
        ),
        "nfrozen_spatial_requested": (
            NFROZEN
        ),
        "reference_energy_hartree": (
            reference_energy
        ),
        "ccsd_correlation_hartree": (
            correlation_energy
        ),
        "ccsd_total_hartree": (
            ccsd_total
        ),
        "ccsdt_deltap3_raw": (
            jsonable(
                ccsdt_raw
            )
        ),
        "ccsdt_selected_key": (
            ccsdt_key
        ),
        "ccsdt_total_hartree": (
            ccsdt_total
        ),
        "crcc23_deltap3_raw": (
            jsonable(
                cr_raw
            )
        ),
        "crcc23_totals_hartree": (
            cr_totals
        ),
    }

    output = (
        OUT
        / "neutral_quartet.json"
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
        output
    )


if __name__ == "__main__":
    main()
