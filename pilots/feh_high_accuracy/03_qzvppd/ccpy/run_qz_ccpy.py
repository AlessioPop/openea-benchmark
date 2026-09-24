from __future__ import annotations

from pathlib import Path
import copy
import json

import numpy as np

from pyscf import scf
from pyscf.scf import chkfile

from ccpy.drivers.driver import Driver


QZ_ROHF_DIR = Path(
    "/srv/storage/homes/analysis/dschmid/photodetachment/"
    "openea-pilot/feh_rohf_qzvppd"
)

OUT = Path(
    "/srv/storage/homes/analysis/dschmid/photodetachment/"
    "openea-pilot/feh_ccpy_rohf_crcc23_qzvppd"
)

TZ_CC_DIR = Path(
    "/srv/storage/homes/analysis/dschmid/photodetachment/"
    "openea-pilot/feh_ccpy_rohf_crcc23"
)

HARTREE_TO_EV = 27.211386245988

#
# Freeze Fe 1s, 2s, 2p:
# five spatial orbitals = ten electrons.
#
NFROZEN = 5


SYSTEMS = (
    {
        "key": "neutral_quartet",
        "checkpoint": (
            "neutral_quartet__rohf_qzvppd.chk"
        ),
        "geometry_angstrom": 1.575,
        "expected_s2": 3.75,
        "expected_mult": 4.0,
        "expected_corr_electrons": 17,
        "expected_occ_alpha": 10,
        "expected_occ_beta": 7,
    },
    {
        "key": "anion_quintet",
        "checkpoint": (
            "anion_quintet__rohf_qzvppd.chk"
        ),
        "geometry_angstrom": 1.675,
        "expected_s2": 6.0,
        "expected_mult": 5.0,
        "expected_corr_electrons": 18,
        "expected_occ_alpha": 11,
        "expected_occ_beta": 7,
    },
)


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
        flat = obj.reshape(-1)

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
            "Expected deltap3 dictionary, got "
            f"{type(obj)}: {obj!r}"
        )

    return {
        str(key): float(value)
        for key, value in obj.items()
        if np.isscalar(value)
    }


def scalar_corr_energy(
    driver,
):
    value = np.asarray(
        driver.correlation_energy,
        dtype=float,
    ).reshape(-1)

    if value.size != 1:
        raise RuntimeError(
            "Unexpected CCpy correlation-energy "
            f"shape: {value.shape}"
        )

    return float(
        value[0]
    )


def load_qz_rohf(
    system,
):
    path = (
        QZ_ROHF_DIR
        / system["checkpoint"]
    )

    if not path.is_file():
        raise RuntimeError(
            f"Missing QZ checkpoint: {path}"
        )

    mol, data = chkfile.load_scf(
        str(path)
    )

    mo_coeff = np.asarray(
        data["mo_coeff"]
    )

    #
    # QZ checkpoints were already generated in C1.
    # Still validate the metadata because CCpy requires it.
    #
    if (
        mol.irrep_name is None
        or mol.symm_orb is None
    ):
        print(
            "Symmetry metadata absent; "
            "rebuilding Mole in C1."
        )

        overlap_before = (
            mol.intor_symmetric(
                "int1e_ovlp"
            )
        )

        mol_c1 = mol.copy()

        mol_c1.build(
            False,
            False,
            symmetry="C1",
        )

        overlap_after = (
            mol_c1.intor_symmetric(
                "int1e_ovlp"
            )
        )

        difference = float(
            np.max(
                np.abs(
                    overlap_after
                    - overlap_before
                )
            )
        )

        if difference > 1.0e-12:
            raise RuntimeError(
                "C1 rebuild changed AO overlap."
            )

        mol = mol_c1

    overlap = (
        mol.intor_symmetric(
            "int1e_ovlp"
        )
    )

    orth_error = float(
        np.max(
            np.abs(
                mo_coeff.T
                @ overlap
                @ mo_coeff
                - np.eye(
                    mo_coeff.shape[1]
                )
            )
        )
    )

    if orth_error > 1.0e-8:
        raise RuntimeError(
            "QZ ROHF MOs are not orthonormal."
        )

    mf = scf.ROHF(
        mol
    )

    mf.mo_coeff = (
        mo_coeff
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

    ss, mult = (
        mf.spin_square()
    )

    print(
        "checkpoint:",
        path,
    )

    print(
        "basis functions:",
        mol.nao_nr(),
    )

    print(
        "electrons:",
        mol.nelectron,
    )

    print(
        "charge:",
        mol.charge,
    )

    print(
        "2S:",
        mol.spin,
    )

    print(
        "C1 irrep names:",
        mol.irrep_name,
    )

    print(
        "MO orthonormality max error:",
        f"{orth_error:.3e}",
    )

    print(
        f"E(ROHF) = {mf.e_tot:.12f} Eh"
    )

    print(
        f"<S^2>   = {float(ss):.10f}"
    )

    print(
        f"mult.    = {float(mult):.10f}"
    )

    if (
        abs(
            float(ss)
            - system["expected_s2"]
        )
        > 1.0e-8
    ):
        raise RuntimeError(
            f"{system['key']}: wrong ROHF spin."
        )

    return (
        path,
        mf,
    )


def print_t1(
    driver,
):
    result = {}

    print()
    print(
        "RAW CCSD T1 AMPLITUDES"
    )

    for spin in (
        "a",
        "b",
    ):
        array = getattr(
            driver.T,
            spin,
            None,
        )

        if array is None:
            print(
                f"T1 {spin}: unavailable"
            )
            continue

        array = np.asarray(
            array
        )

        norm = float(
            np.linalg.norm(
                array
            )
        )

        maximum = float(
            np.max(
                np.abs(
                    array
                )
            )
        )

        result[spin] = {
            "shape": list(
                array.shape
            ),
            "norm": norm,
            "maxabs": maximum,
        }

        print(
            f"T1 {spin}: "
            f"shape={array.shape}  "
            f"norm={norm:.8f}  "
            f"maxabs={maximum:.8f}"
        )

    return result


def run_system(
    system,
):
    print()
    print(
        "=" * 120
    )
    print(
        system["key"],
        "def2-QZVPPD"
    )
    print(
        "=" * 120
    )

    checkpoint, mf = (
        load_qz_rohf(
            system
        )
    )

    print()
    print(
        "--- BUILD CCPY DRIVER ---"
    )

    driver = Driver.from_pyscf(
        mf,
        nfrozen=NFROZEN,
    )

    driver.system.print_info()

    nocc_a = int(
        driver.system.noccupied_alpha
    )

    nocc_b = int(
        driver.system.noccupied_beta
    )

    corr_electrons = (
        nocc_a
        + nocc_b
    )

    print()
    print(
        "Frozen-core validation:"
    )

    print(
        "  correlated electrons:",
        corr_electrons,
    )

    print(
        "  occupied alpha:",
        nocc_a,
    )

    print(
        "  occupied beta:",
        nocc_b,
    )

    if (
        corr_electrons
        != system[
            "expected_corr_electrons"
        ]
    ):
        raise RuntimeError(
            f"{system['key']}: wrong number "
            "of correlated electrons."
        )

    if (
        nocc_a
        != system[
            "expected_occ_alpha"
        ]
        or nocc_b
        != system[
            "expected_occ_beta"
        ]
    ):
        raise RuntimeError(
            f"{system['key']}: wrong correlated "
            "occupied counts."
        )

    driver.options[
        "energy_convergence"
    ] = 1.0e-8

    driver.options[
        "amp_convergence"
    ] = 1.0e-7

    driver.options[
        "maximum_iterations"
    ] = 120

    print()
    print(
        "=" * 120
    )
    print(
        "CCSD"
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

    corr_energy = (
        scalar_corr_energy(
            driver
        )
    )

    ccsd_total = (
        reference_energy
        + corr_energy
    )

    print()
    print(
        f"Reference energy = "
        f"{reference_energy:.12f} Eh"
    )

    print(
        f"CCSD correlation = "
        f"{corr_energy:.12f} Eh"
    )

    print(
        f"CCSD total       = "
        f"{ccsd_total:.12f} Eh"
    )

    t1 = print_t1(
        driver
    )

    print()
    print(
        "=" * 120
    )
    print(
        "CCSD(T)"
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

    if "A" not in ccsdt_delta:
        raise RuntimeError(
            "CCSD(T) A correction missing."
        )

    ccsdt_total = (
        ccsd_total
        + ccsdt_delta[
            "A"
        ]
    )

    print()
    print(
        "CCSD(T) correction:",
        f"{ccsdt_delta['A']:+.12f} Eh"
    )

    print(
        "CCSD(T) total:",
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

    cr_totals = {}

    print()
    print(
        "CR-CC(2,3) corrections:"
    )

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
            "CR-CC(2,3)_D result missing."
        )

    print()
    print(
        "=" * 120
    )
    print(
        system["key"],
        "QZ SUMMARY"
    )
    print(
        "=" * 120
    )

    print(
        f"ROHF        = "
        f"{reference_energy:.12f} Eh"
    )

    print(
        f"CCSD        = "
        f"{ccsd_total:.12f} Eh"
    )

    print(
        f"CCSD(T)     = "
        f"{ccsdt_total:.12f} Eh"
    )

    for key in (
        "A",
        "B",
        "C",
        "D",
    ):
        if key in cr_totals:
            print(
                f"CR-CC(2,3){key} = "
                f"{cr_totals[key]:.12f} Eh"
            )

    result = {
        "system": system[
            "key"
        ],
        "geometry_angstrom": (
            system[
                "geometry_angstrom"
            ]
        ),
        "basis": "def2-qzvppd",
        "nfrozen_spatial": (
            NFROZEN
        ),
        "checkpoint": str(
            checkpoint
        ),
        "reference_energy_hartree": (
            reference_energy
        ),
        "ccsd_correlation_hartree": (
            corr_energy
        ),
        "ccsd_total_hartree": (
            ccsd_total
        ),
        "t1_raw": t1,
        "ccsdt_correction_hartree": (
            ccsdt_delta[
                "A"
            ]
        ),
        "ccsdt_total_hartree": (
            ccsdt_total
        ),
        "crcc23_corrections_hartree": (
            cr_delta
        ),
        "crcc23_totals_hartree": (
            cr_totals
        ),
    }

    output = (
        OUT
        / f"{system['key']}.json"
    )

    output.write_text(
        json.dumps(
            jsonable(
                result
            ),
            indent=2,
        )
    )

    print()
    print(
        "saved:",
        output,
    )

    return result


def ea_ev(
    neutral,
    anion,
):
    return (
        float(
            neutral
        )
        - float(
            anion
        )
    ) * HARTREE_TO_EV


def qz_ea_table(
    neutral,
    anion,
):
    result = {}

    result[
        "ROHF"
    ] = ea_ev(
        neutral[
            "reference_energy_hartree"
        ],
        anion[
            "reference_energy_hartree"
        ],
    )

    result[
        "CCSD"
    ] = ea_ev(
        neutral[
            "ccsd_total_hartree"
        ],
        anion[
            "ccsd_total_hartree"
        ],
    )

    result[
        "CCSD(T)"
    ] = ea_ev(
        neutral[
            "ccsdt_total_hartree"
        ],
        anion[
            "ccsdt_total_hartree"
        ],
    )

    for key in (
        "A",
        "B",
        "C",
        "D",
    ):
        if (
            key
            in neutral[
                "crcc23_totals_hartree"
            ]
            and key
            in anion[
                "crcc23_totals_hartree"
            ]
        ):
            result[
                f"CR-CC(2,3)_{key}"
            ] = ea_ev(
                neutral[
                    "crcc23_totals_hartree"
                ][
                    key
                ],
                anion[
                    "crcc23_totals_hartree"
                ][
                    key
                ],
            )

    return result


def load_tz_eas():
    path = (
        TZ_CC_DIR
        / "ea_comparison.json"
    )

    if not path.is_file():
        raise RuntimeError(
            f"Missing TZ comparison JSON: {path}"
        )

    data = json.loads(
        path.read_text()
    )

    return data[
        "electronic_ea_ev"
    ]


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

    qz_eas = qz_ea_table(
        neutral,
        anion,
    )

    tz_eas = load_tz_eas()

    print()
    print(
        "=" * 120
    )
    print(
        "TZVPPD -> QZVPPD ELECTRONIC EA COMPARISON"
    )
    print(
        "=" * 120
    )

    print()
    print(
        f"{'METHOD':18s}"
        f"{'TZ EA / eV':>16s}"
        f"{'QZ EA / eV':>16s}"
        f"{'QZ-TZ / eV':>16s}"
    )

    print(
        "-" * 66
    )

    for method, qz_value in (
        qz_eas.items()
    ):
        tz_value = (
            tz_eas.get(
                method
            )
        )

        if tz_value is None:
            print(
                f"{method:18s}"
                f"{'n/a':>16s}"
                f"{qz_value:16.6f}"
                f"{'n/a':>16s}"
            )
            continue

        shift = (
            qz_value
            - float(
                tz_value
            )
        )

        print(
            f"{method:18s}"
            f"{float(tz_value):16.6f}"
            f"{qz_value:16.6f}"
            f"{shift:16.6f}"
        )

    print()
    print(
        "QZ CCSD(T) -> CR-D EA shift:"
    )

    print(
        f"{qz_eas['CR-CC(2,3)_D'] - qz_eas['CCSD(T)']:+.6f} eV"
    )

    cr_values = [
        qz_eas[
            f"CR-CC(2,3)_{key}"
        ]
        for key in (
            "A",
            "B",
            "C",
            "D",
        )
        if (
            f"CR-CC(2,3)_{key}"
            in qz_eas
        )
    ]

    print(
        "QZ CR A-D EA spread:"
    )

    print(
        f"{max(cr_values) - min(cr_values):.6f} eV"
    )

    summary = {
        "basis": "def2-qzvppd",
        "nfrozen_spatial": NFROZEN,
        "results": jsonable(
            results
        ),
        "electronic_ea_ev": qz_eas,
        "tz_electronic_ea_ev": (
            tz_eas
        ),
        "qz_minus_tz_ea_ev": {
            method: (
                float(value)
                - float(
                    tz_eas[
                        method
                    ]
                )
            )
            for method, value
            in qz_eas.items()
            if method in tz_eas
        },
    }

    output = (
        OUT
        / "combined.json"
    )

    output.write_text(
        json.dumps(
            jsonable(
                summary
            ),
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
