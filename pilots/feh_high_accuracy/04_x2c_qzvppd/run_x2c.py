from __future__ import annotations

from pathlib import Path
import copy
import json

import numpy as np

from pyscf import scf
from pyscf.scf import chkfile

from ccpy.drivers.driver import Driver


NR_DIR = Path(
    "/srv/storage/homes/analysis/dschmid/photodetachment/"
    "openea-pilot/feh_rohf_qzvppd"
)

OUT = Path(
    "/srv/storage/homes/analysis/dschmid/photodetachment/"
    "openea-pilot/feh_ccpy_x2c_qzvppd"
)

HARTREE_TO_EV = 27.211386245988
NFROZEN = 5


NR_EA = {
    "ROHF": 1.799034,
    "CCSD": 0.914676,
    "CCSD(T)": 0.853599,
    "CR-CC(2,3)_D": 0.853308,
}


SYSTEMS = (
    {
        "key": "neutral_quartet",
        "checkpoint": (
            "neutral_quartet__rohf_qzvppd.chk"
        ),
        "expected_s2": 3.75,
        "expected_occ_alpha": 10,
        "expected_occ_beta": 7,
    },
    {
        "key": "anion_quintet",
        "checkpoint": (
            "anion_quintet__rohf_qzvppd.chk"
        ),
        "expected_s2": 6.0,
        "expected_occ_alpha": 11,
        "expected_occ_beta": 7,
    },
)


def delta_dict(value):
    obj = value

    if isinstance(
        obj,
        (list, tuple),
    ):
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
            f"Unexpected deltap3 object: {obj!r}"
        )

    return {
        str(key): float(val)
        for key, val in obj.items()
        if np.isscalar(val)
    }


def corr_energy(driver):
    values = np.asarray(
        driver.correlation_energy,
        dtype=float,
    ).reshape(-1)

    if values.size != 1:
        raise RuntimeError(
            "Unexpected correlation-energy shape."
        )

    return float(
        values[0]
    )


def occupied_overlap(
    mol,
    old_mo,
    old_occ,
    new_mo,
    new_occ,
):
    nold = int(
        np.count_nonzero(
            old_occ > 0
        )
    )

    nnew = int(
        np.count_nonzero(
            new_occ > 0
        )
    )

    if nold != nnew:
        raise RuntimeError(
            f"Occupied dimension changed: "
            f"{nold} -> {nnew}"
        )

    S = mol.intor_symmetric(
        "int1e_ovlp"
    )

    old = old_mo[
        :,
        :nold
    ]

    new = new_mo[
        :,
        :nnew
    ]

    matrix = (
        old.T
        @ S
        @ new
    )

    sv = np.linalg.svd(
        matrix,
        compute_uv=False,
    )

    return sv


def run_system(system):
    key = system["key"]

    print()
    print("=" * 120)
    print(
        key,
        "SFX2C-1e / def2-QZVPPD"
    )
    print("=" * 120)

    nr_chk = (
        NR_DIR
        / system[
            "checkpoint"
        ]
    )

    if not nr_chk.is_file():
        raise RuntimeError(
            f"Missing NR QZ checkpoint: {nr_chk}"
        )

    mol, nr_data = (
        chkfile.load_scf(
            str(nr_chk)
        )
    )

    nr_mo = np.asarray(
        nr_data[
            "mo_coeff"
        ]
    )

    nr_occ = np.asarray(
        nr_data[
            "mo_occ"
        ]
    )

    print(
        "NR checkpoint:",
        nr_chk,
    )

    print(
        "basis functions:",
        mol.nao_nr(),
    )

    #
    # Spin-free exact-two-component one-electron Hamiltonian.
    #
    mf = scf.ROHF(
        mol
    ).x2c()

    mf.max_memory = 60000
    mf.max_cycle = 100
    mf.conv_tol = 1.0e-11
    mf.conv_tol_grad = 1.0e-7

    mf.chkfile = str(
        OUT
        / f"{key}__x2c_rohf.chk"
    )

    #
    # Same AO basis. Start from the converged
    # nonrelativistic QZ density.
    #
    dm0 = mf.init_guess_by_chkfile(
        str(nr_chk),
        project=False,
    )

    print()
    print(
        "--- X2C ROHF ---"
    )

    mf.kernel(
        dm0=dm0
    )

    if not mf.converged:
        print(
            "Switching to Newton X2C-ROHF."
        )

        mf = mf.newton()

        mf.max_memory = 60000
        mf.max_cycle = 100
        mf.conv_tol = 1.0e-11
        mf.conv_tol_grad = 1.0e-7

        mf.kernel()

    if not mf.converged:
        raise RuntimeError(
            f"{key}: X2C ROHF failed."
        )

    ss, mult = (
        mf.spin_square()
    )

    print()
    print(
        "X2C ROHF RESULT"
    )

    print(
        f"  E = {mf.e_tot:.12f} Eh"
    )

    print(
        f"  <S^2> = {float(ss):.10f}"
    )

    print(
        f"  multiplicity = {float(mult):.10f}"
    )

    if abs(
        float(ss)
        - system[
            "expected_s2"
        ]
    ) > 1.0e-8:
        raise RuntimeError(
            f"{key}: X2C reference "
            "has wrong spin."
        )

    sv = occupied_overlap(
        mol,
        nr_mo,
        nr_occ,
        np.asarray(
            mf.mo_coeff
        ),
        np.asarray(
            mf.mo_occ
        ),
    )

    print()
    print(
        "NR-QZ -> X2C-QZ occupied-space overlap:"
    )

    print(
        np.array2string(
            sv,
            precision=10,
        )
    )

    print(
        "minimum singular value:",
        f"{np.min(sv):.10f}",
    )

    if np.min(
        sv
    ) < 0.90:
        raise RuntimeError(
            f"{key}: X2C converged to "
            "a different occupied state."
        )

    print()
    print(
        "--- CCPY FROM X2C ROHF ---"
    )

    driver = Driver.from_pyscf(
        mf,
        nfrozen=NFROZEN,
    )

    nocc_a = int(
        driver.system.noccupied_alpha
    )

    nocc_b = int(
        driver.system.noccupied_beta
    )

    print(
        "occupied alpha:",
        nocc_a,
    )

    print(
        "occupied beta:",
        nocc_b,
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
            f"{key}: frozen-core "
            "occupation mismatch."
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
        "--- X2C-CCSD ---"
    )

    driver.run_cc(
        method="ccsd"
    )

    reference = float(
        driver.system.reference_energy
    )

    cc_corr = corr_energy(
        driver
    )

    ccsd = (
        reference
        + cc_corr
    )

    t1a = np.asarray(
        driver.T.a
    )

    t1b = np.asarray(
        driver.T.b
    )

    print()
    print(
        "X2C CCSD diagnostic"
    )

    print(
        "  T1a norm/max =",
        f"{np.linalg.norm(t1a):.8f}",
        f"{np.max(np.abs(t1a)):.8f}",
    )

    print(
        "  T1b norm/max =",
        f"{np.linalg.norm(t1b):.8f}",
        f"{np.max(np.abs(t1b)):.8f}",
    )

    print()
    print(
        "--- X2C-CCSD(T) ---"
    )

    driver.run_ccp3(
        method="ccsd(t)"
    )

    ccsdt_delta = delta_dict(
        copy.deepcopy(
            driver.deltap3
        )
    )

    if "A" not in ccsdt_delta:
        raise RuntimeError(
            "CCSD(T) correction missing."
        )

    ccsdt = (
        ccsd
        + ccsdt_delta[
            "A"
        ]
    )

    print()
    print(
        "--- X2C CR-CC(2,3) ---"
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

    cr_delta = delta_dict(
        copy.deepcopy(
            driver.deltap3
        )
    )

    cr_totals = {}

    for label in (
        "A",
        "B",
        "C",
        "D",
    ):
        if label in cr_delta:
            cr_totals[
                label
            ] = (
                ccsd
                + cr_delta[
                    label
                ]
            )

    if "D" not in cr_totals:
        raise RuntimeError(
            "CR-D missing."
        )

    print()
    print(
        "X2C SUMMARY"
    )

    print(
        f"ROHF        = {reference:.12f} Eh"
    )

    print(
        f"CCSD        = {ccsd:.12f} Eh"
    )

    print(
        f"CCSD(T)     = {ccsdt:.12f} Eh"
    )

    for label in (
        "A",
        "B",
        "C",
        "D",
    ):
        if label in cr_totals:
            print(
                f"CR-CC(2,3){label} = "
                f"{cr_totals[label]:.12f} Eh"
            )

    result = {
        "system": key,
        "reference_energy_hartree": (
            reference
        ),
        "ccsd_total_hartree": (
            ccsd
        ),
        "ccsdt_total_hartree": (
            ccsdt
        ),
        "crcc23_totals_hartree": (
            cr_totals
        ),
        "occupied_overlap_min_sv": (
            float(
                np.min(sv)
            )
        ),
        "t1a_norm": float(
            np.linalg.norm(
                t1a
            )
        ),
        "t1a_maxabs": float(
            np.max(
                np.abs(
                    t1a
                )
            )
        ),
        "t1b_norm": float(
            np.linalg.norm(
                t1b
            )
        ),
        "t1b_maxabs": float(
            np.max(
                np.abs(
                    t1b
                )
            )
        ),
    }

    (
        OUT
        / f"{key}.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    return result


def ea(
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

    n = results[
        "neutral_quartet"
    ]

    a = results[
        "anion_quintet"
    ]

    x2c_ea = {
        "ROHF": ea(
            n[
                "reference_energy_hartree"
            ],
            a[
                "reference_energy_hartree"
            ],
        ),
        "CCSD": ea(
            n[
                "ccsd_total_hartree"
            ],
            a[
                "ccsd_total_hartree"
            ],
        ),
        "CCSD(T)": ea(
            n[
                "ccsdt_total_hartree"
            ],
            a[
                "ccsdt_total_hartree"
            ],
        ),
        "CR-CC(2,3)_D": ea(
            n[
                "crcc23_totals_hartree"
            ][
                "D"
            ],
            a[
                "crcc23_totals_hartree"
            ][
                "D"
            ],
        ),
    }

    print()
    print("=" * 120)
    print(
        "SCALAR-RELATIVISTIC X2C EFFECT ON EA"
    )
    print("=" * 120)

    print(
        f"{'METHOD':18s}"
        f"{'NR-QZ / eV':>16s}"
        f"{'X2C-QZ / eV':>16s}"
        f"{'X2C-NR / eV':>16s}"
    )

    print(
        "-" * 66
    )

    for method, value in (
        x2c_ea.items()
    ):
        nr = NR_EA[
            method
        ]

        shift = (
            value
            - nr
        )

        print(
            f"{method:18s}"
            f"{nr:16.6f}"
            f"{value:16.6f}"
            f"{shift:16.6f}"
        )

    print()
    print(
        "X2C CCSD(T)-CR-D difference = "
        f"{x2c_ea['CCSD(T)'] - x2c_ea['CR-CC(2,3)_D']:+.6f} eV"
    )

    output = {
        "systems": results,
        "nr_qz_ea_ev": NR_EA,
        "x2c_qz_ea_ev": x2c_ea,
        "scalar_relativistic_shift_ev": {
            method: (
                value
                - NR_EA[
                    method
                ]
            )
            for method, value
            in x2c_ea.items()
        },
    }

    (
        OUT
        / "combined.json"
    ).write_text(
        json.dumps(
            output,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
