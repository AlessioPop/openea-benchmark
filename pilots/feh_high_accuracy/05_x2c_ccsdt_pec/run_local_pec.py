from __future__ import annotations

from pathlib import Path
import copy
import json
import math

import numpy as np

from pyscf import gto, scf
from pyscf.scf import chkfile

from ccpy.drivers.driver import Driver


CENTER_DIR = Path(
    "/srv/storage/homes/analysis/dschmid/photodetachment/"
    "openea-pilot/feh_ccpy_x2c_qzvppd"
)

OUT = Path(
    "/srv/storage/homes/analysis/dschmid/photodetachment/"
    "openea-pilot/feh_x2c_ccsdt_local_pec"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

BASIS = "def2-qzvppd"
NFROZEN = 5

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_CM = 219474.63136320
AMU_TO_ME = 1822.888486209

MASS_FE56_U = 55.93493633
MASS_H1_U = 1.00782503223

MU_U = (
    MASS_FE56_U * MASS_H1_U
    / (MASS_FE56_U + MASS_H1_U)
)

MU_AU = MU_U * AMU_TO_ME


SYSTEMS = (
    {
        "key": "neutral_quartet",
        "charge": 0,
        "spin": 3,
        "expected_s2": 3.75,
        "center_R": 1.575,
        "center_chk": "neutral_quartet__x2c_rohf.chk",
        "grid": [
            1.475,
            1.500,
            1.525,
            1.550,
            1.575,
        ],
    },
    {
        "key": "anion_quintet",
        "charge": -1,
        "spin": 4,
        "expected_s2": 6.0,
        "center_R": 1.675,
        "center_chk": "anion_quintet__x2c_rohf.chk",
        "grid": [
            1.625,
            1.650,
            1.675,
            1.700,
            1.725,
        ],
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


def scalar_corr_energy(driver):
    values = np.asarray(
        driver.correlation_energy,
        dtype=float,
    ).reshape(-1)

    if values.size != 1:
        raise RuntimeError(
            "Unexpected CC correlation-energy shape."
        )

    return float(values[0])


def build_molecule(system, R):
    return gto.M(
        atom=[
            ["Fe", (0.0, 0.0, 0.0)],
            ["H", (0.0, 0.0, R)],
        ],
        unit="Angstrom",
        basis=BASIS,
        charge=system["charge"],
        spin=system["spin"],
        symmetry="C1",
        verbose=4,
        max_memory=60000,
    )


def occupied_continuity(
    center_mol,
    center_mo,
    center_occ,
    mol,
    mo,
    occ,
):
    n_center = int(
        np.count_nonzero(
            center_occ > 0
        )
    )

    n_new = int(
        np.count_nonzero(
            occ > 0
        )
    )

    if n_center != n_new:
        raise RuntimeError(
            "Occupied dimension changed."
        )

    S_cross = gto.intor_cross(
        "int1e_ovlp",
        center_mol,
        mol,
    )

    overlap = (
        center_mo[:, :n_center].T
        @ S_cross
        @ mo[:, :n_new]
    )

    sv = np.linalg.svd(
        overlap,
        compute_uv=False,
    )

    return sv


def load_existing_center_results():
    path = (
        CENTER_DIR
        / "combined.json"
    )

    if not path.is_file():
        raise RuntimeError(
            f"Missing X2C center result: {path}"
        )

    return json.loads(
        path.read_text()
    )


def run_point(
    system,
    R,
    center_mol,
    center_data,
):
    key = system["key"]

    point_name = (
        f"{key}__R{R:.3f}"
    )

    json_path = (
        OUT
        / f"{point_name}.json"
    )

    if json_path.is_file():
        print()
        print(
            "REUSING EXISTING POINT:",
            json_path,
        )

        return json.loads(
            json_path.read_text()
        )

    print()
    print("=" * 120)
    print(
        key,
        f"R = {R:.3f} A"
    )
    print("=" * 120)

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

    point_chk = (
        OUT
        / f"{point_name}__x2c_rohf.chk"
    )

    mf.chkfile = str(
        point_chk
    )

    center_chk = (
        CENTER_DIR
        / system[
            "center_chk"
        ]
    )

    dm0 = mf.init_guess_by_chkfile(
        str(center_chk),
        project=True,
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
        mf.chkfile = str(
            point_chk
        )

        mf.kernel()

    if not mf.converged:
        raise RuntimeError(
            f"{point_name}: ROHF failed."
        )

    ss, mult = mf.spin_square()

    print(
        f"E(X2C-ROHF) = "
        f"{mf.e_tot:.12f} Eh"
    )

    print(
        f"<S^2> = {float(ss):.10f}"
    )

    if abs(
        float(ss)
        - system["expected_s2"]
    ) > 1.0e-8:
        raise RuntimeError(
            f"{point_name}: wrong spin."
        )

    sv = occupied_continuity(
        center_mol,
        np.asarray(
            center_data["mo_coeff"]
        ),
        np.asarray(
            center_data["mo_occ"]
        ),
        mol,
        np.asarray(
            mf.mo_coeff
        ),
        np.asarray(
            mf.mo_occ
        ),
    )

    min_sv = float(
        np.min(sv)
    )

    print(
        "center -> point occupied "
        "min singular value:",
        f"{min_sv:.10f}",
    )

    if min_sv < 0.95:
        raise RuntimeError(
            f"{point_name}: state continuity failed."
        )

    print()
    print(
        "--- X2C-QZVPPD CCSD ---"
    )

    driver = Driver.from_pyscf(
        mf,
        nfrozen=NFROZEN,
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

    driver.run_cc(
        method="ccsd"
    )

    reference = float(
        driver.system.reference_energy
    )

    corr = scalar_corr_energy(
        driver
    )

    ccsd = (
        reference
        + corr
    )

    t1a = np.asarray(
        driver.T.a
    )

    t1b = np.asarray(
        driver.T.b
    )

    print()
    print(
        "--- X2C-QZVPPD CCSD(T) ---"
    )

    driver.run_ccp3(
        method="ccsd(t)"
    )

    triples = delta_dict(
        copy.deepcopy(
            driver.deltap3
        )
    )

    if "A" not in triples:
        raise RuntimeError(
            "CCSD(T) correction missing."
        )

    ccsdt = (
        ccsd
        + triples["A"]
    )

    print()
    print(
        f"E(CCSD)    = {ccsd:.12f} Eh"
    )

    print(
        f"(T)        = {triples['A']:+.12f} Eh"
    )

    print(
        f"E(CCSD(T)) = {ccsdt:.12f} Eh"
    )

    result = {
        "system": key,
        "R_angstrom": R,
        "x2c_rohf_energy_hartree": (
            float(
                mf.e_tot
            )
        ),
        "ccsd_energy_hartree": (
            ccsd
        ),
        "ccsdt_energy_hartree": (
            ccsdt
        ),
        "triples_hartree": (
            triples["A"]
        ),
        "occupied_min_sv": (
            min_sv
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

    json_path.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    return result


def quadratic_analysis(
    points,
):
    R = np.asarray(
        [
            point[
                "R_angstrom"
            ]
            for point in points
        ],
        dtype=float,
    )

    E = np.asarray(
        [
            point[
                "ccsdt_energy_hartree"
            ]
            for point in points
        ],
        dtype=float,
    )

    coeff = np.polyfit(
        R,
        E,
        2,
    )

    a, b, c = coeff

    if a <= 0.0:
        raise RuntimeError(
            "PEC fit has non-positive curvature."
        )

    Re = float(
        -b
        / (2.0 * a)
    )

    if not (
        np.min(R)
        < Re
        < np.max(R)
    ):
        raise RuntimeError(
            "Fitted minimum lies outside scan: "
            f"Re={Re:.6f} A"
        )

    Emin = float(
        np.polyval(
            coeff,
            Re,
        )
    )

    fitted = np.polyval(
        coeff,
        R,
    )

    residual_ev = (
        (
            E
            - fitted
        )
        * HARTREE_TO_EV
    )

    max_residual_mev = float(
        np.max(
            np.abs(
                residual_ev
            )
        )
        * 1000.0
    )

    #
    # polynomial uses R in Angstrom.
    #
    # d2E/dR_A^2 = 2*a [Eh / A^2]
    #
    # Convert to Eh / bohr^2:
    #
    curvature_angstrom = float(
        2.0 * a
    )

    curvature_bohr = (
        curvature_angstrom
        * BOHR_TO_ANGSTROM**2
    )

    omega_au = math.sqrt(
        curvature_bohr
        / MU_AU
    )

    omega_cm = (
        omega_au
        * HARTREE_TO_CM
    )

    zpe_hartree = (
        0.5
        * omega_au
    )

    zpe_ev = (
        zpe_hartree
        * HARTREE_TO_EV
    )

    return {
        "Re_angstrom": Re,
        "Emin_hartree": Emin,
        "curvature_Eh_per_A2": (
            curvature_angstrom
        ),
        "omega_e_cm-1": (
            omega_cm
        ),
        "zpe_hartree": (
            zpe_hartree
        ),
        "zpe_ev": (
            zpe_ev
        ),
        "max_quadratic_fit_residual_meV": (
            max_residual_mev
        ),
        "fit_coefficients": (
            coeff.tolist()
        ),
    }


def main():
    existing = (
        load_existing_center_results()
    )

    center_results = existing[
        "systems"
    ]

    all_results = {}

    for system in SYSTEMS:
        key = system["key"]

        center_chk = (
            CENTER_DIR
            / system[
                "center_chk"
            ]
        )

        center_mol, center_data = (
            chkfile.load_scf(
                str(center_chk)
            )
        )

        points = []

        for R in system[
            "grid"
        ]:
            if abs(
                R
                - system[
                    "center_R"
                ]
            ) < 1.0e-12:

                center_energy = float(
                    center_results[
                        key
                    ][
                        "ccsdt_total_hartree"
                    ]
                )

                print()
                print(
                    "REUSING EXISTING CENTER:",
                    key,
                    f"R={R:.3f}",
                    f"E={center_energy:.12f} Eh",
                )

                points.append(
                    {
                        "system": key,
                        "R_angstrom": R,
                        "ccsdt_energy_hartree": (
                            center_energy
                        ),
                        "source": (
                            "existing X2C-QZ "
                            "center calculation"
                        ),
                    }
                )

                continue

            point = run_point(
                system,
                R,
                center_mol,
                center_data,
            )

            points.append(
                point
            )

        analysis = (
            quadratic_analysis(
                points
            )
        )

        print()
        print("=" * 120)
        print(
            key,
            "LOCAL PEC RESULT"
        )
        print("=" * 120)

        for point in points:
            print(
                f"R={point['R_angstrom']:.3f} A  "
                f"E={point['ccsdt_energy_hartree']:.12f} Eh"
            )

        print()
        print(
            f"Re = "
            f"{analysis['Re_angstrom']:.6f} A"
        )

        print(
            f"omega_e = "
            f"{analysis['omega_e_cm-1']:.2f} cm^-1"
        )

        print(
            f"ZPE = "
            f"{analysis['zpe_ev']:.6f} eV"
        )

        print(
            "max quadratic residual = "
            f"{analysis['max_quadratic_fit_residual_meV']:.3f} meV"
        )

        all_results[
            key
        ] = {
            "points": points,
            "analysis": analysis,
        }

    neutral = all_results[
        "neutral_quartet"
    ][
        "analysis"
    ]

    anion = all_results[
        "anion_quintet"
    ][
        "analysis"
    ]

    ea_electronic_optimized = (
        neutral[
            "Emin_hartree"
        ]
        - anion[
            "Emin_hartree"
        ]
    ) * HARTREE_TO_EV

    zpe_shift = (
        neutral[
            "zpe_ev"
        ]
        - anion[
            "zpe_ev"
        ]
    )

    ea_0 = (
        ea_electronic_optimized
        + zpe_shift
    )

    fixed_geometry_ea = (
        float(
            center_results[
                "neutral_quartet"
            ][
                "ccsdt_total_hartree"
            ]
        )
        - float(
            center_results[
                "anion_quintet"
            ][
                "ccsdt_total_hartree"
            ]
        )
    ) * HARTREE_TO_EV

    geometry_shift = (
        ea_electronic_optimized
        - fixed_geometry_ea
    )

    print()
    print("=" * 120)
    print(
        "X2C-QZVPPD-CCSD(T) ADIABATIC EA BUILDUP"
    )
    print("=" * 120)

    print(
        f"fixed PBE0 geometries electronic EA = "
        f"{fixed_geometry_ea:.6f} eV"
    )

    print(
        f"CCSD(T) local-PEC optimized EA     = "
        f"{ea_electronic_optimized:.6f} eV"
    )

    print(
        f"geometry-relaxation shift          = "
        f"{geometry_shift:+.6f} eV"
    )

    print(
        f"neutral ZPE                        = "
        f"{neutral['zpe_ev']:.6f} eV"
    )

    print(
        f"anion ZPE                          = "
        f"{anion['zpe_ev']:.6f} eV"
    )

    print(
        f"ZPE contribution to EA             = "
        f"{zpe_shift:+.6f} eV"
    )

    print()
    print(
        f"EA_0 before SOC = "
        f"{ea_0:.6f} eV"
    )

    print(
        f"difference from 0.934 eV = "
        f"{ea_0 - 0.934:+.6f} eV"
    )

    output = {
        "method": (
            "SFX2C-1e-ROHF-CCSD(T)/def2-QZVPPD"
        ),
        "isotope_for_vibration": "56FeH",
        "reduced_mass_u": MU_U,
        "systems": all_results,
        "fixed_geometry_ea_ev": (
            fixed_geometry_ea
        ),
        "optimized_electronic_ea_ev": (
            ea_electronic_optimized
        ),
        "geometry_relaxation_shift_ev": (
            geometry_shift
        ),
        "zpe_shift_ev": (
            zpe_shift
        ),
        "ea_v0_before_soc_ev": (
            ea_0
        ),
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
