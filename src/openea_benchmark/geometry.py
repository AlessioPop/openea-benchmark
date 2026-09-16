from __future__ import annotations

from math import isfinite


GEOMETRY_DECIMALS = 12


def canonicalize_r_angstrom(
    value: float,
) -> float:
    """
    Canonical fixed-R geometry used throughout the DFT scout workflow.

    PySCF input coordinates are written with GEOMETRY_DECIMALS decimal
    places. Canonicalizing records to the same representation prevents
    numerically equivalent Python floats from becoming distinct geometry
    layers while preserving every geometry that is actually sent to PySCF.
    """
    number = float(
        value
    )

    if (
        not isfinite(number)
        or number <= 0.0
    ):
        raise ValueError(
            "r_angstrom must be finite and > 0"
        )

    canonical = float(
        f"{number:.{GEOMETRY_DECIMALS}f}"
    )

    if canonical <= 0.0:
        raise ValueError(
            "r_angstrom is below the canonical geometry resolution"
        )

    return canonical
