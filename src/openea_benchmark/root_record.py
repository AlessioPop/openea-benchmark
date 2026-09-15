from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class SCFRunStatus(str, Enum):
    """
    Lifecycle status of one attempted SCF root.

    FAILED:
        No converged electronic solution was obtained.

    CONVERGED:
        SCF converged, but internal stability has not been established.

    CANONICALIZED:
        The stored solution has passed the configured internal-stability
        canonicalization procedure.

    This status deliberately does not encode spin purity, physical state
    identity, ground-state assignment, or BOUND/UNBOUND classification.
    """

    FAILED = "FAILED"
    CONVERGED = "CONVERGED"
    CANONICALIZED = "CANONICALIZED"


@dataclass(frozen=True)
class SCFRootRecord:
    """
    Immutable scientific record for one SCF solution.

    spin_2s follows the PySCF convention

        spin = N_alpha - N_beta = 2S.

    The record stores evidence. It does not decide whether the solution is
    the physical ground state or whether SR/MR escalation is required.
    """

    root_id: str

    molecule: str
    charge: int
    spin_2s: int
    r_angstrom: float

    functional: str
    basis: str

    origin_guess: str
    scf_path: str
    reference: str

    status: SCFRunStatus

    energy_hartree: float | None = None

    internal_stable: bool | None = None
    external_stable: bool | None = None
    stability_shift_ev: float | None = None

    s2: float | None = None
    observed_multiplicity: float | None = None

    checkpoint_path: str | None = None
    diagnostic_message: str = ""

    ecp_assignments: tuple[
        tuple[str, str],
        ...,
    ] = ()

    def __post_init__(self) -> None:
        if not self.root_id.strip():
            raise ValueError(
                "root_id must be non-empty"
            )

        if not self.molecule.strip():
            raise ValueError(
                "molecule must be non-empty"
            )

        if self.spin_2s < 0:
            raise ValueError(
                "spin_2s must be >= 0"
            )

        if (
            not isfinite(self.r_angstrom)
            or self.r_angstrom <= 0.0
        ):
            raise ValueError(
                "r_angstrom must be finite and > 0"
            )

        if not self.functional.strip():
            raise ValueError(
                "functional must be non-empty"
            )

        if not self.basis.strip():
            raise ValueError(
                "basis must be non-empty"
            )

        if not self.origin_guess.strip():
            raise ValueError(
                "origin_guess must be non-empty"
            )

        if not self.scf_path.strip():
            raise ValueError(
                "scf_path must be non-empty"
            )

        if not self.reference.strip():
            raise ValueError(
                "reference must be non-empty"
            )

        if self.status != SCFRunStatus.FAILED:
            if (
                self.energy_hartree is None
                or not isfinite(
                    self.energy_hartree
                )
            ):
                raise ValueError(
                    "converged roots require a finite energy"
                )

        if (
            self.status
            == SCFRunStatus.CANONICALIZED
            and self.internal_stable is not True
        ):
            raise ValueError(
                "CANONICALIZED requires internal_stable=True"
            )

        for name, value in (
            (
                "stability_shift_ev",
                self.stability_shift_ev,
            ),
            ("s2", self.s2),
            (
                "observed_multiplicity",
                self.observed_multiplicity,
            ),
        ):
            if (
                value is not None
                and not isfinite(value)
            ):
                raise ValueError(
                    f"{name} must be finite when provided"
                )

        seen_elements: set[str] = set()

        for element, ecp_name in self.ecp_assignments:
            if not element.strip():
                raise ValueError(
                    "ECP element names must be non-empty"
                )

            if not ecp_name.strip():
                raise ValueError(
                    "ECP names must be non-empty"
                )

            if element in seen_elements:
                raise ValueError(
                    f"duplicate ECP assignment for {element}"
                )

            seen_elements.add(element)

    @property
    def nominal_spin_s(self) -> float:
        return self.spin_2s / 2.0

    @property
    def nominal_multiplicity(self) -> int:
        return self.spin_2s + 1

    @property
    def expected_s2(self) -> float:
        s = self.nominal_spin_s
        return s * (s + 1.0)

    @property
    def spin_contamination(self) -> float | None:
        """
        Return <S^2> - S(S+1).

        This quantity is diagnostic only. A nonzero value does not by
        itself invalidate, relabel, or discard the SCF root.
        """
        if self.s2 is None:
            return None

        return self.s2 - self.expected_s2
