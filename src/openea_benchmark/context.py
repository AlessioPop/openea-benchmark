from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Sequence


CHECKPOINT_CONTEXT_VERSION = 1


def calculation_context(
    *,
    molecule: str,
    atom_a: str,
    atom_b: str,
    charge: int,
    spin_2s: int,
    r_angstrom: float,
    functional: str,
    basis: str,
    reference: str,
    origin_guess: str,
    ecp_assignments: Sequence[
        tuple[str, str]
    ],
) -> dict[str, Any]:
    """
    Canonical scientific identity of one fixed-geometry SCF calculation.

    Numerical convergence settings are deliberately not part of electronic
    state identity. Molecule composition, Hamiltonian/model context, geometry,
    spin sector, reference, and starting-root provenance are.
    """
    return {
        "molecule": str(
            molecule
        ),
        "atom_a": str(
            atom_a
        ),
        "atom_b": str(
            atom_b
        ),
        "charge": int(
            charge
        ),
        "spin_2s": int(
            spin_2s
        ),
        "r_angstrom": float(
            r_angstrom
        ),
        "functional": str(
            functional
        ),
        "basis": str(
            basis
        ),
        "reference": str(
            reference
        ),
        "origin_guess": str(
            origin_guess
        ),
        "ecp_assignments": [
            [
                str(element),
                str(ecp_name),
            ]
            for element, ecp_name
            in sorted(
                ecp_assignments
            )
        ],
    }


def canonical_json(
    payload: Any,
) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=True,
        allow_nan=False,
    )


def context_digest(
    payload: Any,
    *,
    length: int = 16,
) -> str:
    if length <= 0:
        raise ValueError(
            "digest length must be > 0"
        )

    return sha256(
        canonical_json(
            payload
        ).encode(
            "utf-8"
        )
    ).hexdigest()[
        :length
    ]


def checkpoint_context(
    *,
    root_id: str,
    calculation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": (
            CHECKPOINT_CONTEXT_VERSION
        ),
        "root_id": str(
            root_id
        ),
        "calculation": (
            calculation
        ),
    }


def parse_context_json(
    value,
) -> dict[str, Any]:
    """
    Decode a scalar string returned from a PySCF/HDF5 checkpoint.
    """
    if hasattr(
        value,
        "item",
    ):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(
        value,
        bytes,
    ):
        value = value.decode(
            "utf-8"
        )

    if not isinstance(
        value,
        str,
    ):
        raise ValueError(
            "OpenEA checkpoint metadata is not a JSON string"
        )

    try:
        payload = json.loads(
            value
        )
    except Exception as exc:
        raise ValueError(
            "OpenEA checkpoint metadata is invalid JSON"
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "OpenEA checkpoint metadata must decode to an object"
        )

    return payload
