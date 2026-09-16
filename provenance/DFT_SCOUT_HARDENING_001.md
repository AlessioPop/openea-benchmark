# DFT Scout Contract and Provenance Hardening 001

## Status

PASS

Baseline before hardening:

`223dbe8 Add conservative local PEC construction`

This hardening pass was performed before beginning geometry/minimum scouting.

## Motivation

A repository-wide audit after local PEC construction identified several
scientific-context and provenance contracts that were not protected by the
existing test suite.

The existing implementation was internally functional, but the following
cases were insufficiently guarded:

- different ECP assignments could enter same-state comparison/deduplication;
- root identifiers did not encode ECP assignment;
- root identifiers rounded geometry to six decimal places;
- numerically equivalent Python floating-point geometries could form separate
  geometry layers;
- checkpoint geometry was not checked against the root record;
- atom identities were not retained by `SCFRootRecord`;
- root identifiers did not distinguish atomic composition;
- slug normalization could cause identifier collisions;
- checkpoint files were not self-describing with respect to the full OpenEA
  scientific calculation context.

## Implemented hardening

### Canonical geometry

A single fixed-R geometry representation is now used throughout the DFT scout
workflow.

Bond lengths are canonicalized to 12 decimal places, matching the coordinate
representation actually passed to PySCF.

This prevents ordinary floating-point representation artifacts such as

`1.2`

and

`0.4 * 3`

from becoming distinct geometry layers, while preserving genuinely distinct
sample points.

### Scientific comparison context

Electronic-state identity, deduplication, branch continuity, and branch-graph
construction now preserve or validate the relevant calculation context,
including:

- molecule label;
- ordered atomic composition;
- charge;
- spin sector;
- functional;
- basis;
- reference type;
- ECP assignments where applicable.

Different Hamiltonian/model contexts are therefore not silently compared as
the same state.

### Root identifiers

Root identifiers now include:

- atomic composition;
- 12-decimal canonical geometry;
- functional;
- basis;
- ECP context;
- starting guess;
- a digest of the complete canonical calculation context.

The digest is calculated before slug normalization, preventing collisions
between distinct raw labels that normalize to the same filename-safe text.

### Checkpoint provenance

Final PySCF checkpoints now contain versioned OpenEA metadata under

`openea/context_json`

The embedded context records:

- schema version;
- root ID;
- molecule label;
- atom identities;
- charge;
- spin sector;
- geometry;
- functional;
- basis;
- reference;
- origin guess;
- ECP assignments.

Checkpoint fingerprint reconstruction validates the stored context against the
`SCFRootRecord`.

Checkpoint audit also independently verifies:

- charge;
- spin;
- diatomic geometry;
- bond distance;
- total energy;
- electron count;
- spin trace;
- numerical overlap quality.

A checkpoint that is stale or belongs to another calculation context is not
accepted as evidence for a root.

## Integration validation

A real PySCF H2 test now exercises the complete implemented DFT scout chain
over three geometries and two independent starting guesses per geometry:

SCF attempts
-> stability/canonical roots
-> final checkpoints
-> checkpoint fingerprints
-> same-geometry state deduplication
-> representative roots
-> adjacent-geometry branch continuity
-> layered branch graph
-> local PEC construction

The resulting H2 branch is unambiguous and produces one local PEC.

## Test status

Final suite:

`72 tests`

Result:

`OK`

No production identity or branch thresholds were introduced by this hardening
pass. Threshold values appearing in integration tests are explicitly
validation-only.

## Scope boundary

This hardening pass does not perform:

- PEC interpolation;
- polynomial or spline fitting;
- energy minimization;
- equilibrium-geometry assignment;
- cross-spin ground-state selection;
- electron-affinity calculation;
- adaptive geometry resampling.

Those remain responsibilities of subsequent workflow layers.

The next development step is DFT-0C-8 geometry/minimum scouting.
