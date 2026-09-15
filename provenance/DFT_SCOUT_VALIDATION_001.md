# DFT Scout Validation 001

## Status

Completed exploratory validation through DFT-0B1B-R1.

Date: 2026-09-15

This document records the empirical development evidence used to design the
first OpenEA DFT state-discovery and PEC workflow.

It is a scientific-development checkpoint, not a frozen production method.

The calculations summarized here were performed with the open-source PySCF
stack. The relevant development environment used Python 3.12 and PySCF 2.14.0.

## Scientific role of the DFT layer

DFT is being evaluated as the inexpensive first electronic-structure layer of
OpenEA.

Its intended roles are:

- candidate spin-state discovery;
- SCF-root discovery;
- electronic-branch identification and following;
- preliminary PEC and equilibrium-geometry scouting;
- inexpensive delta-SCF electron-affinity estimates;
- stability, spin, basis, and state-continuity diagnostics.

DFT is not intended to define the final high-accuracy electron affinity.

The final OpenEA result will be obtained only after the appropriate correlated,
basis-set, relativistic, spin-orbit, nuclear-motion, and uncertainty layers have
been applied and validated.

## DFT-0A — software and semantic capability

The initial capability gates established that the candidate DFT methods could
be executed in PySCF with analytical gradients on representative open-shell
diatomics.

The audited candidate set included:

- PBE;
- B3LYP;
- PBE0;
- TPSSh;
- r2SCAN;
- r2SCANh;
- r2SCAN0;
- wB97X-V;
- wB97M-V;
- MN15;
- M06-L.

The audit also established explicit semantics for:

- VV10/nonlocal-correlation handling;
- def2 ECP activation;
- basis availability through the relevant main-group, 3d, 4d, and 5d targets;
- internal and external SCF-stability analysis;
- linear-molecule symmetry handling.

A central result of the stability work was that a converged SCF solution is
not automatically a suitable electronic root.

For the OH X2Pi test case, symmetry-resolved Abelian components demonstrated
that apparent C1 stability problems can originate from the Pi degeneracy rather
than from a physically distinct lower electronic state.

Therefore OpenEA must store stability diagnostics rather than reducing them to
a single pass/fail flag.

## DFT-0B1A — fixed-geometry main-group basis benchmark

A 360-job fixed-geometry benchmark was performed for:

- OH;
- CN;
- BO;
- C2;
- N2.

The functionals were:

- r2SCAN;
- r2SCANh;
- PBE0;
- wB97M-V.

The basis candidates included:

- ma-def2-TZVP;
- ma-def2-TZVPP;
- def2-TZVPD;
- def2-TZVPPD;
- def2-QZVPPD;
- aug-pcseg-2;
- aug-pcseg-3;
- aug-cc-pVTZ;
- aug-cc-pVQZ.

358 of 360 SCF calculations completed.

The two failures were both r2SCAN calculations for the N2 anion with
aug-pcseg-2 or aug-pcseg-3.

### Basis-definition result

For the main-group atoms represented in this test, some nominally different
def2 basis names were found to contain identical atomic basis definitions.

In particular:

- ma-def2-TZVP and ma-def2-TZVPP were identical for B, C, N, and O;
- def2-TZVPD and def2-TZVPPD were identical for B, C, N, and O.

Consequently these name changes must not be interpreted as genuine cardinal or
polarization convergence steps on those atoms.

The corresponding basis pairs do differ for H and for the transition-metal and
heavy-element cases audited later.

### Main-group convergence result

The two independent augmented basis families showed substantially cleaner
within-family convergence than the def2 TZ-to-QZ comparison for several of the
main-group systems.

For CN, BO, and C2, aug-pcseg-3 and aug-cc-pVQZ generally approached each other
closely, whereas def2-QZVPPD retained a systematic offset, particularly for CN
and C2.

OH retained a larger cross-family spread and therefore remained a useful
sensitivity case.

No median or average of the basis families is treated as a reference value.

### N2 unbound control

The converged N2 neutral-anion fixed-geometry energy differences remained
negative and the converged anion HOMOs remained positive.

The result is consistent with the intended unbound control, but neither a
positive Kohn-Sham HOMO nor an SCF failure is used by itself as an OpenEA
BOUND/UNBOUND criterion.

Once systematic evidence establishes that no bound anion ground state exists,
OpenEA will classify the system as UNBOUND rather than attempting to determine
a precise negative electron affinity.

### Cost result

wB97M-V was substantially more expensive than the other candidate scout
functionals in this benchmark.

This cost became important when combined with the stability and convergence
results of the transition-metal gates below.

## DFT-0B1B — transition-metal and heavy-state diagnostic benchmark

The next stage tested fixed-geometry state ordering and SCF-root behavior for:

- FeH;
- TiH;
- NbC.

The benchmark deliberately retained multiple spin sectors and multiple initial
SCF guesses.

Its purpose was not to produce final electron affinities.

The principal result was that simply selecting the lowest converged SCF energy
from a collection of guesses is unsafe.

Different initial guesses frequently converged to different roots, and many of
the automatically lowest solutions were internally unstable.

This established the need for explicit root canonicalization and branch
continuity before basis comparisons can be interpreted scientifically.

The legacy v0.9 data were used only as regression and stress evidence, not as
authoritative physical state labels.

Important stress cases included:

- robust FeH neutral 2S=3 versus 2S=5 ordering;
- functional sensitivity of the TiH anion 2S=2 versus 2S=4 sectors;
- severe spin contamination in nominal low-spin TiH roots;
- broken-symmetry behavior in nominal NbC singlet solutions.

## DFT-0B1B-R1A — anchor-root canonicalization

The R1A gate used def2-QZVPPD as a common anchor basis and attempted to
stability-canonicalize candidate roots for:

- r2SCAN;
- r2SCANh;
- PBE0;
- wB97M-V.

The results showed that r2SCAN, r2SCANh, and PBE0 could provide useful anchor
roots across the intended diagnostic set.

wB97M-V produced substantially more failures on the difficult TiH and NbC
cases in addition to its substantially higher computational cost.

Therefore wB97M-V is not currently admitted to the routine Tier-0 scout path.

This is a workflow-design decision, not a general claim that wB97M-V is an
inferior density functional.

It may remain useful as an optional cross-check.

### State diagnostics

After stability canonicalization, FeH state ordering became reproducible for
the retained functionals.

TiH demonstrated that a reproducible SCF branch can still carry severe spin
contamination.

Therefore reproducibility and SCF stability do not remove the need for an
independent single-reference/multireference diagnostic layer.

### NbC broken-symmetry diagnostic

For the tested retained functionals, the constrained RKS NbC singlet solution
was internally stable but externally unstable.

Following the RKS-to-UKS instability produced lower broken-symmetry solutions.

This behavior is treated as multireference/broken-symmetry evidence rather than
as an automatically accepted spin-pure singlet energy.

## DFT-0B1B-R1B — basis projection of canonicalized roots

R1B projected the stability-canonicalized def2-QZVPPD anchor roots into smaller
or independent basis sets.

For FeH and TiH the tested target bases included:

- ma-def2-TZVPP;
- def2-TZVPPD;
- aug-pcseg-2;
- aug-cc-pVTZ.

For NbC only the locally audited def2 basis route was used:

- ma-def2-TZVPP;
- def2-TZVPPD.

No independent augmented 4d basis family has yet been validated locally for the
Nb test, so the NbC result must not be interpreted as independent-family basis
validation.

113 of 114 projected calculations completed directly.

The principal scientific result was that occupied-space projection provides a
useful state-continuity diagnostic.

Energy differences are not interpreted as basis effects if the projected root
changes qualitatively after SCF or stability canonicalization.

### Provisional scout-basis evidence

ma-def2-TZVPP preserved the relevant anchor branches well across the tested 3d
cases while remaining substantially cheaper than the QZ anchor level.

It is therefore the strongest current candidate for a broadly available
Tier-0 scout basis.

aug-pcseg-2 remains useful as an independent 3d DFT-oriented cross-check.

def2-QZVPPD remains useful as an anchor/refinement basis.

These are provisional development choices, not frozen production policy.

### aug-cc-pVTZ root-continuity warning

A TiH anion case demonstrated that an apparently well-projected aug-cc-pVTZ
solution could undergo a qualitatively large occupied-space rotation during
stability canonicalization.

This confirms that state continuity must be checked explicitly rather than
inferred from convergence or energy alone.

## DFT-0B1B-R1B-R1 — generic SCF-rescue diagnostic

The only direct R1B failure was:

- NbC neutral;
- nominal 2S=1 sector;
- r2SCAN;
- def2-TZVPPD.

The projected density itself was well conditioned and reproduced the intended
electron count.

Neither a long conventional SCF iteration nor a direct Newton attempt
converged.

A generic route consisting of:

1. projected-density initialization;
2. temporary damping and level shifting;
3. removal of the shift;
4. clean unshifted restart;
5. stability canonicalization;

converged successfully.

The final occupied-space overlaps with the projected QZ anchor were:

- alpha minimum overlap: 0.999954;
- beta minimum overlap: 0.999991.

The final stability energy shift was zero to the reported precision.

The resulting solution retained strong spin contamination,
with <S^2> approximately 1.434 for a nominal doublet.

The original R1B failure is therefore interpreted as an SCF-convergence
failure, not as loss of the electronic branch in def2-TZVPPD.

The spin contamination remains a genuine scientific diagnostic.

The generic rescue sequence may be evaluated as workflow infrastructure, but
it must not become a molecule-specific exception.

## Provisional DFT scout decisions

The current evidence supports the following development choices for the next
workflow stage.

### Routine scout functional panel

Retain for development:

- r2SCAN;
- r2SCANh;
- PBE0.

Do not currently use wB97M-V as a routine Tier-0 functional.

It may remain available as an optional diagnostic cross-check.

No single functional has been declared the universally correct DFT model.

Functional disagreement remains useful diagnostic information.

### Basis roles

Current provisional roles are:

- ma-def2-TZVPP: leading universal Tier-0 scout candidate;
- aug-pcseg-2: independent 3d cross-check;
- def2-QZVPPD: anchor/refinement basis;
- def2-TZVPPD: useful same-family control, but not currently preferred as the
  universal default;
- aug-cc-pVTZ: useful comparison basis but not currently preferred as the
  universal state-discovery basis.

For Zr, Nb, and Hf, explicit matching def2 ECP assignment is required where
appropriate.

ECP assignment should be element-specific rather than applied globally to all
atoms in a molecule.

## Workflow requirements established by these gates

The first production-quality DFT layer must not reduce state discovery to one
SCF calculation per spin sector.

It must support:

- multiple generic SCF initial guesses;
- explicit SCF convergence status;
- internal stability canonicalization;
- external stability diagnostics where scientifically relevant;
- storage of stability energy changes;
- spin expectation values and spin-contamination diagnostics;
- root deduplication;
- state/branch identity separate from energetic ordering;
- occupied-space continuity diagnostics;
- checkpoint/restart;
- generic SCF rescue without molecule-specific rules;
- explicit basis and ECP provenance;
- fixed terminology distinguishing fixed-geometry energy differences from EA_e;
- escalation rather than silent acceptance when state identity remains
  ambiguous.

## What has not been established

These gates do not establish:

- the final DFT functional;
- the final production basis;
- final equilibrium geometries;
- final electron affinities;
- a final single-reference/multireference escalation threshold;
- a final bound/unbound classifier;
- the correlated focal-point hierarchy;
- CBS policy;
- core-valence policy;
- scalar-relativistic policy;
- SOC correction policy;
- nuclear-motion treatment;
- final uncertainty assignment.

## Next stage

The next stage is DFT-0C.

DFT-0C will convert the validated ideas above into the first tested OpenEA
workflow component:

1. candidate spin-state discovery;
2. multiple generic SCF roots;
3. stability canonicalization;
4. root deduplication;
5. state and branch identity;
6. local PEC construction;
7. geometry/minimum scouting;
8. preliminary delta-SCF electron-affinity diagnostics.

The first implementation will use a small validation/regression subset before
being expanded.

No prediction calculations begin at this stage.
