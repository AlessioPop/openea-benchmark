# OpenEA workflow scope

## Goal

OpenEA aims to provide a general computational workflow for predicting
electron affinities of diatomic molecules.

The scientific target is not FeH and not one particular chemical
family.

The workflow should ultimately be usable across chemically different
systems such as:

- main-group diatomics
- hydrides
- oxides
- halides
- transition-metal diatomics
- weakly bound anions
- systems for which the anion is not electronically bound

The workflow should return either a defensible positive electron
affinity or, when sufficiently established:

    UNBOUND

## FeH is a stress-test, not the project target

FeH is currently the most developed validation system.

It was deliberately chosen because it combines many difficult features
in a single diatomic molecule:

- open-shell electronic structure
- transition-metal d electrons
- several nearby electronic states
- multireference character
- large correlation effects
- problematic single-reference diagnostics
- basis-set sensitivity
- scalar-relativistic effects
- spin-orbit coupling
- nontrivial neutral/anion state correspondence

This makes FeH useful as a deliberately hard test of the workflow.

The goal is not to construct a computational recipe that reproduces
FeH specifically.

Instead, every problem exposed by FeH should be translated into a
general diagnostic, decision rule or escalation mechanism that can
also be applied to other diatomics.

## Core validation rule

A new method or correction is not accepted simply because it improves
agreement with the experimental FeH electron affinity.

A change should have an independent physical or methodological
justification.

Experimental electron affinities are used after prediction for
validation.

They must not be used internally to:

- select a preferred root
- choose a preferred electronic state
- tune numerical thresholds
- choose a method solely because its answer is closer to experiment
- introduce molecule-specific empirical corrections

## Current workflow architecture

### 1. State discovery

Multiple electronic solutions and spin sectors may need to be explored.

A single SCF solution must not automatically be assumed to represent
the relevant state.

### 2. Strict determinant identity

Similar energies do not imply identical electronic states.

The workflow therefore tracks quantities such as:

- density fingerprints
- spin densities
- occupied orbital subspaces
- spin expectation values

Strict determinant identity remains separate from broader physical
manifold identity.

### 3. Electronic manifolds

Near-degenerate determinants may represent different orientations or
components of the same physical electronic manifold.

A higher-level manifold representation is therefore used without
weakening strict determinant-level identity.

### 4. State continuity along the PEC

Electronic states must be followed along bond length.

Root number and energetic ordering alone are not considered sufficient
for state tracking.

### 5. DFT as a scouting layer

DFT is primarily useful for:

- state discovery
- approximate geometry
- PEC scouting
- continuity analysis

It is not automatically accepted as the final EA method.

### 6. High-level correlation

The final electronic EA should use a higher-level treatment whenever
required by the system and target uncertainty.

The FeH benchmark has already demonstrated that different correlated
methods can produce very different neutral-anion energy differences.

### 7. Method diagnostics

Single-reference calculations require diagnostics.

Large amplitudes, strong state mixing, instabilities or disagreement
between related high-level treatments should trigger further
investigation rather than silent acceptance.

### 8. Basis convergence

Electron affinities are differences between two very large total
energies.

Basis convergence must therefore be checked explicitly.

Diffuse basis functions are particularly important for anions.

### 9. Scalar relativity

Scalar-relativistic effects are tested separately where chemically
relevant.

For FeH they are already known to contribute several tens of meV to
the final electron affinity.

### 10. Geometry and vibration

The adiabatic EA requires appropriate neutral and anion minima.

Zero-point vibrational energies are included through the difference

    ZPE(neutral) - ZPE(anion)

rather than being ignored or approximated as cancelling automatically.

### 11. Spin-orbit coupling

SOC is conceptually separate from scalar relativity.

It should be calculated only when relevant and its contribution to the
neutral-anion energy difference should be assessed explicitly.

### 12. Unbound anions

The workflow must not force every neutral molecule to possess a bound
anion.

If suitable calculations consistently indicate that no bound anion
ground state exists, the result should be:

    UNBOUND

A precise negative EA is unnecessary.

## What FeH has taught us so far

The current FeH development has already exposed several important
general lessons.

### Balanced active spaces matter

Neutral and anion multireference calculations must use physically
comparable orbital spaces.

### Formal convergence does not guarantee accurate differential
correlation

The repaired CASSCF / CASCI / SC-NEVPT2 branch was internally
consistent but still produced a poor electron affinity.

This indicates a methodological limitation rather than merely a root
tracking or convergence bug.

### Dynamic correlation is crucial

The independent ROHF / coupled-cluster branch changed the FeH
electron affinity dramatically and moved it much closer to experiment.

### Single-reference diagnostics remain important

The FeH coupled-cluster amplitudes are large.

Therefore agreement between different triples treatments, basis
convergence and independent multireference checks remain important.

### Basis effects are measurable

Moving from def2-TZVPPD to def2-QZVPPD changed the high-level
electron affinity by several tens of meV.

### Scalar relativity matters

Spin-free X2C changed the FeH electron affinity by roughly
0.04-0.05 eV at the high-level correlated level.

### Geometry and ZPE both matter

The current X2C-QZVPPD-CCSD(T) local PEC treatment gives approximately:

    fixed-geometry electronic EA       0.90466 eV
    optimized electronic EA            0.88805 eV
    geometry contribution             -0.01661 eV
    ZPE contribution                  +0.03705 eV
    EA(v=0) before SOC                 0.92510 eV

The remaining major physical correction currently under investigation
is spin-orbit coupling.

## Why this is encouraging but not the endpoint

The current FeH value is already close to experiment.

That is useful evidence that the architecture is moving in the right
direction.

It is not a reason to declare the workflow finished.

The important next step is to determine whether the same methodology
works across a chemically diverse validation set.

## Long-term workflow questions

A production workflow should eventually answer automatically:

1. Which spin states and electronic states need to be considered?
2. Are neutral and anion state assignments reliable?
3. Are the states continuous along the PEC?
4. Is a single-reference method adequate?
5. If not, which multireference escalation should be used?
6. Is the basis sufficiently converged?
7. Are scalar-relativistic effects important?
8. Is spin-orbit coupling important?
9. Is the anion actually bound?
10. What uncertainty should accompany the result?

The objective is therefore not:

    reproduce FeH

The objective is:

    develop a robust electron-affinity workflow for diatomics
    that still works when confronted with systems as difficult
    as FeH.
