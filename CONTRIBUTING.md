# Contributing to OpenEA

OpenEA is under active scientific development.

The repository is intentionally structured so that another researcher
can fork it, reproduce existing validation work, modify one component
of the workflow and test whether the change improves the general
methodology.

## What this repository is

OpenEA is a general workflow for electron affinities of diatomic
molecules.

It is not a FeH-specific workflow.

FeH is currently the primary high-difficulty validation system.

## Why FeH dominates the current development history

FeH was chosen because it combines many difficult effects in one
small molecule.

It therefore acts as a useful stress-test for:

- electronic-state discovery
- state continuity
- multireference behavior
- dynamic correlation
- coupled-cluster diagnostics
- basis convergence
- scalar relativity
- geometry refinement
- zero-point energy
- spin-orbit coupling

A contribution should preferably turn lessons from FeH into general
workflow logic rather than introduce FeH-specific special cases.

## Repository structure

Generic workflow code belongs in the normal source tree.

Development and validation calculations may live under:

    pilots/

The current FeH development history is stored under:

    pilots/feh_high_accuracy/

Important entry points include:

    pilots/feh_high_accuracy/STATUS.md
    pilots/feh_high_accuracy/01_mr_nevpt2_unified/
    pilots/feh_high_accuracy/02_rohf_ccpy_tzvppd/
    pilots/feh_high_accuracy/03_qzvppd/
    pilots/feh_high_accuracy/04_x2c_qzvppd/
    pilots/feh_high_accuracy/05_x2c_ccsdt_pec/
    pilots/feh_high_accuracy/06_soc/

The pilots are scientific development records.

They should not automatically be interpreted as the final generic
implementation.

## Suggested setup

A typical development workflow is:

    git clone <your-fork>
    cd openea-benchmark

    python -m venv .venv
    source .venv/bin/activate

    pip install -e .

Check the repository dependency files before creating a new environment
because the exact dependencies may evolve.

## Suggested Git workflow

Fork the repository and develop changes on a feature branch:

    git checkout -b feature/<short-description>

Keep commits focused.

Examples:

    Add SOC convergence diagnostic
    Generalize active-space selector
    Add basis convergence criterion
    Add validation diatomic
    Improve PEC state continuity

## Scientific development rule

Before changing the code, identify which layer of the workflow is being
modified.

Examples include:

- state discovery
- determinant identity
- electronic manifolds
- PEC continuity
- geometry scouting
- correlated electronic energy
- multireference escalation
- basis convergence
- scalar relativity
- spin-orbit coupling
- ZPE
- uncertainty estimation
- bound / unbound decision logic

Whenever possible, change one scientific assumption at a time.

This makes it much easier to determine why an electron affinity changed.

## Validation philosophy

Experimental electron affinities are valuable validation data.

They should not be used internally to:

- select the desired root
- choose a preferred electronic state
- tune thresholds
- select a computational result because it is closer to experiment
- introduce an empirical molecule-specific correction

Otherwise the workflow is no longer genuinely predictive.

## Unbound systems

A stable anion must not be assumed.

If the available evidence establishes with sufficient confidence that
the anion ground state is not electronically bound, record:

    UNBOUND

The project does not require a precisely converged negative electron
affinity in that case.

## What should normally be committed

Good repository content includes:

- source code
- tests
- compact JSON result summaries
- small reproducibility metadata
- documentation
- validation scripts

Large runtime artifacts should normally stay outside Git.

Examples:

- large checkpoint files
- raw integral files
- large logs
- virtual environments
- copied third-party repositories

## Good first contributions

Useful tasks for a new contributor include:

- add another validation diatomic
- improve automatic state discovery
- improve state-continuity diagnostics
- generalize active-space construction
- add coupled-cluster reliability diagnostics
- test another open-source multireference method
- improve SOC convergence testing
- automate basis-set convergence estimates
- improve uncertainty propagation
- add bound / unbound diagnostics
- improve provenance and result summaries

A useful question for every contribution is:

    Does this make the workflow more reliable for diatomics
    in general, rather than only improving one benchmark molecule?

## Current scientific philosophy

FeH is deliberately being used as a difficult benchmark.

Success means more than reproducing the experimental FeH electron
affinity.

The more important success criterion is that the procedures required
to handle FeH can be expressed as defensible, automated and
transferable rules for other diatomic systems.
