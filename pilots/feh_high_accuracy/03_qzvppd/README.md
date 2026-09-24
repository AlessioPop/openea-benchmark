# def2-QZVPPD basis-convergence control

The converged TZVPPD ROHF states were projected into def2-QZVPPD and
their occupied-subspace continuity was explicitly checked.

The same frozen-core CC calculations were then repeated in QZVPPD.

Representative QZ electronic EAs:

- ROHF: ~1.7990 eV
- CCSD: ~0.9147 eV
- CCSD(T): ~0.8536 eV
- CR-CC(2,3)_D: ~0.8533 eV

Notable result:

At QZVPPD the CCSD(T) and CR-CC(2,3)_D electron affinities differ by
only about 0.3 meV, even though the individual FeH references remain
challenging.
