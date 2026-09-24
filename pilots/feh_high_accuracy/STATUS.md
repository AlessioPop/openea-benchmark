# FeH high-accuracy validation status

## Scientific progression

The FeH / FeH- system is used as a difficult validation case for the
OpenEA molecular electron-affinity workflow.

### Multireference branch

A unified, charge-balanced CAS was constructed for FeH and FeH-.
Separated multi-root CASCI reproduced the state-averaged CASSCF roots
after convergence repair.

Representative results:

- SA4 CASSCF EA: 0.393990 eV
- SA4 SC-NEVPT2 EA: 0.251238 eV
- SA2 CASSCF EA: 0.398331 eV
- SA2 SC-NEVPT2 EA: 0.297613 eV

The low NEVPT2 EA therefore does not originate from the earlier
charge-unbalanced active spaces, failed CASCI convergence, or the
SA4 state averaging itself.

### Independent ROHF / coupled-cluster control

Spin-pure ROHF references were used with CCpy.

def2-TZVPPD:

- CCSD: 0.871133 eV
- CCSD(T): 0.834866 eV
- CR-CC(2,3)_D: 0.801057 eV

def2-QZVPPD:

- CCSD: 0.914676 eV
- CCSD(T): 0.853599 eV
- CR-CC(2,3)_D: 0.853308 eV

The QZ CCSD(T) and CR-D EAs agree to about 0.3 meV.
Large T1 amplitudes nevertheless flag substantial non-single-reference
character.

### Scalar relativity

SFX2C-1e/def2-QZVPPD gives:

- CCSD: 0.965962 eV
- CCSD(T): 0.904660 eV
- CR-CC(2,3)_D: 0.892157 eV

The scalar-relativistic contribution is roughly +0.04 to +0.05 eV
for the high-level EA.

### Geometry and zero-point energy

Local X2C-QZVPPD-CCSD(T) PECs yield:

- optimized electronic EA: 0.888050 eV
- geometry-relaxation shift: -0.016609 eV
- ZPE contribution: +0.037052 eV
- EA(v=0) before SOC: 0.925102 eV

### Current work

The remaining major correction under active investigation is
spin-orbit coupling using multistate CASSCF/CASCI plus fci-siso.

No experimental EA is used as a hidden state-selection or
method-selection criterion. Experimental data are validation only.
