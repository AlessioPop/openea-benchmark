# FeH unified multireference validation

Pilot calculations for the FeH / FeH- electron-affinity validation
using a charge-balanced unified active-space construction.

The branch includes the development and repair work around:

- unified 15-orbital Fe/H target space
- SA-CASSCF
- separated multi-root CASCI validation
- root-specific SC-NEVPT2
- ground-manifold SA4 and SA2 diagnostics

Important scientific conclusion:

The low FeH electron affinity obtained from the CASSCF/SC-NEVPT2
branch survives the active-space balancing, CASCI convergence repair,
and SA4 -> SA2 ground-manifold test. It is therefore not treated as
a simple numerical/root-tracking failure.

Representative validated values:

- SA4 CASSCF electronic EA: ~0.3940 eV
- SA4 SC-NEVPT2 electronic EA: ~0.2512 eV
- SA2 CASSCF electronic EA: ~0.3983 eV
- SA2 SC-NEVPT2 electronic EA: ~0.2976 eV

These pilot files are validation material, not yet the generic
production workflow.
