# Spin-orbit coupling pilot

Current SOC pilot for FeH / FeH-.

Scalar reference:

- SFX2C-1e
- def2-QZVPPD
- unified CAS(9/10,15)

Planned state-interaction spaces:

Neutral:
- doublet
- quartet
- sextet

Anion:
- triplet
- quintet
- septet

The calculation compares:

1. SOC within the ground-spin multiplicity
2. SOC including adjacent spin multiplicities

The purpose is to obtain an additive differential SOC correction

    Delta EA_SOC = delta E_SOC(FeH) - delta E_SOC(FeH-)

for the current pre-SOC value of about 0.92510 eV.

At the time of this snapshot the SOC calculation is still in progress.
Runtime checkpoints and logs are intentionally not version-controlled.
