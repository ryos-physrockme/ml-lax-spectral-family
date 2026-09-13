# Experiment 05: S2 flat-family discovery from the Jacobian null space

This is an independent reproduction of the discovery stage in the collaborator note. The relations a=c=1 and bd=1 are not imposed in the landing loss or in the Jacobian analysis.

- Random initializations: 256
- Converged initializations: 252
- Points used for the tangent-invariant analysis: 252
- Landing batch size: 2048
- Jacobian diagnostic batch size: 16384
- Nullity histogram: {'2': 252}
- Median ratio s6/s7: 8.800915e+14
- Maximum |delta a| in the two numerical null vectors: 2.432376e-10
- Maximum |delta c| in the two numerical null vectors: 2.415865e-10
- Median |a-1|: 4.441320e-16
- Median |c-1|: 4.372758e-16
- Invariant exponents normalized to p=1: (1.000000, 1.000000)
- Singular-value ratio in the invariant fit: 8.532739e+10
- Slope log|d| versus log|b|: -1.000000
- Slope arg(d) versus arg(b): -1.000000
- Maximum |bd-1|: 1.931505e-07
- Sampled family coverage in |b|: 0.2153 to 4.6409

## Collaborator-note reference

The collaborator note used 256 random initializations on a fixed batch of 16384 on-shell samples and retained 243 converged points. It reported nullity two at every retained point, a Jacobian singular-value gap of about 1e13--1e15, frozen a and c directions, invariant exponents (1,1), slopes -1 for both modulus and phase relations, and bd=1 to about 5e-6.

The landing optimizer and its hyperparameters are not specified in the available note. This reproduction therefore uses a smaller landing batch and an explicitly documented Adam optimization, while keeping the Jacobian diagnostic batch at 16384 samples. Agreement should be judged by the recovered manifold dimension and algebraic structure, not by the exact number of converged initializations.

The singular-value decomposition is performed after unit-normalizing the Jacobian columns for conditioning. Before interpreting a numerical null vector as a tangent in the original coefficient coordinates, the components must be divided by the corresponding pre-normalization column norms. The first implementation omitted this inverse coordinate rescaling; the corrected analysis above restores the exact exponent pair (1,1) to numerical precision.
