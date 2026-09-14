# Experiment 06: one-dimensional continuation through the S2 Lax family

The starting point is a converged coefficient vector produced by Experiment 05. The continuation code never substitutes the analytic equations a=c=1 or bd=1 into the predictor, tangent calculation, or corrector.

- Total corrected points: 461
- Points toward increasing |b|, including the start: 226
- Points toward decreasing |b|, including the start: 236
- Coverage in |b|: 0.079843 to 12.041551
- Coverage in |d|: 0.083044 to 12.524212
- Maximum component-normalized flatness residual: 9.985550e-11
- Maximum |a-1|: 2.286260e-05
- Maximum |c-1|: 1.827969e-05
- Maximum |bd-1|: 4.374041e-05
- Minimum Jacobian spectral gap s6/s7 along the stored path: 4.389357e+04

## Interpretation

The signed arclength is a reproducible coordinate on this particular one-dimensional path through the two-real-dimensional flat-connection solution manifold. It should not be identified with a unique or canonical complex spectral parameter. Different initial tangent choices can trace different one-dimensional curves in the same complex-one-dimensional family.

## Collaborator-note reference

The collaborator note reports a predictor-corrector path with about 1200 accepted steps, coverage |b| approximately 0.05 to 18.7, component-normalized residual at or below 1e-10, frozen coefficients stable to about 1.5e-5, and |bd-1| at or below about 6.6e-5. It also reports a numerical fold near the large-|b| end caused by tangential drift of the Adam corrector. The present validation deliberately stops at |b|=12 so that the clean monotonic region can be tested before studying that fold separately.
