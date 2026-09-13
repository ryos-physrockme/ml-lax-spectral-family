# Experiment 06: one-dimensional continuation through the S2 Lax family

The starting point is a converged coefficient vector produced by Experiment 05. The continuation code never substitutes the analytic equations a=c=1 or bd=1 into the predictor, tangent calculation, or corrector.

- Total corrected points: 1
- Points toward increasing |b|, including the start: 1
- Points toward decreasing |b|, including the start: 1
- Coverage in |b|: 1.001278 to 1.001278
- Coverage in |d|: 0.998724 to 0.998724
- Maximum component-normalized flatness residual: 1.627348e-17
- Maximum |a-1|: 2.326655e-13
- Maximum |c-1|: 2.414183e-13
- Maximum |bd-1|: 8.555993e-09
- Minimum Jacobian spectral gap s6/s7 along the stored path: 1.903546e+12

## Interpretation

The signed arclength is a reproducible coordinate on this particular one-dimensional path through the two-real-dimensional flat-connection solution manifold. It should not be identified with a unique or canonical complex spectral parameter. Different initial tangent choices can trace different one-dimensional curves in the same complex-one-dimensional family.

## Collaborator-note reference

The collaborator note reports a predictor-corrector path with about 1200 accepted steps, coverage |b| approximately 0.05 to 18.7, component-normalized residual at or below 1e-10, frozen coefficients stable to about 1.5e-5, and |bd-1| at or below about 6.6e-5. It also reports a numerical fold near the large-|b| end caused by tangential drift of the Adam corrector. The present validation deliberately stops at |b|=12 so that the clean monotonic region can be tested before studying that fold separately.
