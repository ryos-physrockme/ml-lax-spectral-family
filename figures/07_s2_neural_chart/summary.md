# Experiment 07: neural chart of the S2 flat-connection family

The network maps two real coordinates `(s,t)` to the eight real components of four complex coefficients `(a,b,c,d)`.  The line `t=0` is anchored to the one-dimensional pseudo-arclength continuation from Experiment 06.  The analytic relations `a=c=1` and `bd=1` are used only for post-training validation.

Two losses were compared.  The first reproduces the collaborator-note idea of constraining the norms of both coordinate derivatives.  The second adds a weak penalty when the two derivatives become nearly parallel.

## Derivative-norm baseline

- Anchor RMSE: 3.411e-3
- Median component-normalized flatness residual: 8.351e-5
- Maximum component-normalized flatness residual: 1.081e-3
- Median chart-Jacobian singular-value ratio sigma_min/sigma_max: 2.416e-3
- Minimum singular-value ratio: 1.604e-4
- Fraction of the evaluation grid with singular-value ratio below 0.05: 1.000
- Median squared sine of the angle between the two coordinate tangent vectors: 2.335e-5
- Maximum |a-1|: 6.627e-3
- Maximum |c-1|: 6.644e-3
- Maximum |bd-1|: 1.295e-1

The map fits the anchor path and has a small flatness residual, but its two coordinate tangent vectors are almost parallel over the entire evaluation grid.  Therefore the derivative-norm condition does not establish that the network gives a two-real-dimensional chart.

## Weak Jacobian-rank penalty

- Anchor RMSE: 3.873e-3
- Median component-normalized flatness residual: 5.628e-5
- Maximum component-normalized flatness residual: 2.923e-3
- Median chart-Jacobian singular-value ratio: 6.940e-3
- Minimum singular-value ratio: 8.243e-4
- Fraction of the evaluation grid with singular-value ratio below 0.05: 0.971
- Median squared sine of the tangent-vector angle: 1.927e-4
- Maximum |a-1|: 1.642e-2
- Maximum |c-1|: 1.309e-2
- Maximum |bd-1|: 1.517e-1

The additional penalty improves the rank diagnostic but does not remove the near-rank-one collapse.  The calculation therefore motivates a stronger treatment of the second tangent direction rather than validating the original charting prescription.
