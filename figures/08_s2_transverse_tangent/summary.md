# Experiment 08: stronger rank control for the S2 neural chart

Experiment 07 showed that constraining the norms of the two coordinate derivatives does not prevent an almost rank-one map.  Experiment 08 tested two stronger prescriptions on the same coefficient ansatz.  The analytic relations `a=c=1` and `bd=1` were not used in either training loss.

The numerical tangent plane used in the second prescription was obtained from the two-dimensional null space of the flatness-residual Jacobian.  The finite-difference tangent of the pseudo-arclength path lies in that plane with minimum projection cosine 0.999969 and median projection cosine 0.999999999.  The minimum Jacobian singular-value gap `s6/s7` at the anchor points was 6.94e4.

## Logarithmic rank barrier

A scale-independent logarithmic barrier on the squared sine of the angle between the two coordinate tangent vectors removes the rank collapse:

- Median chart-Jacobian singular-value ratio: 0.9693
- Minimum singular-value ratio: 0.7034
- Median squared sine of the tangent-vector angle: 0.999833
- Minimum squared sine: 0.89797
- Fraction of the evaluation grid with singular-value ratio below 0.05: 0

However, rank recovery alone is not sufficient.  The held-out component-normalized flatness residual has median 1.02e-3 and maximum 2.32e-2, while the post-training analytic checks give maximum deviations `|a-1|=0.482`, `|c-1|=0.603`, and `|bd-1|=0.739`.  Thus the network obtains a genuinely two-dimensional image partly by leaving the exact flat-connection family.

## Null-space tangent-frame anchoring

The second variant additionally constrains the derivatives on the anchor line to agree with two independent unit vectors in the numerically determined flatness-Jacobian null space.  On the anchor line the learned tangent directions agree closely with the targets:

- Median cosine for the pseudo-arclength tangent: 0.999899
- Minimum cosine for the pseudo-arclength tangent: 0.998730
- Median cosine for the transverse tangent: 0.999682
- Minimum cosine for the transverse tangent: 0.998327
- Median chart-Jacobian singular-value ratio on the anchor line: 0.9711

The two-dimensionality is also retained away from the anchor line: the median singular-value ratio on the evaluation grid is 0.9521 and its minimum is 0.6282.  Nevertheless the final stored network still has median held-out flatness residual 9.46e-3 and substantial violations of the analytic family relations.  The last optimization step also exhibits a stochastic loss spike, so the final iterate is not a reliable model-selection rule.

## Consequence

These results separate two issues that were conflated in the original neural-chart prescription.  First, an immersion condition requires control of the rank of the chart Jacobian, not only the norms of its columns.  Second, enforcing an almost orthonormal pair of derivatives everywhere can compete with the geometry of the actual solution manifold and with flatness.  The next calculation therefore removes the global unit-derivative condition, retains only a mild immersion condition, anchors the two numerical tangent directions on `t=0`, gives flatness a substantially larger weight, and selects the network from a fixed validation set rather than using the last optimization step.
