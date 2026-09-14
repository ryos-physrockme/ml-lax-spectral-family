# Experiment 09: an immersed neural chart with flatness-prioritized training

## Reason for this experiment

Experiments 07 and 08 separated two requirements that a neural representation of a Lax family must satisfy.

1. The map from two real chart coordinates to the Lax coefficients must have Jacobian rank two.  Otherwise two nominal input coordinates can describe only a one-dimensional curve.
2. The image must remain on the flat-connection solution manifold.  A rank-two map with a small but nonzero flatness residual is not sufficient if it creates its second direction by leaving that manifold.

Experiment 08 showed both effects explicitly.  A strong angle barrier produced a rank-two map, but the learned coefficients departed substantially from the known analytic family.  Directly anchoring the second numerical null direction fixed the local tangent geometry on the continuation path, but the final optimization iterate still traded flatness against the auxiliary geometric losses.

There is also a geometric reason not to impose unit derivative norms everywhere.  For the known analytic family

\[
a=c=1,\qquad b=e^{\rho+i\phi},\qquad d=e^{-\rho-i\phi},
\]

the Euclidean metric inherited from the coefficient space is

\[
\mathrm ds^2_{
\mathrm{coeff}}
=2\cosh(2\rho)\left(\mathrm d\rho^2+\mathrm d\phi^2\right).
\]

Its scale varies along the family.  Requiring two globally unit coordinate derivatives therefore introduces an unnecessary coordinate-dependent bias.  A chart only needs a nonsingular Jacobian, not a prescribed Euclidean metric.

## Training map

The neural network is

\[
N:(s,t)\in\mathbb R^2\longmapsto
(a,b,c,d)\in\mathbb C^4.
\]

The coordinate \(s\) is the signed pseudo-arclength of the one-dimensional path obtained in Experiment 06.  The line \(t=0\) is anchored to that path.  The coordinate \(t\) labels the second real direction.  Neither coordinate is identified with a unique or canonical spectral parameter.

At every anchor point the two-dimensional null space of the flatness-residual Jacobian is recomputed numerically.  The first target tangent is the pseudo-arclength tangent projected into this null space.  The second target tangent is the orthogonal direction inside the same null space.  The analytic equations \(a=c=1\) and \(bd=1\) are not used to construct either tangent.

## Loss function

The global unit-derivative penalty used in Experiments 07 and 08 is removed.  The loss contains four ingredients.

### Flatness

The component-normalized curvature residual is given the largest weight.  This term constrains the image of the neural network to the flat-connection solution manifold.

### Coefficient anchors

On \(t=0\), the network reproduces the continuation coefficients from Experiment 06.

### Tangent-frame anchors

On the same line, \(\partial_sN\) and \(\partial_tN\) are matched to the two numerical null-space tangent directions.  This fixes the local scale and orientation of the two chart coordinates without prescribing the metric away from the anchor line.

### Mild immersion conditions

Let \(\sigma_{\max}\geq\sigma_{\min}\geq0\) be the two singular values of the real \(8\times2\) chart Jacobian.  Two one-sided penalties are used:

\[
\max\left(0,r_{\min}-\frac{\sigma_{\min}}{\sigma_{\max}}\right)^2,
\qquad
\max\left(0,\sigma_{\mathrm{floor}}-\sigma_{\min}\right)^2.
\]

They only prevent a degenerate map.  They do not force the two singular values to be equal or to remain equal to one.

## Coordinate-width curriculum

The transverse interval is enlarged during training.  The network first learns a narrow neighborhood of the anchor path, then the sampled interval in \(t\) is widened.  This makes the calculation closer to continuation of a local chart than to learning the full finite strip in one step.

## Model selection

The last optimization iterate is not used automatically.  A fixed validation set, independent of the stochastic training batches, is evaluated at regular intervals.  The stored network is the iterate with the smallest validation objective constructed from flatness, coefficient anchoring, tangent-frame anchoring, and the two mild immersion penalties.  The known analytic relations are not used for model selection.

## Post-training checks

After model selection, the following are evaluated on a dense two-dimensional grid:

- component-normalized flatness residual;
- both singular values of the chart Jacobian and their ratio;
- alignment with the two numerical tangent directions on \(t=0\);
- the known relations \(a=1\), \(c=1\), and \(bd=1\), used only as external validation.
