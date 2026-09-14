# Experiment 08: recovering the second real tangent direction of the S2 Lax family

## Motivation

Experiment 07 tested a neural-network map

\[
N:(s,t)\in\mathbb R^2\longmapsto
(a,b,c,d)\in\mathbb C^4
\]

for the flat Lax-connection family of the two-dimensional sigma model with target space
\(S^2=SU(2)/U(1)\).  The line \(t=0\) was anchored to the pseudo-arclength continuation obtained in Experiment 06.

The derivative-norm conditions

\[
\|\partial_s N\|^2\simeq1,
\qquad
\|\partial_t N\|^2\simeq1
\]

did not make the image two-real-dimensional.  On the evaluation grid the two derivatives became almost parallel: the median ratio of the smaller to the larger singular value of the chart Jacobian was approximately \(2.4\times10^{-3}\).  A weak penalty on the angle between the derivatives improved this ratio only to approximately \(6.9\times10^{-3}\).  Therefore the calculation in Experiment 07 does not yet demonstrate a two-real-dimensional chart.

This experiment tests two stronger ways of preventing that collapse.

## Flatness-residual null space

For the coefficient vector

\[
x=(\operatorname{Re}a,\operatorname{Re}b,
   \operatorname{Re}c,\operatorname{Re}d,
   \operatorname{Im}a,\operatorname{Im}b,
   \operatorname{Im}c,\operatorname{Im}d)\in\mathbb R^8,
\]

let \(R(x)\) denote the realified flatness residual evaluated on a fixed set of on-shell field samples.  Experiment 05 found that the Jacobian

\[
J_R=\frac{\partial R}{\partial x}
\]

has a two-dimensional real null space at generic points of the learned flat-connection family.  This null space is the numerically identified tangent plane of the flat solution manifold inside the coefficient space.

At each anchor point on the continuation path, the first unit tangent vector is obtained by differentiating the anchor coefficients with respect to the signed pseudo-arclength \(s\) and projecting the result into \(\ker J_R\).  A second unit vector is then chosen inside \(\ker J_R\), orthogonal to the first.  Its sign is fixed continuously along the anchor path.  No analytic use is made of the known relations \(a=c=1\) or \(bd=1\) in this construction.

## Variant 1: logarithmic rank barrier

The first variant keeps the coefficient anchors and derivative-norm loss of Experiment 07, but replaces the weak threshold penalty by the scale-independent barrier

\[
\mathcal L_{\mathrm{barrier}}
=-\left\langle
\log\!\left(
\frac{\det G}{G_{ss}G_{tt}}+\epsilon
\right)
\right\rangle,
\]

where

\[
G=J_N^T J_N,
\qquad
J_N=(\partial_sN,\partial_tN).
\]

The ratio inside the logarithm is the squared sine of the angle between the two coordinate tangent vectors.  It approaches zero when the neural map loses rank.

## Variant 2: null-space tangent-frame anchoring

The second variant uses the numerically determined two-dimensional tangent plane directly.  On the line \(t=0\), in addition to matching the coefficients, it minimizes

\[
\mathcal L_{\mathrm{frame}}
=
\left\langle
\|\partial_sN-v_s\|^2
+
\|\partial_tN-v_t\|^2
\right\rangle,
\]

where \(v_s\) is the unit continuation tangent and \(v_t\) is the orthogonal unit vector in the Jacobian null space.  A small logarithmic rank barrier is retained away from the anchor line.

This is a stronger condition than prescribing two derivative norms: it specifies two independent tangent directions known numerically to preserve flatness to first order.

## Validation

The analytic family relations

\[
a=1,\qquad c=1,\qquad bd=1
\]

are used only after training.  The main diagnostics are

- component-normalized flatness residual on a two-dimensional grid;
- singular values of the chart Jacobian \(J_N\);
- squared sine of the angle between \(\partial_sN\) and \(\partial_tN\);
- alignment of the two learned tangent vectors with the numerical tangent frame on \(t=0\);
- deviations from \(a=1\), \(c=1\), and \(bd=1\).

The purpose is not to declare the chart coordinate \(s+it\) a canonical spectral parameter.  It is to test whether the learned map actually covers a two-real-dimensional patch of the flat-connection family.
