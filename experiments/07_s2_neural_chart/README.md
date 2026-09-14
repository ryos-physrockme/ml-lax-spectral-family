# Experiment 07: neural chart of the S2 Lax-connection family

## Purpose

This experiment tests whether a neural network can represent a two-real-dimensional patch of the flat Lax-connection family of the two-dimensional sigma model with target space

\[
S^2 = SU(2)/U(1).
\]

The experiment starts from the coefficient ansatz

\[
L_+ = a J_+^{(0)} + b J_+^{(1)},\qquad
L_- = c J_-^{(0)} + d J_-^{(1)},
\]

where \(a,b,c,d\in\mathbb C\).  The analytic relations \(a=c=1\) and \(bd=1\) are **not** supplied to the neural network.  They are used only after training as diagnostic quantities.

Experiment 05 identified a two-real-dimensional null space of the flatness-residual Jacobian.  Experiment 06 followed one real one-dimensional path inside that solution manifold by pseudo-arclength continuation.  The present experiment uses points on that path as anchors and asks a neural network to extend them to a two-real-dimensional map.

## Chart coordinate

The neural network takes two real inputs \((s,t)\), combined as a complex chart coordinate

\[
z=s+it.
\]

The coordinate \(s\) is fixed on the line \(t=0\) by the signed arclength used in Experiment 06.  Therefore \(s\) is a reproducible coordinate on the chosen continuation path, but it is not assumed to be a canonical spectral parameter.  The transverse coordinate \(t\) is learned from flatness and derivative constraints.

The network returns eight real numbers, interpreted as the real and imaginary parts of

\[
(a(s,t),b(s,t),c(s,t),d(s,t))\in\mathbb C^4.
\]

## Loss terms

For a batch of chart points, the first term is the component-normalized flatness loss used in Experiments 05 and 06.

The second term is an anchor loss.  Points from the pseudo-arclength continuation are placed at \(t=0\), and the network output is required to reproduce their four complex coefficients.

The third term prevents a coordinate-independent map.  If \(N(s,t)\in\mathbb R^8\) denotes the real network output, it penalizes deviations of

\[
\left\|\partial_s N\right\|^2,\qquad
\left\|\partial_t N\right\|^2
\]

from one.  Since \(s\) is an arclength coordinate on the anchor line, the first condition has a direct geometric interpretation there.

Two training variants are compared.

1. **Derivative-norm baseline.** Flatness, anchors, and the two derivative-norm conditions are used.  This reproduces the structure of the charting calculation in the collaborator note.
2. **Jacobian-rank-aware variant.** The same loss is supplemented by a penalty when \(\partial_s N\) and \(\partial_t N\) become nearly parallel.  This addresses a logical gap in the derivative-norm condition: two nonzero derivatives can both have unit norm while spanning only one real direction.

For the second variant, define the \(8\times2\) chart Jacobian

\[
J_N=(\partial_s N,\partial_t N)
\]

and its Gram matrix \(G=J_N^T J_N\).  The quantity

\[
\frac{\det G}{G_{ss}G_{tt}}
\]

is the squared sine of the angle between the two coordinate tangent vectors.  It vanishes when the chart loses rank.  The additional loss penalizes values below a fixed positive threshold.

## Quantities used only for validation

After training, the following quantities are measured on a two-dimensional grid in \((s,t)\):

- component-normalized flatness residual;
- anchor reconstruction error on \(t=0\);
- the two singular values of \(J_N\) and their ratio;
- \(|a-1|\), \(|c-1|\), and \(|bd-1|\).

The last three quantities compare the learned chart with the known analytic family, but they are not training targets.

## Inputs and outputs

The workflow first reruns Experiment 06 so that the continuation trace and its provenance are generated in the same GitHub Actions job.  The chart-training script then reads

`results/06_s2_pseudo_arclength_continuation/trace.json`.

Numerical outputs are written to

`results/07_s2_neural_chart/`,

and publication-oriented diagnostic figures are written to

`figures/07_s2_neural_chart/`.
