# Experiment 05: discovering the S2 Lax family from the flatness-residual Jacobian

## Purpose

This experiment independently reproduces the local-discovery stage of N. Tanahashi, *Machine-Learning the Spectral Parameter for the S2 Coset: Prior-Free Discovery, Tracing, and Charting of the Lax Family* (June 9, 2026).

The target result is not supplied to the optimization. The calculation asks whether low-flatness solutions form an isolated set or a continuous family, how many real directions the family has, which Lax coefficients remain fixed along it, and whether a conserved relation between the moving coefficients can be inferred from the tangent vectors.

## Physical setup

Use the symmetric coset

\[
S^2=SU(2)/U(1)
\]

and split the left-invariant current according to the symmetric-space grading,

\[
J_\pm=J_\pm^{(0)}+J_\pm^{(1)}.
\]

The candidate Lax connection is

\[
L_+=aJ_+^{(0)}+bJ_+^{(1)},
\qquad
L_-=cJ_-^{(0)}+dJ_-^{(1)},
\]

where `a,b,c,d` are unrestricted complex numbers during the landing stage.

The calculation uses the standard spherical coordinates `(theta,phi)`. The on-shell relations are

\[
\partial_+\partial_-\theta
=\sin\theta\cos\theta\,\partial_+\phi\,\partial_-\phi,
\]

\[
\sin\theta\,\partial_+\partial_-\phi
+\cos\theta\left(
\partial_+\theta\,\partial_-\phi
+\partial_-\theta\,\partial_+\phi
\right)=0.
\]

Random first derivatives are sampled, and the mixed second derivatives are fixed by these equations. The relations `a=c=1` and `bd=1` are not used to construct the samples or the loss.

## Component-normalized flatness loss

Write

\[
F_{+-}=\partial_+L_- - \partial_-L_+ + [L_+,L_-].
\]

The bare flatness loss has a trivial zero at the zero connection. Following the collaborator note, the landing stage instead minimizes

\[
\mathcal L_{\rm cn}
=\left\langle
\frac{\|F_{+-}\|^2}
{\|\partial_+L_-\|^2+\|\partial_-L_+\|^2+\|[L_+,L_-]\|^2+10^{-10}}
\right\rangle.
\]

The denominator contains only the three terms already needed to evaluate the curvature and does not contain the known analytic solution.

## Landing calculation

The collaborator note reports 256 random initializations on a fixed batch of 16384 on-shell samples and retains 243 solutions after a data-driven loss cut. The available note does not specify the optimizer, learning rate, number of landing steps, or initialization distribution.

This independent calculation therefore documents its own landing optimization rather than filling those missing details by assumption:

- 256 independent complex initializations;
- 2048 fixed on-shell samples for the landing optimization;
- Adam with learning rate `0.03`;
- 800 optimization steps;
- Gaussian coefficient initialization with standard deviation `0.8` for each real and imaginary component;
- convergence criterion `component-normalized loss < 1e-8`.

The smaller landing batch is used to keep the validation calculation inexpensive. The subsequent Jacobian diagnosis uses 16384 fresh on-shell samples, matching the diagnostic sample count in the collaborator note.

## Jacobian diagnosis

At every converged point, the vector-valued flatness residual is differentiated with respect to the eight real coordinates

\[
(\operatorname{Re}a,\operatorname{Re}b,\operatorname{Re}c,\operatorname{Re}d,
\operatorname{Im}a,\operatorname{Im}b,\operatorname{Im}c,\operatorname{Im}d).
\]

The eight Jacobian columns are normalized to unit Euclidean norm before the singular-value decomposition. If the flat solution set is locally one complex dimensional, the realified Jacobian should have two null directions.

For each numerical null vector, the four complex coefficient variations are reconstructed. The components along `a` and `c` test whether those coefficients are frozen. For the moving coefficients, we form

\[
\delta\log b=\frac{\delta b}{b},
\qquad
\delta\log d=\frac{\delta d}{d}.
\]

A monomial `b^p d^q` is constant along the family if

\[
p\,\delta\log b+q\,\delta\log d=0
\]

for every tangent vector. Stacking the real and imaginary parts of the measured tangent changes and taking the null vector of the resulting two-column matrix gives the exponent pair `(p,q)` without assuming `bd` in advance.

## Additional point-cloud checks

Independently of the tangent analysis, the converged coefficients are tested by fitting

\[
\log|d| = m\log|b|+k
\]

and

\[
\arg d = m_\varphi\arg b+k_\varphi.
\]

If the inferred relation is `bd=constant`, both slopes should be close to `-1`. The common product itself is then evaluated as an independent numerical check.
