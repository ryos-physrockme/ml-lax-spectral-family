# Experiment 02: off-shell factorization in the S^2 = SU(2)/U(1) sigma model

## Question

Can a neural network recover the linear map from the two independent equation-of-motion residuals of the S^2 sigma model to the three Lie-algebra components of the Lax curvature, without being told the sparse analytic matrix structure?

## Coordinate equations of motion

Use spherical coordinates `theta` and `phi`. Define

\[
E_\theta
=
\partial_+\partial_-\theta
-\sin\theta\cos\theta\,\partial_+\phi\,\partial_-\phi,
\]

and

\[
E_\phi
=
\sin\theta\,\partial_+\partial_-\phi
+\cos\theta\left(
\partial_+\theta\,\partial_-\phi
+\partial_-\theta\,\partial_+\phi
\right).
\]

The field equations are `E_theta = E_phi = 0`. Training samples keep these two residuals nonzero and independent.

## Lax connection

For the symmetric-coset decomposition

\[
J_\pm=J_\pm^{(0)}+J_\pm^{(1)},
\]

use the standard one-complex-parameter family

\[
L_+(\lambda)=J_+^{(0)}+\lambda J_+^{(1)},
\qquad
L_-(\lambda)=J_-^{(0)}+\lambda^{-1}J_-^{(1)}.
\]

Writing the curvature in a Lie-algebra basis `(T1,T2,T3)`, direct substitution gives

\[
F^1=(\lambda^{-1}-\lambda)E_\phi,
\qquad
F^2=(\lambda-\lambda^{-1})E_\theta,
\qquad
F^3=0.
\]

Thus, for the residual vector ordered as `(E_phi,E_theta)`, the exact linear map is

\[
Q(\lambda)=
\begin{pmatrix}
\lambda^{-1}-\lambda & 0\\
0 & \lambda-\lambda^{-1}\\
0 & 0
\end{pmatrix}.
\]

This analytic matrix is used only after training for validation.

## Off-shell sampling

The first derivatives of `theta` and `phi`, together with `E_theta` and `E_phi`, are sampled independently. The mixed second derivatives are then defined so that they realize those chosen residuals. The polar regions where `sin(theta)` is close to zero are excluded to avoid a coordinate singularity.

The spectral-parameter domain is

\[
0.5\leq \operatorname{Re}\lambda\leq1.5,
\qquad
-0.5\leq \operatorname{Im}\lambda\leq0.5,
\]

which avoids the pole at `lambda = 0` while retaining the special point `lambda = 1`. At `lambda = 1`, the exact map `Q` vanishes, so the Lax curvature is flat independently of the equations of motion. This provides a known rank-deficient control analogous to `lambda = 0` in the principal chiral model benchmark.

## Learner

The neural network maps `(Re lambda, Im lambda)` to a completely general complex `3 x 2` matrix. The training objective is

\[
\mathcal L_Q=
\frac{\langle\|F-QE\|^2\rangle}
{\langle\|F\|^2\rangle+10^{-30}}.
\]

No penalty enforces the zero third row, the two off-diagonal zeros, or the relation `Q[0,0] = -Q[1,1]`.

## Diagnostics

The experiment measures the held-out `F = Q E` residual, the relative matrix error against the analytic map, the fraction of learned matrix norm in analytically forbidden entries, the violation of `Q[0,0] + Q[1,1] = 0`, and the two singular values as functions of the spectral parameter.
