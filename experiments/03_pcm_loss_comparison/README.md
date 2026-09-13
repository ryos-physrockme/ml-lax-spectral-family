# Experiment 03: loss construction for explicit spectral-parameter learning in the principal chiral model

## Purpose

This experiment independently reproduces the hard-parametrization comparison in N. Tanahashi, *Explicit Spectral-Parameter Learning in the PCM: Implementation Notes and Numerical Results* (May 24, 2026).

The field theory is the SU(2) principal chiral model. Let

\[
J_\pm=g^{-1}\partial_\pm g
\]

be the right-invariant currents. On solutions of the field equations and the Maurer--Cartan identity,

\[
\partial_+J_-=-\frac12[J_+,J_-],\qquad
\partial_-J_+=+\frac12[J_+,J_-].
\]

We use the separated Lax-connection ansatz

\[
L_+=a(\lambda)J_+,\qquad L_-=c(\lambda)J_-.
\]

Its curvature reduces on shell to

\[
F_{+-}
=-\frac{a+c-2ac}{2}[J_+,J_-].
\]

The flatness condition therefore gives the spectral curve

\[
a+c-2ac=0.
\]

In this experiment the spectral-parameter convention is fixed by

\[
a(\lambda)=\lambda.
\]

The unique target coefficient is then

\[
c_{\rm exact}(\lambda)=\frac{\lambda}{2\lambda-1}.
\]

The neural network receives `(Re lambda, Im lambda)` and learns the complex scalar `c(lambda)`. The analytic target is not used in training.

## Training domain and network

The spectral parameter is sampled uniformly from

\[
-2\leq\operatorname{Re}\lambda\leq0,
\qquad
-1\leq\operatorname{Im}\lambda\leq1,
\]

so the pole at `lambda = 1/2` lies outside the domain. The network has two hidden layers of width 64 with `tanh` activation and uses double precision.

At the beginning of a run, 64 spectral-parameter values are sampled from this rectangle and kept fixed during the 10000 training steps. For every step and for every spectral-parameter value, 64 independent current pairs are freshly sampled. This choice makes the exponential moving average in the adaptive method a well-defined history for each indexed spectral-parameter value. The collaborator note states that the spectral parameter is sampled uniformly and defines an exponential moving average for indexed values `lambda_n`, but it does not explicitly state whether the spectral-parameter sample itself is redrawn at every optimization step. The fixed-sample convention used here is therefore an implementation choice, not a fact inferred from the note.

The optimizer is Adam. The learning rate warms linearly to `2e-3` during the first 150 steps and then follows cosine annealing to `2e-5` at 10000 steps.

## Three loss constructions

### 1. Bare flatness loss

The first objective is

\[
\mathcal L_{\rm bare}
=\mathbb E_{\lambda,J}\|F_{+-}\|^2.
\]

No spectral-parameter-dependent reweighting is used.

### 2. Analytic per-spectral-parameter weighting

With `a(lambda)=lambda`, the gradient strength contains the factor `|1-2 lambda|^2`. The collaborator note compensates this by

\[
w(\lambda)\propto\left|\lambda-\frac12\right|^{-2},
\qquad \langle w\rangle_\lambda=1.
\]

The training objective is the weighted average of the bare flatness loss.

### 3. Component-norm normalization and adaptive spectral-parameter weighting

Write

\[
T_1=\partial_+L_-,\qquad
T_2=\partial_-L_+,\qquad
T_3=[L_+,L_-].
\]

The normalized residual is

\[
\ell(\lambda,J)=
\frac{\|F_{+-}\|^2}
{\|T_1\|^2+\|T_2\|^2+\|T_3\|^2+10^{-10}}.
\]

For each sampled spectral parameter, the field configurations are averaged first. An exponential moving average with coefficient `0.9` tracks this per-spectral-parameter loss. After 100 warmup steps, poorly learned spectral-parameter values receive larger weights. The ratio to the batch mean is clipped to `[1/3,3]`, and the weights are normalized to unit mean.

This construction does not use the analytic spectral curve or the analytic gradient formula.

## Evaluation

The final network is evaluated on a `200 x 200` grid over the complex training domain. We report

- the relative L2 error of `c(lambda)` against `lambda/(2 lambda-1)`;
- the mean of `|a+c-2ac|`;
- the maximum of `|a+c-2ac|`.

The 3000-step values saved by this implementation are checkpoints of the same 10000-step optimization. They are useful diagnostics but are not identical to independent 3000-step runs with a separately scheduled cosine annealing.

## Reproduction boundary

The collaborator note states that smoothness and parameter-L2 regularization terms of weight `1e-4` were present and numerically negligible. It does not specify the smoothness functional in enough detail to reconstruct that term without the original code. This independent implementation therefore reproduces the three central flatness-loss constructions without those small regularizers. This difference must be kept in mind when comparing numerical values run by run.
