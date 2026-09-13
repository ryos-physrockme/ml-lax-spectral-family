# Experiment 04: spectral-parameter sampling cadence in principal-chiral-model learning

## Question

The adaptive loss in the collaborator note keeps an exponential moving average of the loss associated with sampled spectral-parameter values. The note specifies that the spectral parameter is sampled uniformly, but it does not state whether the sampled spectral-parameter batch is kept fixed across optimization steps or redrawn at every step.

This experiment measures whether that implementation choice changes the comparison among the three loss constructions studied in Experiment 03.

## Common physical setup

For the `SU(2)` principal chiral model, use

\[
L_+=a(\lambda)J_+,
\qquad
L_-=c(\lambda)J_-,
\]

with the spectral-parameter convention

\[
a(\lambda)=\lambda.
\]

On shell,

\[
F_{+-}
=-\frac{a+c-2ac}{2}[J_+,J_-],
\]

and the exact coefficient is

\[
c_{\rm exact}(\lambda)=\frac{\lambda}{2\lambda-1}.
\]

The neural network learns the complex function `c(lambda)` from flatness alone. The analytic expression is used only for evaluation.

The network, optimizer, spectral-parameter domain, field sampling, and number of training steps are the same as in Experiment 03.

## Sampling conventions

### Fixed spectral-parameter batch

At the beginning of a run, 64 complex spectral-parameter values are sampled uniformly from

\[
-2\leq \operatorname{Re}\lambda\leq0,
\qquad
-1\leq \operatorname{Im}\lambda\leq1.
\]

Those values are kept fixed for all 10000 optimization steps. Field-current samples are redrawn at every step.

For the adaptive loss, the exponential moving average attached to batch index `n` is therefore the history of one fixed spectral-parameter value `lambda_n`.

### Resampled spectral-parameter batch

A new set of 64 complex spectral-parameter values is drawn at every optimization step. Field-current samples are also redrawn.

For the adaptive loss, the exponential moving average attached to batch index `n` is then not the history of one fixed point in the spectral-parameter plane: the value of `lambda_n` changes between steps. This convention is included because it is a natural alternative reading of “sample lambda uniformly”, and because it approximately reproduces the numerical ordering reported in the collaborator note in an earlier implementation.

It should not be interpreted as the known convention of the collaborator code. The available note does not determine that point.

## Compared losses

Both sampling conventions are tested with

1. the bare flatness loss;
2. the analytic weight proportional to `|lambda - 1/2|^{-2}`;
3. component-norm normalization combined with adaptive spectral-parameter weighting.

The main evaluation quantities after 10000 steps are the relative `L2` error of `c(lambda)` and the maximum spectral-curve residual `|a+c-2ac|` over a `200 x 200` complex grid.

## Interpretation rule

A conclusion about the merit of a loss construction is considered robust to this ambiguity only if it persists under both sampling conventions. If the ordering changes, the result is recorded as an implementation sensitivity rather than attributed to the physical model or to the loss construction alone.
