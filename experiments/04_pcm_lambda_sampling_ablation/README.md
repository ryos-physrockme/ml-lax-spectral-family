# Experiment 04: spectral-parameter sampling cadence in principal-chiral-model learning

## Question

The adaptive loss in the collaborator note keeps an exponential moving average of the loss associated with sampled spectral-parameter values. The note specifies uniform sampling of the spectral parameter and lists `--n-lambda 64` as the “Number of lambda samples per step”. This wording suggests that the spectral-parameter sample may be redrawn during optimization, but the note does not explicitly state the redraw rule. At the same time, the adaptive prescription labels its exponential moving average by `lambda_n`, which has the clearest pointwise interpretation when `lambda_n` denotes a persistent spectral-parameter value.

This experiment therefore does not silently choose between those readings. It measures both conventions and records whether the comparison among the three loss constructions depends on that implementation detail.

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

For the adaptive loss, the exponential moving average attached to batch index `n` is then not the history of one fixed point in the spectral-parameter plane: the value of `lambda_n` changes between steps. The moving average is consequently an average attached to a batch slot rather than to a fixed spectral point.

The phrase “64 spectral-parameter samples per step” in the collaborator note makes this convention plausible, but it is not sufficient to establish the implementation of the original code. The ablation is therefore a test of an unresolved implementation detail, not a claim that either convention is the collaborator’s actual code.

## Compared losses

Both sampling conventions are tested with

1. the bare flatness loss;
2. the analytic weight proportional to `|lambda - 1/2|^{-2}`;
3. component-norm normalization combined with adaptive spectral-parameter weighting.

The main evaluation quantities after 10000 steps are the relative `L2` error of `c(lambda)` and the maximum spectral-curve residual `|a+c-2ac|` over a `200 x 200` complex grid.

## Interpretation rule

A conclusion about the merit of a loss construction is considered robust to this ambiguity only if it persists under both sampling conventions. If the ordering changes, the result is recorded as an implementation sensitivity rather than attributed to the physical model or to the loss construction alone.
