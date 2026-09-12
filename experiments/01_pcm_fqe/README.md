# Experiment 01: PCM `F = Q E` benchmark

## Question

Can a neural network recover the map from the principal-chiral-model equation-of-motion residual to the Lax curvature without being told the analytic form or even that the map is proportional to the identity?

## Data construction

The experiment works off shell with respect to the dynamics but on the Maurer--Cartan constraint surface. For each complex spectral parameter `lambda`, random coefficients for `J_+`, `J_-`, and the equation-of-motion residual

\[
E=\partial_+J_-+\partial_-J_+
\]

are drawn. With `C=[J_+,J_-]`, the derivative jets are set to

\[
\partial_+J_- = \frac{E-C}{2},\qquad
\partial_-J_+ = \frac{E+C}{2},
\]

so that

\[
\partial_+J_- - \partial_-J_+ + [J_+,J_-]=0
\]

holds exactly while `E` remains nonzero and unconstrained.

For the standard PCM Lax family,

\[
a=(1-\lambda)^{-1},\qquad c=(1+\lambda)^{-1},
\]

the target relation is

\[
F_{+-}=Q(\lambda)E,
\qquad
Q(\lambda)=-\frac{\lambda}{1-\lambda^2}I_3.
\]

The analytic expression for `Q` is used only for evaluation, never in the training loss.

## Learner

`ComplexMatrixNet` maps `(Re lambda, Im lambda)` to a completely general complex `3 x 3` matrix. The loss is the batch-normalized squared residual

\[
\mathcal L_Q=
\frac{\langle\|F-QE\|^2\rangle}
{\langle\|F\|^2\rangle+\epsilon}.
\]

No penalty enforces diagonality, equality of diagonal entries, or the analytic rational dependence on `lambda`.

## Diagnostics

The run records

1. held-out relative `F-QE` residual;
2. relative error of the learned matrix against the known analytic `Q(lambda)`;
3. off-diagonal norm fraction;
4. spread of the three diagonal entries;
5. singular values at selected spectral parameters.

The point `lambda=0` is a deliberately included degeneracy. There `Q=0` because `L_+=J_+`, `L_-=J_-` is already flat by the Maurer--Cartan identity. At generic `lambda`, the exact `Q` has rank three. A vertically stacked collection of exact `Q(lambda_i)` matrices has rank three as soon as at least one nonzero spectral parameter is included.

This benchmark therefore checks both a positive full-rank case and a known rank-deficient point before the same diagnostic is applied to less trivial Lax candidates.

## Run

From the repository root:

```bash
pip install -e '.[train,dev]'
pytest
python experiments/01_pcm_fqe/train.py --config experiments/01_pcm_fqe/config.yaml
```

Outputs are written below `results/01_pcm_fqe/` and are not committed.
