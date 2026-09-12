# ml-lax-spectral-family

Numerical and machine-learning experiments for discovering and validating spectral-parameter families of Lax connections.

This repository follows `ml-lax-nlsm`. The emphasis here is not only on finding low-flatness Lax candidates, but on resolving the structure of a Lax family and testing whether its curvature actually encodes the equations of motion.

## Current experiment

### Experiment 01: PCM off-shell factorization `F = Q E`

For the principal chiral model (PCM), define the equation-of-motion residual

\[
E := \partial_+J_-+\partial_-J_+
\]

and impose the Maurer--Cartan identity

\[
M := \partial_+J_- - \partial_-J_+ + [J_+,J_-]=0
\]

on the sampled jets. For

\[
L_+=a(\lambda)J_+,\qquad L_-=c(\lambda)J_-,
\]

the curvature is

\[
F_{+-}=c\,\partial_+J_- - a\,\partial_-J_+ + ac[J_+,J_-].
\]

On the PCM spectral curve

\[
a+c-2ac=0,
\]

this factorizes off shell with respect to the equation of motion as

\[
F_{+-}=Q(\lambda)E,\qquad
Q(\lambda)=\frac{c(\lambda)-a(\lambda)}{2}\,\mathbf 1_{\mathfrak{su}(2)}.
\]

For the standard parametrization

\[
a(\lambda)=\frac{1}{1-\lambda},\qquad
c(\lambda)=\frac{1}{1+\lambda},
\]

one has

\[
Q(\lambda)=-\frac{\lambda}{1-\lambda^2}\,\mathbf 1_{\mathfrak{su}(2)}.
\]

The first machine-learning test deliberately does **not** assume that `Q` is proportional to the identity: a neural network receives complex `lambda` and outputs a general complex `3 x 3` matrix. Training minimizes the normalized residual of `F - Q E` on off-shell PCM samples satisfying the Maurer--Cartan identity. The learned matrix is then tested for

- held-out `F = Q E` residual,
- agreement with the analytic `Q(lambda)`,
- suppression of off-diagonal entries,
- equality of the three diagonal entries,
- singular values and rank diagnostics at selected spectral parameters.

The special point `lambda = 0` is intentionally retained: there `L_+=J_+`, `L_-=J_-`, so flatness follows from the Maurer--Cartan identity alone and `Q(0)=0`. This is a useful positive control for the rank diagnostic.

## Planned sequence

1. PCM: learn `Q(lambda)` from off-shell data and validate the rank diagnostic.
2. `S^2 = SU(2)/U(1)`: repeat the factorization test for the spectral family.
3. Fake-flatness control: test whether `Q` loses rank when flatness does not encode the full equations of motion.
4. Combine the `Q` diagnostic with Jacobian-null-space analysis, continuation, and explicit spectral-family charting.

## Layout

```text
src/ml_lax_spectral_family/   reusable model and physics code
experiments/01_pcm_fqe/       first validation experiment
tests/                        analytic and numerical sanity checks
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train,dev]'
pytest
python experiments/01_pcm_fqe/train.py --config experiments/01_pcm_fqe/config.yaml
```

The training script writes machine-readable diagnostics to `results/01_pcm_fqe/`.
