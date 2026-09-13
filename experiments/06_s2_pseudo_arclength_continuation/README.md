# Experiment 06: tracing a one-dimensional path through the S2 flat-connection family

## Purpose

Experiment 05 establishes from the flatness-residual Jacobian that the low-flatness solution set has two real tangent directions, corresponding to one complex parameter. This experiment follows one of those tangent directions numerically and tests whether a connected path can be traced between regions of small and large `|b|` without inserting the analytic relations `a=c=1` and `bd=1` into the continuation algorithm.

The calculation is an independent validation of the continuation stage in N. Tanahashi, *Machine-Learning the Spectral Parameter for the S2 Coset: Prior-Free Discovery, Tracing, and Charting of the Lax Family* (June 9, 2026).

## Starting point

The initial coefficient vector is stored in

`figures/05_s2_jacobian_family_discovery/continuation_seed.json`.

It is selected from the converged coefficient cloud of Experiment 05 by minimizing the distance of `|b|` from one among points whose realified flatness-residual Jacobian has a two-dimensional numerical null space. The selection criterion uses only the learned coefficients and the measured Jacobian nullity. The equations `a=c=1` and `bd=1` are not used to construct the seed.

## Tangent calculation

At a current coefficient vector

\[
p=(a,b,c,d)\in\mathbb C^4,
\]

the vector-valued flatness residual is differentiated with respect to the eight real coordinates of `p`. The Jacobian columns are normalized before the singular-value decomposition for numerical conditioning. The two numerical null vectors are then transformed back to the original coefficient coordinates by undoing that column scaling, and they are orthonormalized in the ordinary Euclidean metric on the eight real coefficient coordinates.

At the first point, the gradient of `|b|` is projected into this two-dimensional tangent space. This selects the tangent direction that changes the modulus of `b` most strongly. Both signs are followed: one toward increasing `|b|` and one toward decreasing `|b|`.

At later points, the previous tangent is projected into the new two-dimensional null space and its sign is chosen continuously. This is the numerical continuation analogue of transporting the selected direction along the flat-connection manifold.

## Predictor and corrector

For tangent `t_k` and step length `h`, the predictor is

\[
\widetilde p_{k+1}=p_k+h t_k.
\]

Starting from the predicted point, the four complex coefficients are optimized with Adam to minimize the same component-normalized flatness loss used in Experiment 05. No penalty involving `a=1`, `c=1`, or `bd=1` is used in the corrector.

If the corrected point does not reach a component-normalized loss below `1e-10`, the step length is halved and the predictor-corrector attempt is repeated. The minimum allowed step length is `2e-3`, matching the minimum step length mentioned in the collaborator note. The initial step length is `0.05` and the maximum is `0.08`; these two values are implementation choices because the available note does not specify them.

## Scope of this first continuation validation

The collaborator note reports a path extending to `|b|` about `18.7`, with a numerical fold near the large-`|b|` end caused by tangential drift of the Adam corrector. The present experiment first tests the clean region only, stopping when either

\[
|b|\leq0.08
\]

or

\[
|b|\geq12.
\]

If this range can be traced while retaining flatness and the algebraic relations independently inferred in Experiment 05, a later calculation can deliberately push farther to study the reported fold and compare an unconstrained Adam corrector with a tangentially constrained corrector.

## Interpretation of arclength

The continuation path is one real-dimensional curve inside a solution manifold that Experiment 05 identifies as two real dimensional. The accumulated predictor step length is therefore a reproducible coordinate along this particular path. It is not, by itself, a unique or canonical complex spectral parameter for the entire Lax family.
